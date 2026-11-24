import re
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


LEGAL_ARCHETYPES = {
    "Power of Attorney / Authorization Deed": (
        "Special power of attorney to all whom these presents shall come, we hereby authorise, "
        "empower, appoint and constitute our manager to execute the lease on our behalf and present "
        "the same for registration at the office of the Sub-Registrar."
    ),
    "Property Deed / Indenture": (
        "This indenture witnesseth that the grantor hereby conveys, grants, bargains, "
        "and sells the parcel of land, tenement, and hereditaments unto the grantee forever."
    ),
    "Sworn Affidavit / Deposition": (
        "Personally appeared before me the undersigned magistrate or notary public, the deponent "
        "being duly sworn upon oath deposes and states the following facts to be true."
    ),
    "Court Pleading / Judicial Writ": (
        "In the High Court of Judicature, petition and writ of habeas corpus or mandamus filed by "
        "the plaintiff or petitioner against the defendant respondent praying for judicial relief."
    ),
    "Legislative / Constitutional Provision": (
        "Be it enacted by Parliament that no person shall be deprived of life or personal liberty "
        "except according to procedure established by law, protection against arrest and detention."
    ),
    "Will / Testamentary Instrument": (
        "I hereby declare this to be my last will and testament, revoking all former wills, "
        "appointing executors and bequeathing my estate, chattels, and property."
    ),
}

LEGAL_LEXICON = {
    "procedural_and_judicial": [
        "affidavit", "deponent", "magistrate", "jurisdiction", "plaintiff", "defendant",
        "petitioner", "respondent", "summons", "warrant", "custody", "detention",
        "arrest", "offence", "tribunal", "adjudication", "writ", "habeas corpus",
        "testimony", "oath", "sworn", "decree", "judgment", "appeal", "prosecution"
    ],
    "property_and_contract": [
        "indenture", "deed", "conveyance", "grantor", "grantee", "tenement",
        "hereditaments", "covenant", "consideration", "encumbrance", "mortgage",
        "lease", "power of attorney", "authorise", "empower", "appoint", "constitute", "registration", "registered", "sub-registrar", "director", "bequeath", "executor", "testament", "estate", "proprietor", "title"
    ],
    "formal_legal_connectors": [
        "whereas", "witnesseth", "heretofore", "hereinafter", "aforesaid", "pursuant",
        "notwithstanding", "thereof", "therein", "hereby", "whereof", "stipulated",
        "enacted", "promulgated", "provision", "clause", "schedule", "statute"
    ],
}


def extract_structured_entities(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts dates, statutory citations, named parties, and legal lexicon hits with line provenance."""
    date_pattern = re.compile(
        r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{2,4}"
        r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}"
        r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
        r"|(?:17|18|19|20)\d{2})\b",
        re.IGNORECASE,
    )
    citation_pattern = re.compile(
        r"\b(?:Article|Section|Clause|Schedule|Part|Chapter|Act|Rule|Order)\s+(?:[IVXLCDM]+|\d+[A-Za-z]?(?:\(\d+\))?)\b",
        re.IGNORECASE,
    )
    party_pattern = re.compile(
        r"\b(?:between|against|by|unto|signed\s+by|before)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
    )

    dates = []
    citations = []
    parties = []
    lexicon_hits = {category: {} for category in LEGAL_LEXICON}

    for item in lines:
        line_id = item.get("line_id")
        text = item.get("text", "")
        if not text:
            continue

        for match in date_pattern.findall(text):
            dates.append({"value": match.strip(), "line_id": line_id})

        for match in citation_pattern.findall(text):
            citations.append({"reference": match.strip(), "line_id": line_id})

        for match in party_pattern.findall(text):
            parties.append({"entity": match.strip(), "line_id": line_id})

        lower_text = text.lower()
        for category, terms in LEGAL_LEXICON.items():
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", lower_text):
                    entry = lexicon_hits[category].setdefault(term, {"count": 0, "line_ids": []})
                    entry["count"] += 1
                    if line_id not in entry["line_ids"]:
                        entry["line_ids"].append(line_id)

    return {
        "dates": dates,
        "statutory_and_clause_citations": citations,
        "extracted_parties_and_actors": parties,
        "legal_lexicon_matches": lexicon_hits,
    }


class LegalContextExtractor:
    """
    Generates semantic document context, archetype classification, line salience,
    and pgvector-ready 768-d embeddings using Legal-BERT within a 4 GB VRAM budget.
    """

    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        device: str = None,
        fp16: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fp16 = fp16 and (self.device.type == "cuda")
        print(f"[CONTEXT] Loading Legal-BERT ({model_name}) on {self.device} (FP16: {self.fp16})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if self.fp16:
            self.model.half()

        self.model.to(self.device)
        self.model.eval()

        # Pre-compute archetype prototype embeddings
        self.archetype_labels = list(LEGAL_ARCHETYPES.keys())
        archetype_texts = [LEGAL_ARCHETYPES[k] for k in self.archetype_labels]
        self.archetype_embeddings = self.encode(archetype_texts)

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> np.ndarray:
        """Mean-pooled, L2-normalized 768-d embeddings."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()

        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = (summed / counts).float()

        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().numpy()

    def analyze_document(
        self,
        lines: List[Dict[str, Any]],
        include_full_embedding: bool = False,
    ) -> Dict[str, Any]:
        """Builds full structured document context from transcribed lines."""
        non_empty_lines = [l for l in lines if l.get("text", "").strip()]
        full_text = " ".join(l["text"].strip() for l in non_empty_lines)

        entities = extract_structured_entities(non_empty_lines)

        if not full_text:
            return {
                "document_classification": {"primary_type": "Unknown / Empty", "scores": {}},
                "structural_summary": {"line_count": 0, "word_count": 0, "estimated_clauses": 0},
                "entities_and_lexicon": entities,
                "key_operative_lines": [],
                "vector_index_payload": {"model": "nlpaueb/legal-bert-base-uncased", "dimensions": 768},
            }

        # Encode full document and individual lines
        doc_vec = self.encode([full_text])[0]
        line_texts = [l["text"].strip() for l in non_empty_lines]
        line_vecs = self.encode(line_texts)

        # Classify document archetype via cosine similarity
        raw_sims = np.dot(self.archetype_embeddings, doc_vec)
        # Temperature-scaled softmax for interpretable confidence scores
        exp_scores = np.exp((raw_sims - np.max(raw_sims)) * 12.0)
        probs = exp_scores / np.sum(exp_scores)

        ranked_types = sorted(
            zip(self.archetype_labels, probs.tolist(), raw_sims.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # Score each line's semantic salience relative to the full document
        line_salience = np.dot(line_vecs, doc_vec).tolist()
        scored_lines = []
        for item, score in zip(non_empty_lines, line_salience):
            scored_lines.append({
                "line_id": item["line_id"],
                "salience_score": round(float(score), 4),
                "text": item["text"].strip(),
            })

        top_operative_lines = sorted(scored_lines, key=lambda x: x["salience_score"], reverse=True)[:5]

        # Estimate clause/sentence boundaries
        clause_count = max(1, len(re.split(r"[.;:]\s+", full_text)))
        word_count = len(full_text.split())

        vector_payload: Dict[str, Any] = {
            "model": "nlpaueb/legal-bert-base-uncased",
            "dimensions": int(doc_vec.shape[0]),
            "l2_norm": round(float(np.linalg.norm(doc_vec)), 4),
            "embedding_preview": [round(float(v), 5) for v in doc_vec[:8]],
        }
        if include_full_embedding:
            vector_payload["embedding_vector"] = [round(float(v), 6) for v in doc_vec.tolist()]

        return {
            "document_classification": {
                "primary_type": ranked_types[0][0],
                "confidence": round(ranked_types[0][1], 4),
                "all_scores": {
                    label: {"confidence": round(conf, 4), "cosine_sim": round(sim, 4)}
                    for label, conf, sim in ranked_types
                },
            },
            "structural_summary": {
                "non_empty_lines": len(non_empty_lines),
                "total_words": word_count,
                "estimated_clauses": clause_count,
            },
            "entities_and_lexicon": entities,
            "key_operative_lines": top_operative_lines,
            "vector_index_payload": vector_payload,
        }
