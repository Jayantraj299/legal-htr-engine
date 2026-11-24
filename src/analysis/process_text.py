import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.context_extractor import LegalContextExtractor


def load_document_lines(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        raw_text = "\n".join((p.extract_text() or "") for p in reader.pages)
    else:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    lines = []
    for idx, line in enumerate(raw_text.splitlines()):
        cleaned = line.strip()
        if cleaned:
            lines.append({"line_id": idx, "text": cleaned})
    return raw_text, lines


def main():
    parser = argparse.ArgumentParser(description="Run Legal-BERT context analysis on .txt or .pdf documents.")
    parser.add_argument("-i", "--input", required=True, help="Path to a .txt/.pdf file or a directory of documents")
    parser.add_argument("-o", "--output-dir", default="data/processed/text_contexts", help="Output directory for JSON files")
    parser.add_argument("--full-embedding", action="store_true", help="Store full 768-d embedding vector in JSON")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [in_path] if in_path.is_file() else sorted(
        p for p in in_path.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    if not targets:
        print(f"No .txt, .md, or .pdf files found at {in_path}")
        sys.exit(1)

    extractor = LegalContextExtractor(fp16=True)

    for doc_path in targets:
        raw_text, lines = load_document_lines(doc_path)
        context = extractor.analyze_document(lines, include_full_embedding=args.full_embedding)
        payload = {
            "document": str(doc_path),
            "num_lines": len(lines),
            "text": raw_text,
            "lines": lines,
            "document_context": context,
        }
        out_file = out_dir / f"{doc_path.stem}_context.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        cls = context["document_classification"]
        print(f"[DONE] {doc_path.name} -> {cls['primary_type']} ({cls['confidence']*100:.1f}%) | Saved: {out_file}")


if __name__ == "__main__":
    main()
