import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def reconstruct_paragraphs(lines: List[Dict[str, Any]]) -> List[str]:
    """Joins OCR line strings into flowing paragraphs using vertical spacing when available."""
    non_empty = [l for l in lines if sum(c.isalnum() for c in l.get("text", "")) >= 3]
    if not non_empty:
        return []

    paragraphs: List[str] = []
    current_para: List[str] = []
    prev_bottom = None
    prev_height = None

    for item in non_empty:
        text = item["text"].strip()
        bbox = item.get("bbox")  # [x, y, w, h] if available
        start_new = False

        if bbox and prev_bottom is not None and prev_height is not None:
            gap = bbox[1] - prev_bottom
            if gap > max(prev_height * 1.15, 28):
                start_new = True

        if start_new and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = [text]
        else:
            current_para.append(text)

        if bbox:
            prev_bottom = bbox[1] + bbox[3]
            prev_height = bbox[3]

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs


def write_readable_markdown(result: Dict[str, Any], out_path: Path) -> None:
    """Exports a clean, human-readable Markdown document report."""
    doc_name = Path(result.get("document", "document")).name
    ctx = result.get("document_context", {})
    cls = ctx.get("document_classification", {})
    ent = ctx.get("entities_and_lexicon", {})
    summary = ctx.get("structural_summary", {})
    paragraphs = reconstruct_paragraphs(result.get("lines", []))

    primary_type = cls.get("primary_type", "Unclassified")
    confidence = cls.get("confidence", 0.0) * 100.0

    md_lines = [
        f"# Digitized Legal Document: {doc_name}",
        "",
        f"**Processed On:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Predicted Document Type:** {primary_type} ({confidence:.1f}% confidence)  ",
        f"**Detected Lines:** {result.get('num_lines', 0)} | **Total Words:** {summary.get('total_words', 0)}",
        "",
        "---",
        "",
        "## Reconstructed Document Text",
        "",
    ]

    if paragraphs:
        for para in paragraphs:
            md_lines.append(para)
            md_lines.append("")
    else:
        md_lines.append("*No readable text lines were detected.*")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Extracted Legal Context & Metadata",
        "",
    ])

    dates = [d["value"] for d in ent.get("dates", [])]
    citations = [c["reference"] for c in ent.get("statutory_and_clause_citations", [])]
    parties = [p["entity"] for p in ent.get("extracted_parties_and_actors", [])]

    md_lines.append(f"- **Dates Found:** {', '.join(dates) if dates else 'None detected'}")
    md_lines.append(f"- **Statutory / Clause Citations:** {', '.join(citations) if citations else 'None detected'}")
    md_lines.append(f"- **Parties / Actors:** {', '.join(parties) if parties else 'None detected'}")

    lex_hits = []
    for cat, terms in ent.get("legal_lexicon_matches", {}).items():
        if terms:
            lex_hits.append(f"{cat.replace('_', ' ')}: {', '.join(terms.keys())}")
    md_lines.append(f"- **Legal Lexicon Matches:** {' | '.join(lex_hits) if lex_hits else 'None detected'}")
    md_lines.append("")

    top_lines = ctx.get("key_operative_lines", [])
    if top_lines:
        md_lines.extend([
            "## Key Operative Lines (Ranked by Semantic Salience)",
            "",
        ])
        for op in top_lines:
            md_lines.append(f"- **Line {op['line_id']:02d}** (score `{op['salience_score']:.3f}`): {op['text']}")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Line-by-Line Transcription Audit",
        "",
        "| Line | Transcribed Text |",
        "| :--- | :--- |",
    ])
    for item in result.get("lines", []):
        safe_text = item.get("text", "").replace("|", "\\|")
        md_lines.append(f"| {item['line_id']:02d} | {safe_text} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_readable_html(result: Dict[str, Any], out_path: Path) -> None:
    """Exports a clean, styled HTML document viewable in any browser or printable to PDF."""
    doc_name = html.escape(Path(result.get("document", "document")).name)
    ctx = result.get("document_context", {})
    cls = ctx.get("document_classification", {})
    ent = ctx.get("entities_and_lexicon", {})
    summary = ctx.get("structural_summary", {})
    paragraphs = reconstruct_paragraphs(result.get("lines", []))

    primary_type = html.escape(cls.get("primary_type", "Unclassified"))
    confidence = cls.get("confidence", 0.0) * 100.0

    para_html = "".join(f"<p>{html.escape(p)}</p>\n" for p in paragraphs) or "<p><em>No readable text detected.</em></p>"

    dates = ", ".join(html.escape(d["value"]) for d in ent.get("dates", [])) or "None detected"
    citations = ", ".join(html.escape(c["reference"]) for c in ent.get("statutory_and_clause_citations", [])) or "None detected"
    parties = ", ".join(html.escape(p["entity"]) for p in ent.get("extracted_parties_and_actors", [])) or "None detected"

    rows_html = "".join(
        f"<tr><td class='ln'>{item['line_id']:02d}</td><td>{html.escape(item.get('text', ''))}</td></tr>\n"
        for item in result.get("lines", [])
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Digitized Document - {doc_name}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 860px; margin: 40px auto; padding: 0 24px; color: #1e293b; line-height: 1.7; background: #f8fafc; }}
    .paper {{ background: #ffffff; padding: 40px 48px; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    h1 {{ font-family: system-ui, sans-serif; font-size: 1.5rem; margin-bottom: 6px; color: #0f172a; }}
    h2 {{ font-family: system-ui, sans-serif; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.05em; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; }}
    .meta {{ font-family: system-ui, sans-serif; font-size: 0.9rem; color: #475569; margin-bottom: 24px; }}
    .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }}
    .doc-body {{ font-size: 1.08rem; background: #fcfcfd; border-left: 3px solid #0284c7; padding: 16px 22px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 180px 1fr; gap: 8px 16px; font-family: system-ui, sans-serif; font-size: 0.92rem; }}
    .label {{ font-weight: 600; color: #334155; }}
    table {{ width: 100%; border-collapse: collapse; font-family: system-ui, sans-serif; font-size: 0.9rem; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
    td.ln {{ width: 52px; color: #64748b; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="paper">
    <h1>Digitized Document: {doc_name}</h1>
    <div class="meta">
      <span class="badge">{primary_type} ({confidence:.1f}%)</span>
      &nbsp;&bull;&nbsp; {result.get('num_lines', 0)} lines
      &nbsp;&bull;&nbsp; {summary.get('total_words', 0)} words
      &nbsp;&bull;&nbsp; Processed {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>

    <h2>Reconstructed Document</h2>
    <div class="doc-body">
      {para_html}
    </div>

    <h2>Extracted Legal Context</h2>
    <div class="grid">
      <div class="label">Dates Found</div><div>{dates}</div>
      <div class="label">Statutory Citations</div><div>{citations}</div>
      <div class="label">Parties / Actors</div><div>{parties}</div>
    </div>

    <h2>Line-by-Line Audit</h2>
    <table>
      <thead><tr><th>Line</th><th>Transcription</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
