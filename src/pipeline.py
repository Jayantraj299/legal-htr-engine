import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.line_segment import extract_line_boundaries_with_seams
from src.preprocessing.archival_clean import (
    suppress_watermark_and_enhance,
    deskew_clamped,
    segment_lines_archival,
    split_wide_line,
)
from src.preprocessing.stamp_deed_clean import clean_stamp_paper_deed, segment_typed_lines, split_typed_line_on_spaces
from src.engine.model import TrOCREngine
from src.analysis.context_extractor import LegalContextExtractor
from src.analysis.formatter import write_readable_markdown, write_readable_html


def sauvola_binarize(
    gray_img: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    padded = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded.astype(np.float64) ** 2)
    h, w = gray_img.shape
    area = window_size * window_size
    sum_val = (
        integral[window_size : h + window_size, window_size : w + window_size]
        - integral[0 : h, window_size : w + window_size]
        - integral[window_size : h + window_size, 0 : w]
        + integral[0 : h, 0 : w]
    )
    sum_sq = (
        integral_sq[window_size : h + window_size, window_size : w + window_size]
        - integral_sq[0 : h, window_size : w + window_size]
        - integral_sq[window_size : h + window_size, 0 : w]
        + integral_sq[0 : h, 0 : w]
    )
    mean = sum_val / area
    variance = np.maximum((sum_sq / area) - (mean ** 2), 0.0)
    threshold = mean * (1.0 + k * (np.sqrt(variance) / r - 1.0))
    return np.where(gray_img > threshold, 255, 0).astype(np.uint8)


class LegalHTRPipeline:
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = None,
        fp16: bool = True,
        extract_context: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.fp16 = fp16
        self.extract_context = extract_context
        self.engine = TrOCREngine(model_name=model_name, device=device, fp16=fp16)

    def _release_htr_vram(self):
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine.model
            del self.engine
            self.engine = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_image(
        self,
        image_path: Path,
        work_dir: Path,
        deskew: bool = True,
        classic_sauvola: bool = False,
        stamp_deed: bool = False,
        window_size: int = 25,
        k: float = 0.2,
        batch_size: int = 4,
        include_full_embedding: bool = False,
    ) -> Dict[str, Any]:
        start_total = time.perf_counter()
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

        work_dir.mkdir(parents=True, exist_ok=True)
        lines_dir = work_dir / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)
        for old in lines_dir.glob("line_*.png"):
            old.unlink()

        t0 = time.perf_counter()
        if stamp_deed:
            enhanced_gray, char_mask = clean_stamp_paper_deed(raw_bgr)
            cv2.imwrite(str(work_dir / "enhanced_clean.png"), enhanced_gray)
            cv2.imwrite(str(work_dir / "ink_mask.png"), char_mask)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            segmented = segment_typed_lines(enhanced_gray, char_mask)
            segment_ms = (time.perf_counter() - t1) * 1000.0
            max_aspect = 6.5
        elif classic_sauvola:
            binary_img = sauvola_binarize(raw_gray, window_size=window_size, k=k)
            cv2.imwrite(str(work_dir / "binarized.png"), binary_img)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            raw_crops = extract_line_boundaries_with_seams(binary_img)
            segmented = [(None, crop) for crop in raw_crops]
            segment_ms = (time.perf_counter() - t1) * 1000.0
            max_aspect = 7.5
        else:
            enhanced_gray, ink_mask = suppress_watermark_and_enhance(raw_gray)
            if deskew:
                enhanced_gray, ink_mask = deskew_clamped(enhanced_gray, ink_mask)
            cv2.imwrite(str(work_dir / "enhanced_clean.png"), enhanced_gray)
            cv2.imwrite(str(work_dir / "ink_mask.png"), ink_mask)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            segmented = segment_lines_archival(enhanced_gray, ink_mask)
            segment_ms = (time.perf_counter() - t1) * 1000.0
            max_aspect = 7.5

        pil_chunks = []
        chunk_line_map = []
        crop_records = []
        for idx, (bbox, crop) in enumerate(segmented):
            crop_name = f"line_{idx:03d}.png"
            crop_path = lines_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            rec: Dict[str, Any] = {"line_id": idx, "filename": crop_name, "path": str(crop_path)}
            if bbox is not None:
                rec["bbox"] = list(bbox)
            crop_records.append(rec)

            splitter = split_typed_line_on_spaces if stamp_deed else split_wide_line
            for sub in splitter(crop, max_aspect=9.0 if stamp_deed else max_aspect):
                pil_chunks.append(Image.fromarray(sub).convert("RGB"))
                chunk_line_map.append(idx)

        t2 = time.perf_counter()
        chunk_texts = []
        for i in range(0, len(pil_chunks), batch_size):
            batch = pil_chunks[i : i + batch_size]
            chunk_texts.extend(self.engine.predict_batch(batch))
        ocr_ms = (time.perf_counter() - t2) * 1000.0

        grouped = {i: [] for i in range(len(crop_records))}
        for line_idx, txt in zip(chunk_line_map, chunk_texts):
            if txt and txt.strip():
                grouped[line_idx].append(txt.strip())

        transcriptions = []
        for r in crop_records:
            joined = " ".join(grouped[r["line_id"]])
            r["text"] = joined
            transcriptions.append(joined)

        context_ms = 0.0
        document_context = None
        if self.extract_context:
            self._release_htr_vram()
            t3 = time.perf_counter()
            extractor = LegalContextExtractor(device=self.device, fp16=self.fp16)
            document_context = extractor.analyze_document(
                crop_records,
                include_full_embedding=include_full_embedding,
            )
            context_ms = (time.perf_counter() - t3) * 1000.0

        result: Dict[str, Any] = {
            "document": str(image_path),
            "num_lines": len(crop_records),
            "text": "\n".join(transcriptions),
            "lines": crop_records,
            "timings_ms": {
                "preprocess": round(preprocess_ms, 2),
                "segment": round(segment_ms, 2),
                "ocr": round(ocr_ms, 2),
                "context_analysis": round(context_ms, 2),
                "total": round((time.perf_counter() - start_total) * 1000.0, 2),
            },
        }
        if document_context is not None:
            result["document_context"] = document_context

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Legal Document HTR & Readable Export Pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to input scanned document image")
    parser.add_argument("-o", "--output", default=None, help="Base output path (writes .md, .html, and .json)")
    parser.add_argument("--work-dir", default=None, help="Directory to save intermediate cleaned images and line crops")
    parser.add_argument("--model", default="microsoft/trocr-base-handwritten", help="Hugging Face TrOCR model ID")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--stamp-deed", action="store_true", help="Color-aware red-ink removal & stamp-banner stripping for typed Indian deeds")
    parser.add_argument("--classic-sauvola", action="store_true", help="Use classic Sauvola + HPP instead of archival watermark cleaner")
    parser.add_argument("--window-size", type=int, default=25, help="Sauvola window size (classic mode)")
    parser.add_argument("-k", type=float, default=0.2, help="Sauvola threshold factor (classic mode)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskewing")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 half precision")
    parser.add_argument("--no-context", action="store_true", help="Skip Legal-BERT context extraction")
    parser.add_argument("--full-embedding", action="store_true", help="Include full 768-d vector in JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist.")
        sys.exit(1)

    work_dir = Path(args.work_dir) if args.work_dir else Path("data/processed") / f"run_{input_path.stem}"

    pipeline = LegalHTRPipeline(
        model_name=args.model,
        fp16=not args.no_fp16,
        extract_context=not args.no_context,
    )
    result = pipeline.process_image(
        image_path=input_path,
        work_dir=work_dir,
        deskew=not args.no_deskew,
        classic_sauvola=args.classic_sauvola,
        stamp_deed=args.stamp_deed,
        window_size=args.window_size,
        k=args.k,
        batch_size=args.batch_size,
        include_full_embedding=args.full_embedding,
    )

    if args.output:
        raw_out = Path(args.output)
        out_base = (work_dir / raw_out) if len(raw_out.parts) == 1 else raw_out
    else:
        out_base = work_dir / input_path.stem
    out_base = out_base.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    md_path = out_base.with_suffix(".md")
    html_path = out_base.with_suffix(".html")
    json_path = out_base.with_suffix(".json")

    write_readable_markdown(result, md_path)
    write_readable_html(result, html_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"DETECTED LINES: {result['num_lines']} | SOURCE: {input_path.name}")
    print("=" * 68)
    print(md_path.read_text(encoding="utf-8"))
    print("=" * 68)
    print(f"Readable Markdown : {md_path}")
    print(f"Readable HTML Doc : {html_path}")
    print(f"Raw JSON Metadata : {json_path}")


if __name__ == "__main__":
    main()
