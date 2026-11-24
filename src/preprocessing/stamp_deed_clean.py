from typing import List, Tuple
import cv2
import numpy as np


def clean_stamp_paper_deed(bgr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    White-balances yellowed stamp paper per channel before suppressing red/magenta
    ink via (r_norm - g_norm), preserving faded black typewriter strokes intact.
    """
    h, w = bgr_img.shape[:2]
    _, g, r = cv2.split(bgr_img)

    # 1. Estimate smooth local paper background per channel
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    blur_k = max(31, (min(h, w) // 16) | 1)
    r_bg = cv2.GaussianBlur(cv2.morphologyEx(r, cv2.MORPH_CLOSE, close_k), (blur_k, blur_k), 0)
    g_bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, close_k), (blur_k, blur_k), 0)

    # 2. White-balance Red and Green channels against local paper illumination
    r_norm = cv2.divide(r, np.clip(r_bg, 1, 255), scale=255)
    g_norm = cv2.divide(g, np.clip(g_bg, 1, 255), scale=255)

    # 3. Softly lift ONLY chromatic red/magenta ink (r_norm >> g_norm)
    rg_diff = r_norm.astype(np.float32) - g_norm.astype(np.float32)
    red_only = (rg_diff > 18.0) & (r_norm > 125)
    r_clean = r_norm.astype(np.float32)
    r_clean[red_only] = np.clip(r_clean[red_only] + 1.8 * (rg_diff[red_only] - 18.0), 0.0, 255.0)

    # 4. Crisp contrast stretch + gentle gamma darken for sharp typewriter serifs
    stretched = np.clip((r_clean - 50.0) / 182.0, 0.0, 1.0)
    enhanced_gray = np.clip((stretched ** 1.35) * 255.0, 0.0, 255.0).astype(np.uint8)

    # 5. Build clean character mask and strip scanner borders
    ink_mask = (enhanced_gray < 205).astype(np.uint8) * 255
    mx = max(8, int(w * 0.02))
    my_top = max(8, int(h * 0.02))
    ink_mask[:my_top, :] = 0
    ink_mask[h - 4 :, :] = 0
    ink_mask[:, :mx] = 0
    ink_mask[:, w - mx :] = 0

    # 6. Strip ornate top stamp-paper banner (contiguous dense rows > 7% page height)
    row_density = np.mean(ink_mask > 0, axis=1)
    dense_rows = row_density > 0.14
    run_start = None
    min_banner_h = max(35, int(h * 0.07))
    for y in range(h):
        if dense_rows[y] and run_start is None:
            run_start = y
        elif not dense_rows[y] and run_start is not None:
            if (y - run_start) >= min_banner_h:
                pad = max(6, int(h * 0.01))
                ink_mask[max(0, run_start - pad) : min(h, y + pad), :] = 0
            run_start = None
    if run_start is not None and (h - run_start) >= min_banner_h:
        ink_mask[max(0, run_start - 6) : h, :] = 0

    # 7. Keep character-sized & underlined-heading components (filter tall seals & borders)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    char_mask = np.zeros_like(ink_mask)
    max_char_h = max(26, int(h * 0.038))
    max_line_w = int(w * 0.55)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 4 or bh > max_char_h or bw > max_line_w:
            continue
        char_mask[labels == i] = 255

    return enhanced_gray, char_mask


def _split_tall_box_by_valleys(
    mask: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
    est_line_h: float,
    min_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Recursively splits vertically merged text lines at the deepest horizontal ink valley."""
    if bh <= est_line_h * 1.65:
        return [(x, y, bw, bh)]

    sub = mask[y : y + bh, x : x + bw]
    proj = np.sum(sub > 0, axis=1).astype(np.float32)
    smooth = np.convolve(proj, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")

    # Search for the deepest whitespace valley away from the top/bottom edges
    margin = max(min_h, int(est_line_h * 0.65))
    if margin * 2 >= bh:
        return [(x, y, bw, bh)]

    split_row = margin + int(np.argmin(smooth[margin : bh - margin]))

    top_boxes = _split_tall_box_by_valleys(mask, x, y, bw, split_row, est_line_h, min_h)
    bot_boxes = _split_tall_box_by_valleys(mask, x, y + split_row, bw, bh - split_row, est_line_h, min_h)
    return top_boxes + bot_boxes


def segment_typed_lines(
    enhanced_gray: np.ndarray,
    char_mask: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Slices tightly spaced typewritten lines without merging stacked lines."""
    h, w = enhanced_gray.shape[:2]

    # Estimate typical single-line height from character component heights
    num_l, _, stats, _ = cv2.connectedComponentsWithStats(char_mask, connectivity=8)
    char_heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, num_l)
        if 5 <= stats[i, cv2.CC_STAT_HEIGHT] <= int(h * 0.03) and stats[i, cv2.CC_STAT_AREA] >= 8
    ]
    med_char_h = float(np.median(char_heights)) if char_heights else max(10.0, h * 0.014)
    est_line_h = max(12.0, med_char_h * 1.45)

    horiz_k = max(14, int(w * 0.028))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_k, 1))
    dilated = cv2.dilate(char_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_h = max(7, int(med_char_h * 0.70))
    frag_min_w = max(18, int(w * 0.03))
    final_min_w = max(45, int(w * 0.10))

    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < min_h or bw < frag_min_w:
            continue
        if (bw / float(max(bh, 1))) < 1.1:
            continue
        split_boxes = _split_tall_box_by_valleys(char_mask, x, y, bw, bh, est_line_h, min_h)
        for sx, sy, sbw, sbh in split_boxes:
            strip = char_mask[sy : sy + sbh, sx : sx + sbw]
            ys, xs = np.where(strip > 0)
            if len(ys) < 10:
                continue
            ty1, ty2 = int(ys.min()), int(ys.max()) + 1
            tx1, tx2 = int(xs.min()), int(xs.max()) + 1
            if (ty2 - ty1) >= min_h and (tx2 - tx1) >= frag_min_w:
                raw_boxes.append((sx + tx1, sy + ty1, tx2 - tx1, ty2 - ty1))

    # Identify main body left column from wide lines so left-margin seals do not merge in
    wide_lefts = [b[0] for b in raw_boxes if b[2] >= int(w * 0.30)]
    body_left = int(np.median(wide_lefts)) if len(wide_lefts) >= 4 else 0

    # Drop isolated narrow fragments sitting entirely inside the left seal/margin gutter
    if body_left > int(w * 0.12):
        raw_boxes = [b for b in raw_boxes if (b[0] + b[2]) >= (body_left - 8)]

    # Merge left/right halves sitting on the same baseline (handles red-ink mid-line gaps)
    raw_boxes.sort(key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    max_horiz_gap = int(w * 0.35)
    for x, y, bw, bh in raw_boxes:
        cy = y + bh / 2.0
        placed = False
        for m in merged:
            mcy = m[1] + m[3] / 2.0
            horiz_gap = max(0, max(x - (m[0] + m[2]), m[0] - (x + bw)))
            new_h = max(m[1] + m[3], y + bh) - min(m[1], y)
            if abs(cy - mcy) < est_line_h * 0.60 and horiz_gap <= max_horiz_gap and new_h <= est_line_h * 1.75:
                nx = min(m[0], x)
                ny = min(m[1], y)
                nx2 = max(m[0] + m[2], x + bw)
                ny2 = max(m[1] + m[3], y + bh)
                m[0], m[1], m[2], m[3] = nx, ny, nx2 - nx, ny2 - ny
                placed = True
                break
        if not placed:
            merged.append([x, y, bw, bh])

    if len(merged) >= 5:
        widths = sorted(m[2] for m in merged)
        ref_w = widths[int(len(widths) * 0.7)]
        merged = [m for m in merged if m[2] >= max(final_min_w, int(ref_w * 0.24))]

    merged.sort(key=lambda b: b[1])

    results = []
    for x, y, bw, bh in merged:
        pad_y = max(3, int(bh * 0.22))
        pad_x = max(8, int(bw * 0.02))
        y1 = max(0, y - pad_y)
        y2 = min(h, y + bh + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(w, x + bw + pad_x)
        crop = enhanced_gray[y1:y2, x1:x2]

        if crop.shape[0] < 56:
            scale = min(3.0, 56.0 / float(max(crop.shape[0], 1)))
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            # Mild unsharp mask to crisp up upscaled low-res typewriter strokes
            blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
            crop = cv2.addWeighted(crop, 1.30, blur, -0.30, 0)

        crop = cv2.copyMakeBorder(crop, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=255)
        results.append(((x1, y1, x2 - x1, y2 - y1), crop))

    return results


def split_typed_line_on_spaces(crop: np.ndarray, max_aspect: float = 9.0) -> List[np.ndarray]:
    """Splits wide typed lines into 2-3 clean chunks strictly at inter-word spaces."""
    ch, cw = crop.shape[:2]
    if ch == 0 or (cw / float(ch)) <= max_aspect:
        return [crop]

    n_chunks = max(2, int(np.ceil((cw / float(ch)) / max_aspect)))
    target_w = cw // n_chunks

    band = crop[int(ch * 0.15) : int(ch * 0.85), :]
    col_ink = np.sum(band < 190, axis=0)
    is_space = col_ink <= 1

    gaps = []
    in_gap, start = False, 0
    for x, val in enumerate(is_space):
        if val and not in_gap:
            start, in_gap = x, True
        elif not val and in_gap:
            gaps.append((start, x, x - start))
            in_gap = False
    if in_gap:
        gaps.append((start, cw, cw - start))

    cuts = [0]
    for k in range(1, n_chunks):
        ideal = k * target_w
        lo = max(cuts[-1] + int(ch * 2.5), ideal - int(target_w * 0.30))
        hi = min(cw - int(ch * 2.5), ideal + int(target_w * 0.30))
        if lo >= hi:
            continue

        cand_gaps = [g for g in gaps if lo <= (g[0] + g[1]) // 2 <= hi and g[2] >= 5]
        if cand_gaps:
            best = max(cand_gaps, key=lambda g: (g[2], -abs(((g[0] + g[1]) // 2) - ideal)))
            cuts.append((best[0] + best[1]) // 2)
        else:
            cuts.append(lo + int(np.argmin(col_ink[lo:hi])))
    cuts.append(cw)

    chunks = []
    for i in range(len(cuts) - 1):
        sub = crop[:, cuts[i] : cuts[i + 1]]
        if sub.shape[1] >= max(24, ch) and np.any(sub < 185):
            padded = cv2.copyMakeBorder(sub, 6, 6, 14, 14, cv2.BORDER_CONSTANT, value=255)
            chunks.append(padded)

    return chunks if chunks else [crop]
