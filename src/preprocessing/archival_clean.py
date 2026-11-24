from typing import List, Tuple
import cv2
import numpy as np


def suppress_watermark_and_enhance(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separates thin cursive ink strokes from thick diagonal watermarks, aged paper
    gradients, and dark scan margins using morphological closing / black-hat filtering.

    Returns:
        enhanced_gray: Clean light-background grayscale image suitable for TrOCR crops.
        ink_mask: Clean binary mask (255 = ink, 0 = background) for line segmentation.
    """
    h, w = gray_img.shape[:2]

    # 1. Normalize uneven yellowed parchment illumination via large background division
    bg_k = max(31, (min(h, w) // 25) | 1)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_k, bg_k))
    background = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, bg_kernel)
    norm = cv2.divide(gray_img, np.clip(background, 1, 255), scale=255)

    # 2. Identify thick artifacts (watermark letters, dark borders, punch holes)
    # A small closing erases thin cursive strokes (2-5 px) but leaves thick watermarks dark
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_small = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, thick_kernel)

    # Black-hat extracts ONLY thin dark strokes (handwriting)
    blackhat = cv2.subtract(closed_small, norm)

    # Suppress watermark interior + dark scanner edges
    # Thick artifacts appear dark in closed_small; dampen halo pixels around them
    thick_artifacts = (closed_small < 215).astype(np.uint8) * 255
    thick_artifacts = cv2.dilate(
        thick_artifacts,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    # Clean grayscale image for TrOCR: white background with enhanced thin dark ink
    # Keep handwriting even if overlapping watermark by preserving strong blackhat signal
    cleaned_signal = blackhat.copy()
    weak_watermark_halo = (thick_artifacts > 0) & (blackhat < 28)
    cleaned_signal[weak_watermark_halo] = 0

    # Stretch contrast for TrOCR readability (white paper = 255, ink = dark)
    p99 = max(float(np.percentile(cleaned_signal, 99.5)), 25.0)
    scaled = np.clip((cleaned_signal.astype(np.float32) / p99) * 255.0, 0, 255).astype(np.uint8)
    enhanced_gray = 255 - scaled

    # 3. Build binary ink mask for line segmentation
    thresh_val = max(18, int(np.percentile(cleaned_signal[cleaned_signal > 0], 68))) if np.any(cleaned_signal > 0) else 25
    ink_mask = (cleaned_signal >= thresh_val).astype(np.uint8) * 255

    # Mask out outer 2% scanner edge border noise
    mx, my = max(8, int(w * 0.02)), max(8, int(h * 0.02))
    ink_mask[:my, :] = 0
    ink_mask[h - my :, :] = 0
    ink_mask[:, :mx] = 0
    ink_mask[:, w - mx :] = 0

    # Remove tiny dust specks and huge non-text blobs (circular stamps / borders)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    filtered_mask = np.zeros_like(ink_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 12:
            continue
        if bh > h * 0.14 or bw > w * 0.45:
            continue
        filtered_mask[labels == i] = 255

    return enhanced_gray, filtered_mask


def deskew_clamped(gray_img: np.ndarray, ink_mask: np.ndarray, max_angle: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Deskews only within +/- max_angle degrees using cleaned ink so diagonal watermarks cannot tilt the page."""
    coords = np.column_stack(np.where(ink_mask > 0))
    if len(coords) < 100:
        return gray_img, ink_mask

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    if abs(angle) < 0.4 or abs(angle) > max_angle:
        return gray_img, ink_mask

    h, w = gray_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot_gray = cv2.warpAffine(gray_img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    rot_mask = cv2.warpAffine(ink_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot_gray, rot_mask


def segment_lines_archival(
    enhanced_gray: np.ndarray,
    ink_mask: np.ndarray,
    min_line_height: int = 16,
    min_line_width: int = 55,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Segments text lines via horizontal morphological band merging rather than global
    full-width projection, making it resilient to stamps, margin notes, and watermarks.

    Returns:
        List of ((x, y, w, h), grayscale_line_crop) sorted in reading order.
    """
    h, w = enhanced_gray.shape[:2]

    # Merge words horizontally into line bands without merging lines vertically
    horiz_k = max(25, min(70, int(w * 0.048)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_k, 3))
    dilated = cv2.dilate(ink_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < min_line_height or bw < min_line_width:
            continue
        if bh > h * 0.22:
            # Split overly tall merged regions into horizontal strips
            n_splits = max(2, int(round(bh / max(28, h * 0.045))))
            strip_h = bh // n_splits
            for s in range(n_splits):
                sy = y + s * strip_h
                sh = strip_h if s < n_splits - 1 else (y + bh - sy)
                if sh >= min_line_height:
                    boxes.append((x, sy, bw, sh))
        else:
            boxes.append((x, y, bw, bh))

    # Merge fragments sitting on the same horizontal row
    boxes.sort(key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    for x, y, bw, bh in boxes:
        cy = y + bh / 2.0
        placed = False
        for m in merged:
            mcy = m[1] + m[3] / 2.0
            if abs(cy - mcy) < max(bh, m[3]) * 0.55:
                nx = min(m[0], x)
                ny = min(m[1], y)
                nx2 = max(m[0] + m[2], x + bw)
                ny2 = max(m[1] + m[3], y + bh)
                m[0], m[1], m[2], m[3] = nx, ny, nx2 - nx, ny2 - ny
                placed = True
                break
        if not placed:
            merged.append([x, y, bw, bh])

    merged.sort(key=lambda b: b[1])

    line_results = []
    for x, y, bw, bh in merged:
        pad_y = max(6, int(bh * 0.25))
        pad_x = max(10, int(bw * 0.02))
        y1 = max(0, y - pad_y)
        y2 = min(h, y + bh + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(w, x + bw + pad_x)
        crop = enhanced_gray[y1:y2, x1:x2]
        line_results.append(((x1, y1, x2 - x1, y2 - y1), crop))

    return line_results
