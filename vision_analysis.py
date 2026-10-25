"""
PlantVerse — Classical Computer Vision Foliar & Lesion Analysis (`vision_analysis.py`)
======================================================================================
Implements:
1. HSV & Excess-Green (2G - R - B > 25) + border LAB background-aware leaf segmentation.
2. Intra-lamina lesion mask extraction (chlorotic, necrotic, rust, and powdery patches)
   and affected leaf area percentage calculation.
3. Multi-leaf and cluttered-background quality warning detection.
4. Visual overlay rendering (leaf contour + lesion contour highlights).
5. Project-defined severity category mapping (Healthy / Early / Moderate / Severe)
   from centralized thresholds in `ml_config.py`.
6. Derived foliar health score (0-100), explicitly labeled as a heuristic derived metric.

Honest Labeling Note:
This module uses color-threshold segmentation (classical CV, not a trained deep
segmentation model). Known limitations include strong cast shadows, soil/bark matching
lesion hues, diffuse viral mottling, and naturally variegated or non-green leaves.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

for _venv_sp in sorted((Path(__file__).resolve().parent / "venv" / "lib").glob("python*/site-packages")):
    if str(_venv_sp) not in sys.path:
        sys.path.append(str(_venv_sp))

import cv2
import numpy as np
from PIL import Image, ImageOps

from ml_config import (
    SEVERITY_THRESHOLDS_PCT,
    VARIEGATED_OR_COLORED_FOLIAGE_PLANTS,
)

SEGMENTATION_METHOD_LABEL = (
    "Color-threshold segmentation (classical CV: HSV + Excess-Green index + border LAB background filter — not a trained segmentation model)"
)

SEGMENTATION_LIMITATIONS: List[str] = [
    "Cast shadows or dark specular glare on leaves can be misclassified as necrotic tissue.",
    "Soil, brown bark, or dead background foliage touching the leaf margin can inflate lesion area if chroma matches.",
    "Non-green, purple, or variegated leaves (e.g., Croton, Coleus, Money Plant, Snake Plant) trigger false color masks and are excluded.",
    "Diffuse viral mottling or mild nutrient chlorosis without discrete spots may be underestimated.",
]

SEVERITY_SCHEME_LABEL = "Project-defined severity categories (Early < 5%, Moderate 5–20%, Severe >= 20% affected leaf area; not agricultural standards)"
HEALTH_SCORE_LABEL = "Derived metric (not scientifically validated)"

MAX_SEGMENTATION_SIDE = 512

_URGENCY_PENALTY = {
    "high": 25,
    "critical": 25,
    "moderate to high": 18,
    "medium": 12,
    "moderate": 12,
    "low": 5,
    "none": 0,
}


def _resize_for_cv(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = img_rgb.shape[:2]
    scale = MAX_SEGMENTATION_SIDE / float(max(h, w))
    if scale >= 1.0:
        return img_rgb, 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _largest_component_and_multi_check(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Return the significant foliar components mask (each >= 2.0% of image area and
    >= 30% of largest component) and the count of significant foliar components.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx_rel = int(np.argmax(areas))
    max_area = float(areas[max_idx_rel])
    min_sig_area = max(0.020 * mask.size, 0.30 * max_area)
    sig_ids = [1 + idx for idx, a in enumerate(areas) if float(a) >= min_sig_area]
    if not sig_ids:
        sig_ids = [1 + max_idx_rel]
    sig_mask = np.isin(labels, sig_ids).astype(np.uint8) * 255
    return sig_mask, len(sig_ids)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _normalized_polygon(contour: np.ndarray, w: int, h: int, eps: float) -> List[List[float]]:
    approx = cv2.approxPolyDP(contour, eps * cv2.arcLength(contour, True), True)
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in approx.reshape(-1, 2)]


def is_variegated_plant(plant_name_or_id: Optional[str]) -> bool:
    if not plant_name_or_id:
        return False
    clean = plant_name_or_id.strip().lower().replace("_", " ")
    if clean in VARIEGATED_OR_COLORED_FOLIAGE_PLANTS:
        return True
    for v in VARIEGATED_OR_COLORED_FOLIAGE_PLANTS:
        if v in clean:
            return True
    return False


def compute_severity_category(
    affected_pct: Optional[float],
    is_healthy: bool,
) -> str:
    """
    Map affected leaf area % to project-defined severity categories:
    Healthy / Early / Moderate / Severe (or 'Unknown' when segmentation fails).
    """
    if is_healthy:
        return "Healthy"
    if affected_pct is None:
        return "Unknown (Affected area could not be estimated)"
    if affected_pct <= 0.0:
        return "Early"
    if affected_pct < SEVERITY_THRESHOLDS_PCT["early_max"]:
        return "Early"
    if affected_pct < SEVERITY_THRESHOLDS_PCT["moderate_max"]:
        return "Moderate"
    return "Severe"


def compute_derived_health_score(
    affected_pct: Optional[float],
    is_healthy: bool,
    urgency_or_severity: str = "medium",
) -> Optional[int]:
    """
    Compute the heuristic 0–100 plant health score from affected leaf area %
    and condition urgency. Returns None if affected_pct could not be estimated
    on a diseased plant.
    """
    if is_healthy:
        if affected_pct is None or affected_pct < 10.0:
            return 100
        return 90
    if affected_pct is None:
        return None
    urg_key = (urgency_or_severity or "medium").strip().lower()
    penalty = _URGENCY_PENALTY.get(urg_key, 12)
    score = int(round(100.0 - penalty - 1.5 * float(affected_pct)))
    return max(5, min(95, score))


def segment_leaf(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Classical CV leaf + lesion segmentation compatible with FloraCare's vision.py,
    extended with multi-leaf and cluttered-background quality diagnostics and
    proximity-bounded chlorosis expansion.
    """
    small, _ = _resize_for_cv(img_rgb)
    h, w = small.shape[:2]
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, g, b = (blurred[..., i].astype(np.int16) for i in range(3))

    # Healthy foliage tissue: green hue (31..95) or Excess-Green index (2G - R - B > 22)
    exg = 2 * g - r - b
    green = (hue >= 31) & (hue <= 95) & (((sat >= 40) & (val >= 40)) | (exg > 22))

    # Symptom tones: yellowing, tan/brown necrosis, rust, purple-red spots
    tone = ((hue < 31) | (hue >= 165)) & (sat >= 35) & (val >= 50)
    tone_chlorosis = (hue >= 18) & (hue < 33) & (sat >= 90) & (val >= 90)

    # Border background chroma model in CIE LAB space
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.int16)
    border = np.concatenate(
        [
            lab[:6].reshape(-1, 3),
            lab[-6:].reshape(-1, 3),
            lab[:, :6].reshape(-1, 3),
            lab[:, -6:].reshape(-1, 3),
        ]
    )
    a_bg, b_bg = np.median(border[:, 1]), np.median(border[:, 2])
    border_chroma_std = float(np.std(border[:, 1:3]))
    bg_like = np.hypot(lab[..., 1] - a_bg, lab[..., 2] - b_bg) < 9

    dark = val < 60
    white = (sat < 40) & (val > 175)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_u8 = cv2.morphologyEx(green.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    core, sig_components = _largest_component_and_multi_check(green_u8)

    empty_res = {
        "found": False,
        "leaf_polygon": [],
        "leaf_polygons": [],
        "lesions": [],
        "bbox": None,
        "severity": 0.0,
        "leaf_area_ratio": 0.0,
        "multiple_leaves_detected": False,
        "cluttered_background_detected": border_chroma_std > 18.0,
        "quality_warnings": [
            "Affected area could not be estimated: no distinct green leaf lamina could be segmented from the background."
        ],
    }
    if core.sum() == 0 or (core > 0).mean() < 0.02:
        return empty_res

    plant = cv2.morphologyEx(((green | (tone & ~bg_like)) * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    _, labels = cv2.connectedComponents(plant)
    core_labels = np.unique(labels[core > 0])
    core_labels = core_labels[core_labels > 0]
    grown = np.isin(labels, core_labels).astype(np.uint8) * 255
    core_filled = _fill_holes(cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel))
    touches_border = grown[0].mean() + grown[-1].mean() + grown[:, 0].mean() + grown[:, -1].mean()
    if (grown > 0).sum() > 2.5 * (core_filled > 0).sum() or touches_border > 2 * 255:
        # Unbounded tone growth leaked into background; use proximity-bounded chlorosis expansion around core_filled
        prox_radius = max(15, int(min(h, w) * 0.12))
        prox_mask = cv2.dilate(core_filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prox_radius, prox_radius)))
        core_ys = np.where(core_filled > 0)[0]
        max_core_y = int(np.max(core_ys)) if len(core_ys) else h
        prox_mask[min(h, max_core_y + int(0.02 * h)) :, :] = 0
        bounded = cv2.morphologyEx(
            (((core_filled > 0) | (tone_chlorosis & (prox_mask > 0))) * 255).astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        )
        leaf = _fill_holes(bounded)
    else:
        leaf = _fill_holes(grown)

    leaf_area = int((leaf > 0).sum())
    leaf_area_ratio = float(leaf_area) / float(h * w)
    if leaf_area_ratio < 0.02:
        return empty_res

    inner = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    lesion = ((tone | dark | white) & ~green & (inner > 0)).astype(np.uint8) * 255
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    min_spot = max(6, int(leaf_area * 0.0005))
    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_spot]
    contours.sort(key=cv2.contourArea, reverse=True)
    lesion_area = sum(cv2.contourArea(c) for c in contours)

    leaf_contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_leaf_contours = [c for c in leaf_contours if cv2.contourArea(c) >= 0.01 * h * w]
    if not sig_leaf_contours:
        sig_leaf_contours = [max(leaf_contours, key=cv2.contourArea)]
    leaf_contour = max(sig_leaf_contours, key=cv2.contourArea)
    all_pts = np.vstack(sig_leaf_contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)

    multiple_leaves = sig_components >= 2
    cluttered_bg = (border_chroma_std > 16.0 and leaf_area_ratio < 0.35) or (touches_border > 2.2 * 255 and sig_components >= 2)
    quality_warnings: List[str] = []
    if multiple_leaves:
        quality_warnings.append(
            "Multiple leaves or foliage clusters detected in frame. For highest diagnostic accuracy, photograph a single isolated leaf."
        )
    if cluttered_bg:
        quality_warnings.append(
            "Cluttered or high-variance background detected around the specimen. Background colors may affect classical CV lesion boundary estimation."
        )

    affected_pct = round(min(100.0, 100.0 * float(lesion_area) / float(leaf_area)), 1)

    return {
        "found": True,
        "leaf_polygon": _normalized_polygon(leaf_contour, w, h, 0.004),
        "leaf_polygons": [_normalized_polygon(c, w, h, 0.004) for c in sig_leaf_contours],
        "lesions": [_normalized_polygon(c, w, h, 0.006) for c in contours[:40]],
        "bbox": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
        "severity": affected_pct,
        "leaf_area_ratio": round(leaf_area_ratio, 4),
        "lesion_count": len(contours),
        "multiple_leaves_detected": multiple_leaves,
        "cluttered_background_detected": cluttered_bg,
        "quality_warnings": quality_warnings,
    }


def crop_to_bbox(img_rgb: np.ndarray, bbox: Optional[List[float]], margin: float = 0.08) -> np.ndarray:
    """Crop the RGB image around the segmented leaf bounding box."""
    if not bbox:
        return img_rgb
    H, W = img_rgb.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, int((x - margin) * W))
    y0 = max(0, int((y - margin) * H))
    x1 = min(W, int((x + bw + margin) * W))
    y1 = min(H, int((y + bh + margin) * H))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return img_rgb
    return img_rgb[y0:y1, x0:x1]


def render_segmentation_overlay(
    image_rgb: np.ndarray,
    seg_result: Dict[str, Any],
) -> bytes:
    """
    Render the classical CV leaf contours (emerald green) and lesion masks
    (semi-transparent red fill + bright red outline) onto a copy of the image,
    returning JPEG bytes for Streamlit display.
    """
    canvas = image_rgb.copy()
    overlay = image_rgb.copy()
    h, w = canvas.shape[:2]

    if seg_result.get("found"):
        leaf_polys = seg_result.get("leaf_polygons") or (
            [seg_result["leaf_polygon"]] if seg_result.get("leaf_polygon") else []
        )
        for leaf_poly in leaf_polys:
            if len(leaf_poly) >= 3:
                pts = np.array([[int(px * w), int(py * h)] for px, py in leaf_poly], dtype=np.int32)
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=True,
                    color=(16, 185, 129),
                    thickness=max(2, int(min(h, w) * 0.006)),
                )

        lesions = seg_result.get("lesions", [])
        for l_poly in lesions:
            if len(l_poly) >= 3:
                l_pts = np.array([[int(px * w), int(py * h)] for px, py in l_poly], dtype=np.int32)
                cv2.fillPoly(overlay, [l_pts], color=(239, 68, 68))
                cv2.polylines(
                    canvas,
                    [l_pts],
                    isClosed=True,
                    color=(220, 38, 38),
                    thickness=max(1, int(min(h, w) * 0.004)),
                )

        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    pil_out = Image.fromarray(canvas)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def analyze_foliar_lesions(
    image: Image.Image,
    is_healthy: bool,
    plant_name_or_id: Optional[str] = None,
    urgency_or_severity: str = "medium",
) -> Dict[str, Any]:
    """
    End-to-end classical CV affected-area, severity, and derived health score analysis.
    """
    img = ImageOps.exif_transpose(image).convert("RGB")
    img_rgb = np.asarray(img)
    seg = segment_leaf(img_rgb)
    variegated = is_variegated_plant(plant_name_or_id)
    full_lesion_overlay_bytes = render_segmentation_overlay(img_rgb, seg)

    if variegated:
        affected_pct: Optional[float] = None
        estimation_status = (
            "Affected area could not be estimated (plant has naturally variegated or non-green foliage; "
            "color-threshold segmentation is disabled by default to prevent false lesion masks)."
        )
        seg_overlay = dict(seg, lesions=[])
        overlay_bytes = render_segmentation_overlay(img_rgb, seg_overlay)
    elif not seg["found"]:
        affected_pct = None
        estimation_status = "Affected area could not be estimated (leaf boundary could not be isolated from the background)."
        overlay_bytes = full_lesion_overlay_bytes
    else:
        affected_pct = float(seg["severity"])
        estimation_status = "Estimated via classical CV color-threshold segmentation"
        overlay_bytes = full_lesion_overlay_bytes

    severity_category = compute_severity_category(affected_pct, is_healthy=is_healthy)
    derived_health = compute_derived_health_score(
        affected_pct,
        is_healthy=is_healthy,
        urgency_or_severity=urgency_or_severity,
    )

    return {
        "segmentation_found": bool(seg["found"]),
        "is_variegated_foliage": variegated,
        "affected_area_pct": affected_pct,
        "raw_unfiltered_severity_pct": float(seg["severity"]) if seg.get("found") else None,
        "raw_unfiltered_lesion_count": int(seg.get("lesion_count", 0)) if seg.get("found") else 0,
        "estimation_status": estimation_status,
        "lesion_count": seg.get("lesion_count", 0) if not variegated else 0,
        "leaf_area_ratio": seg.get("leaf_area_ratio", 0.0),
        "bbox": seg.get("bbox"),
        "project_severity_category": severity_category,
        "severity_scheme_label": SEVERITY_SCHEME_LABEL,
        "derived_health_score": derived_health,
        "health_score_label": HEALTH_SCORE_LABEL,
        "method_label": SEGMENTATION_METHOD_LABEL,
        "limitations": SEGMENTATION_LIMITATIONS,
        "multiple_leaves_detected": seg.get("multiple_leaves_detected", False),
        "cluttered_background_detected": seg.get("cluttered_background_detected", False),
        "quality_warnings": seg.get("quality_warnings", []),
        "overlay_image_bytes": overlay_bytes,
        "overlay_with_lesions_bytes": full_lesion_overlay_bytes,
    }

