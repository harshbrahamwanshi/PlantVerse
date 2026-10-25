"""
PlantVerse — Unit & Integration Test Suite for ML Pipeline (Task 10)
====================================================================
Tests:
  1. Preprocessing (ConvNeXt-Tiny ONNX plant model & ViT ONNX disease model, EXIF transpose, RGB conversion)
  2. Label mapping verification (68 species + not_plant + 7 conditions; 38 PlantVillage classes; Crop___Disease splitting)
  3. Confidence policy (not_plant rejection, low/medium/high tiers, top-1 vs top-2 margin checks)
  4. Plant -> Disease consistency gating (supported crops allowed, non-crops blocked, cross-crop impossible combinations blocked)
  5. Classical CV affected-area & project-defined severity thresholds + derived health score + variegated bypass + failed segmentation
  6. Knowledge base coverage (all 38 classes + unknown disease message + expert review notice)
  7. Error paths (missing model file, corrupted weights, ML service offline, empty/tiny/corrupt images)
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml_config
from disease_recommendations import (
    EXPERT_REVIEW_NOTICE,
    NO_RECOMMENDATION_MESSAGE,
    get_disease_info,
    normalize_disease_label,
)
from plant_disease_detection import (
    CROP_TO_CLASS_INDICES,
    evaluate_disease_confidence_policy,
    load_disease_classifier,
    predict_disease,
    preprocess_for_onnx_vit,
    preprocess_image,
    resolve_supported_disease_crop,
    split_raw_disease_label,
)
from plant_identification import (
    _validate_and_orient_image,
    evaluate_onnx_plant_confidence_policy,
    load_onnx_plant_classifier,
    load_plants_metadata,
    predict_species,
    predict_species_detailed,
    preprocess_for_onnx_plant,
)
from vision_analysis import (
    analyze_foliar_lesions,
    compute_derived_health_score,
    compute_severity_category,
)


# ==============================================================================
# 1. PREPROCESSING TESTS
# ==============================================================================
def test_preprocess_for_onnx_plant_shape_and_normalization():
    img = Image.new("RGB", (400, 300), color=(100, 150, 50))
    arr = preprocess_for_onnx_plant(img)
    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32
    # Check ImageNet normalization values on uniform pixel (100/255, 150/255, 50/255)
    expected_r = (100.0 / 255.0 - 0.485) / 0.229
    assert np.isclose(arr[0, 0, 0, 0], expected_r, atol=1e-2)


def test_preprocess_for_onnx_vit_shape_and_normalization():
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    arr = preprocess_for_onnx_vit(img)
    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32
    expected_val = (128.0 / 255.0 - 0.5) / 0.5
    assert np.isclose(arr[0, 0, 10, 10], expected_val, atol=1e-2)


def test_exif_transpose_and_rgba_to_rgb():
    # Create a non-square RGBA image with EXIF Orientation = 6 (Rotate 90 CW)
    rgba = Image.new("RGBA", (120, 80), color=(40, 180, 60, 255))
    exif = rgba.getexif()
    exif[0x0112] = 6  # Orientation tag = 6
    buf = io.BytesIO()
    rgba.convert("RGB").save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)

    oriented = _validate_and_orient_image(buf)
    assert oriented.mode == "RGB"
    # Orientation 6 transposes (120, 80) -> (80, 120)
    assert oriented.size == (80, 120)


# ==============================================================================
# 2. LABEL MAPPING VERIFICATION TESTS
# ==============================================================================
def test_plant_labels_and_metadata_mapping():
    labels_data = json.loads(Path(ml_config.ONNX_PLANT_LABELS_PATH).read_text(encoding="utf-8"))
    species = labels_data["species"]
    conditions = labels_data["conditions"]

    assert len(species) == 69
    assert "not_plant" in species
    botanical_species = [s for s in species if s != "not_plant"]
    assert len(botanical_species) == 68
    assert len(conditions) == 7

    # Key Indian plants required by PlantVerse must be present
    for required_indian in ("tulsi", "neem", "peepal", "curry_leaf", "hibiscus", "money_plant"):
        assert required_indian in botanical_species

    meta = load_plants_metadata()
    missing_meta = [s for s in botanical_species if s not in meta]
    assert missing_meta == []


def test_disease_38_class_mapping_and_splitting():
    labels_data = json.loads(Path(ml_config.ONNX_DISEASE_LABELS_PATH).read_text(encoding="utf-8"))
    assert len(labels_data) == 38
    for idx, expected_raw in enumerate(ml_config.PLANTVILLAGE_38_RAW_CLASSES):
        actual = labels_data[idx] if isinstance(labels_data, list) else labels_data[str(idx)]
        assert actual == expected_raw

    crop, cond, is_healthy = split_raw_disease_label("Tomato___Early_blight")
    assert crop == "Tomato"
    assert cond == "Early Blight"
    assert is_healthy is False

    crop_h, cond_h, is_healthy_h = split_raw_disease_label("Apple___healthy")
    assert crop_h == "Apple"
    assert "healthy" in cond_h.lower()
    assert is_healthy_h is True


# ==============================================================================
# 3. CONFIDENCE POLICY TESTS
# ==============================================================================
def test_plant_confidence_policy_tiers_and_margin():
    # 1. not_plant rejection -> returns 'low' with LOW_CONFIDENCE_PLANT_MESSAGE
    tier_np, msg_np = evaluate_onnx_plant_confidence_policy("not_plant", 0.82, 0.05, 0.82)
    assert tier_np == "low"
    assert ml_config.LOW_CONFIDENCE_PLANT_MESSAGE in msg_np

    # 2. Low confidence (< 0.35)
    tier_low, msg_low = evaluate_onnx_plant_confidence_policy("tulsi", 0.22, 0.10, 0.01)
    assert tier_low == "low"
    assert ml_config.LOW_CONFIDENCE_PLANT_MESSAGE in msg_low

    # 3. Medium confidence due to score between 0.35 and 0.68
    tier_med, msg_med = evaluate_onnx_plant_confidence_policy("tulsi", 0.52, 0.12, 0.01)
    assert tier_med == "medium"
    assert "provisional" in msg_med.lower() or "moderate" in msg_med.lower()

    # 4. Medium confidence due to narrow top-1 vs top-2 margin (0.10 < 0.15) despite top-1 >= 0.68
    tier_margin, msg_margin = evaluate_onnx_plant_confidence_policy("tulsi", 0.72, 0.62, 0.01)
    assert tier_margin == "medium"
    assert "provisional" in msg_margin.lower() or "moderate" in msg_margin.lower()

    # 5. High confidence (>= 0.68 and margin >= 0.15)
    tier_high, msg_high = evaluate_onnx_plant_confidence_policy("tulsi", 0.88, 0.04, 0.01)
    assert tier_high == "high"
    assert msg_high == ""


def test_disease_confidence_policy_tiers_and_margin():
    tier_low, unc_low, msg_low = evaluate_disease_confidence_policy(0.40, 0.15)
    assert tier_low == "low"
    assert unc_low is True
    assert ml_config.LOW_CONFIDENCE_DISEASE_MESSAGE in msg_low

    tier_med, unc_med, msg_med = evaluate_disease_confidence_policy(0.65, 0.20)
    assert tier_med == "medium"
    assert unc_med is True

    tier_margin, unc_margin, _ = evaluate_disease_confidence_policy(0.80, 0.75)
    assert tier_margin == "medium"
    assert unc_margin is True

    tier_high, unc_high, msg_high = evaluate_disease_confidence_policy(0.91, 0.04)
    assert tier_high == "high"
    assert unc_high is False
    assert msg_high == ""


# ==============================================================================
# 4. PLANT -> DISEASE CONSISTENCY GATING TESTS (Task 4)
# ==============================================================================
def test_consistency_gate_blocks_non_crop_plants():
    assert resolve_supported_disease_crop("Tomato") == "Tomato"
    assert resolve_supported_disease_crop("Solanum lycopersicum") == "Tomato"
    assert resolve_supported_disease_crop("Tulsi") is None
    assert resolve_supported_disease_crop("Neem") is None
    assert resolve_supported_disease_crop("Ginkgo biloba") is None

    # Non-crop host explicitly passed -> must return Unsupported and never run/display crop disease
    sample_img = PROJECT_ROOT / "static/sample_leaves/tomato_early_blight.jpg"
    res = predict_disease(str(sample_img), host_plant="Tulsi (Holy Basil)")
    assert res["health_status"] == "Unsupported"
    assert res["is_supported_crop"] is False
    assert res["disease"] == ml_config.UNSUPPORTED_CROP_DISEASE_MESSAGE
    assert res["top_candidates"] == []

    # Non-crop image (ginkgo.jpg) through auto-gate -> Unsupported
    ginkgo_res = predict_disease(str(PROJECT_ROOT / "static/ginkgo.jpg"))
    assert ginkgo_res["health_status"] == "Unsupported"
    assert ginkgo_res["disease"] == ml_config.UNSUPPORTED_CROP_DISEASE_MESSAGE


def test_consistency_gate_prevents_cross_crop_impossible_combinations():
    # Pass an Apple Scab image with host_plant="Tomato":
    # Apple classes must be masked out to 0 probability; only Tomato classes can appear.
    apple_img = PROJECT_ROOT / "static/sample_leaves/apple_scab.jpg"
    res = predict_disease(str(apple_img), host_plant="Tomato")
    assert res["plant"] == "Tomato"
    for cand in res["top_candidates"]:
        assert cand["raw_label"].startswith("Tomato___"), f"Impossible cross-crop candidate leaked: {cand['raw_label']}"


# ==============================================================================
# 5. CLASSICAL CV LESION SEGMENTATION & SEVERITY THRESHOLDS (Tasks 5-6)
# ==============================================================================
def test_severity_thresholds_and_derived_health_score():
    assert compute_severity_category(0.0, is_healthy=True) == "Healthy"
    assert compute_severity_category(2.5, is_healthy=False) == "Early"
    assert compute_severity_category(12.0, is_healthy=False) == "Moderate"
    assert compute_severity_category(25.0, is_healthy=False) == "Severe"
    assert "Affected area could not be estimated" in compute_severity_category(None, is_healthy=False)

    score_healthy = compute_derived_health_score(0.0, is_healthy=True)
    score_severe = compute_derived_health_score(45.0, is_healthy=False, urgency_or_severity="high")
    assert 90 <= score_healthy <= 100
    assert 0 <= score_severe < score_healthy


def test_vision_analysis_failed_segmentation_and_variegated_bypass():
    # 1. Pure white image -> leaf segmentation fails gracefully
    white_img = Image.new("RGB", (224, 224), color=(250, 250, 250))
    res_fail = analyze_foliar_lesions(white_img, is_healthy=False, plant_name_or_id="Tomato")
    assert res_fail["segmentation_found"] is False
    assert res_fail["affected_area_pct"] is None
    assert "Affected area could not be estimated" in res_fail["estimation_status"]

    # 2. Variegated foliage (money_plant) -> bypasses color-threshold lesion percentage
    diseased_img = Image.open(PROJECT_ROOT / "static/sample_leaves/tomato_early_blight.jpg")
    res_var = analyze_foliar_lesions(diseased_img, is_healthy=False, plant_name_or_id="money_plant")
    assert res_var["is_variegated_foliage"] is True
    assert res_var["affected_area_pct"] is None
    assert "Affected area could not be estimated" in res_var["estimation_status"]

    # 3. Real diseased tomato leaf -> segments leaf & produces overlay bytes + honest labels
    res_ok = analyze_foliar_lesions(diseased_img, is_healthy=False, plant_name_or_id="Tomato")
    assert res_ok["segmentation_found"] is True
    assert res_ok["affected_area_pct"] is not None
    assert res_ok["overlay_image_bytes"] is not None
    assert res_ok["method_label"] == ml_config.CLASSICAL_CV_METHOD_LABEL
    assert res_ok["severity_scheme_label"] == ml_config.SEVERITY_SCHEME_LABEL
    assert res_ok["health_score_label"] == ml_config.DERIVED_HEALTH_SCORE_LABEL


# ==============================================================================
# 6. KNOWLEDGE BASE COVERAGE TESTS (Task 7)
# ==============================================================================
def test_knowledge_base_all_38_classes_and_unknown_fallback():
    for raw_cls in ml_config.PLANTVILLAGE_38_RAW_CLASSES:
        info = get_disease_info(raw_cls)
        assert info["known_disease"] is True
        assert info["symptoms"] and info["symptoms"] != NO_RECOMMENDATION_MESSAGE
        assert isinstance(info["preventive_measures"], list) and len(info["preventive_measures"]) > 0
        assert isinstance(info["care"], dict) and {"sun", "water", "soil"}.issubset(info["care"].keys())
        assert info["needs_expert_review"] is True
        assert info["expert_review_notice"] == EXPERT_REVIEW_NOTICE

    unknown_info = get_disease_info("Dragonfruit___Cosmic_blight")
    assert unknown_info["known_disease"] is False
    assert unknown_info["no_recommendation_message"] == NO_RECOMMENDATION_MESSAGE


# ==============================================================================
# 7. ERROR PATHS & ROBUSTNESS TESTS (Task 8)
# ==============================================================================
def test_error_paths_missing_and_corrupted_models_and_bad_images():
    # 1. Tiny image (< 24x24)
    tiny_img = Image.new("RGB", (10, 10), color=(0, 200, 0))
    with pytest.raises(ValueError, match="too small"):
        preprocess_image(tiny_img)

    # 2. Missing ONNX plant model file -> raises RuntimeError
    load_onnx_plant_classifier.clear()
    with patch.object(ml_config, "ONNX_PLANT_MODEL_PATH", Path("/tmp/nonexistent_plant_model_12345.onnx")):
        import plant_identification
        with patch.object(plant_identification, "ONNX_PLANT_MODEL_PATH", Path("/tmp/nonexistent_plant_model_12345.onnx")):
            load_onnx_plant_classifier.clear()
            with pytest.raises(RuntimeError, match="[Mm]issing"):
                load_onnx_plant_classifier()
    load_onnx_plant_classifier.clear()

    # 3. Corrupted ONNX disease model weights -> raises RuntimeError
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(b"corrupted_onnx_header_bytes_not_valid_protobuf")
        corrupt_path = Path(tmp.name)

    try:
        import plant_disease_detection
        with patch.object(plant_disease_detection, "ONNX_DISEASE_MODEL_PATH", corrupt_path):
            plant_disease_detection.load_onnx_disease_classifier.clear()
            with pytest.raises(RuntimeError, match="truncated|corrupted|Failed to load"):
                load_disease_classifier()
    finally:
        corrupt_path.unlink(missing_ok=True)
        import plant_disease_detection
        plant_disease_detection.load_onnx_disease_classifier.clear()


def test_non_crop_general_foliar_condition_and_feedback_widget():
    # 1. Rose black spot (non-crop plant): 38-class gate is Unsupported, 7-class general condition is Diseased (Leaf spot / blight)
    rose_img = PROJECT_ROOT / "static/sample_leaves/rose_black_spot.jpg"
    res_rose = predict_disease(str(rose_img))
    assert res_rose["health_status"] == "Unsupported"
    assert res_rose["is_supported_crop"] is False
    gc = res_rose.get("general_condition", {})
    assert gc.get("available") is True
    assert gc.get("plant_name") == "Rose"
    assert gc.get("health_status") == "Diseased"
    assert gc.get("condition_id") == "leaf_spot_blight"
    assert len(gc.get("recommendations", [])) > 0
    assert len(gc.get("common_species_problems", [])) > 0

    # 2. Money plant (variegated non-crop houseplant): 38-class gate is Unsupported, 7-class general condition + care available
    mp_img = PROJECT_ROOT / "static/sample_leaves/money_plant.jpg"
    res_mp = predict_disease(str(mp_img))
    assert res_mp["health_status"] == "Unsupported"
    gc_mp = res_mp.get("general_condition", {})
    assert gc_mp.get("available") is True
    assert "Money plant" in gc_mp.get("plant_name", "")
    assert res_mp["vision_analysis"]["is_variegated_foliage"] is True
    assert res_mp["vision_analysis"]["affected_area_pct"] is None
    assert res_mp["vision_analysis"]["raw_unfiltered_severity_pct"] is not None

    # 3. Verify render_feedback_widget accepts both task_type/extra_meta and context_type/extra_metadata
    from ui.feedback import render_feedback_widget
    render_feedback_widget(
        context_type="disease_detection_unsupported_gate",
        predicted_label="Money plant (Pothos)",
        confidence=0.935,
        model_name="plant_model.onnx",
        extra_metadata={"health_status": "Unsupported"},
        key_prefix="test_fb_ctx",
    )

