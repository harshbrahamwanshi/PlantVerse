import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

from ar_location_plant_map import PlantLocationFinder
from disease_recommendations import get_disease_info
from ml_config import (
    CLASSICAL_CV_LIMITATIONS,
    CLASSICAL_CV_METHOD_LABEL,
    DERIVED_HEALTH_SCORE_LABEL,
    EXPERT_REVIEW_NOTICE,
    LOW_CONFIDENCE_DISEASE_MESSAGE,
    NO_RECOMMENDATION_MESSAGE,
    SEVERITY_SCHEME_LABEL,
    UNSUPPORTED_CROP_DISEASE_MESSAGE,
    load_plants_metadata,
)
from plant_disease_detection import get_supported_disease_crops_list, predict_disease
from plant_identification import LANGUAGES, translate_text
from ui.chrome import definition_row, init_state, kicker, section_rule
from ui.feedback import render_feedback_widget
from ui.presentation import upsert_disease_diagnosis, upsert_library_entry

init_state()
finder = PlantLocationFinder()

# ==============================================================================
# HEADER & MULTILINGUAL LOCALIZATION SELECTOR
# ==============================================================================
top_left, top_right = st.columns([1.35, 0.65], vertical_alignment="top")

with top_left:
    st.markdown(
        """<div class="pv-academic-badge">
            <span>🔬</span>
            <span>Plant Pathology Engine • Dual-Stage Crop & Foliar Diagnostic Studio</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.header("Plant Disease Detection Studio")
    st.markdown(
        "Screen agricultural crops, medicinal herbs, trees, and indoor houseplants for foliar infections, "
        "blights, rusts, chlorosis, and pests. The pipeline enforces **Host-Crop Consistency Gating** for the "
        "38-class agricultural ViT model (`models/model.onnx`) and provides **7-Class General Foliar Condition & "
        "Species Care Diagnostics** (`models/plant_model.onnx`) across all 68 supported botanical species."
    )
    supported_crops = get_supported_disease_crops_list()
    plants_meta_all = load_plants_metadata()
    non_crop_plants = sorted(
        [p["name"] for p in plants_meta_all.values() if not p.get("pv_crop")]
    )
    with st.expander(":material/agriculture: Supported hosts (14 agricultural crops + 54 regional/indoor species)", expanded=False):
        st.markdown(
            "**14 Agricultural Crops (38-class ViT Crop Pathology + 7-class General Condition):** "
            + ", ".join(supported_crops)
        )
        st.markdown(
            "**54 Regional, Medicinal & Indoor Species (Host-Crop Gate + 7-class General Foliar Condition & Cultural Care):** "
            + ", ".join(non_crop_plants)
        )

with top_right:
    with st.container(border=True):
        selected_lang = st.selectbox(
            "Diagnostic Language",
            list(LANGUAGES.keys()),
            key="disease_language",
            help="Select evaluation language for translation of plant names, pathology, and recommendations.",
        )
        lang_code = LANGUAGES[selected_lang]
        st.caption(f":material/translate: Active Language: **{selected_lang}**")

st.space("small")
section_rule("section-rule-disease-input")
st.space("small")

# ==============================================================================
# SAMPLE SPECIMENS SHORTCUT BAR
# ==============================================================================
st.caption(translate_text("Quick Evaluation Benchmark Specimens (Crops & Houseplants):", lang_code))
sample_cols = st.container(horizontal=True, gap="small")
sample_choice: Optional[str] = None
with sample_cols:
    if st.button("🍅 " + translate_text("Tomato Early Blight", lang_code), key="sample_eb"):
        sample_choice = "static/sample_leaves/tomato_early_blight.jpg"
    if st.button("🥔 " + translate_text("Potato Late Blight", lang_code), key="sample_plb"):
        sample_choice = "static/sample_leaves/Potato_Late_blight_5305.jpg"
    if st.button("🍏 " + translate_text("Apple Scab", lang_code), key="sample_scab"):
        sample_choice = "static/sample_leaves/apple_scab.jpg"
    if st.button("🌿 " + translate_text("Healthy Tomato", lang_code), key="sample_htomato"):
        sample_choice = "static/sample_leaves/healthy_tomato.jpg"
    if st.button("🌹 " + translate_text("Rose Black Spot", lang_code), key="sample_rose_bs"):
        sample_choice = "static/sample_leaves/rose_black_spot.jpg"
    if st.button("🪴 " + translate_text("Money Plant (Pothos)", lang_code), key="sample_mplant"):
        sample_choice = "static/sample_leaves/money_plant.jpg"
    if st.button("🌱 " + translate_text("Tulsi (Holy Basil)", lang_code), key="sample_tulsi"):
        sample_choice = "static/sample_leaves/tulsi.jpg"

if sample_choice and os.path.exists(sample_choice):
    st.session_state.disease_sample_path = sample_choice
    st.session_state.disease_file_id = f"sample-{sample_choice}"
    st.session_state.disease_result = None

active_sample = st.session_state.get("disease_sample_path")

st.space("small")

# ==============================================================================
# UPLOAD STUDIO & IMAGING PROTOCOL
# ==============================================================================
upload_col, guide_col = st.columns([1.08, 0.92], gap="large")

with upload_col:
    with st.container(border=True):
        st.markdown(f"#### {translate_text('Foliar Specimen Acquisition', lang_code)}")
        st.caption(
            translate_text(
                "Upload a high-resolution photograph or capture a live camera frame of the affected leaf or foliage.",
                lang_code,
            )
        )

        acq_mode = st.radio(
            translate_text("Input Source", lang_code),
            [
                translate_text("Upload Image File", lang_code),
                translate_text("Capture via Camera", lang_code),
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="disease_acq_mode",
        )

        img_file = None
        if "Camera" in acq_mode:
            img_file = st.camera_input(
                translate_text("Capture leaf specimen", lang_code),
                key="disease_camera_input",
            )
        else:
            with st.container(key="disease-upload"):
                img_file = st.file_uploader(
                    translate_text("Leaf photograph", lang_code),
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                    key="disease_uploader",
                )

        host_options = ["Auto-detect plant species (Recommended)"] + supported_crops + non_crop_plants
        selected_host_option = st.selectbox(
            translate_text("Host Plant Override (Optional)", lang_code),
            host_options,
            index=0,
            key="disease_host_override",
            help=translate_text(
                "Leave on Auto-detect to identify the plant via ConvNeXt-Tiny (`plant_model.onnx`), "
                "or explicitly specify the host plant to gate pathology candidates.",
                lang_code,
            ),
        )
        explicit_host: Optional[str] = (
            None if selected_host_option.startswith("Auto-detect") else selected_host_option
        )

        preview_image = None
        preview_caption = ""

        if img_file:
            st.session_state.disease_sample_path = None
            try:
                if hasattr(img_file, "seek"):
                    img_file.seek(0)
                preview_image = Image.open(img_file)
                if hasattr(img_file, "seek"):
                    img_file.seek(0)
                preview_caption = f"{getattr(img_file, 'name', 'camera_capture.jpg')} • Dimensions: {preview_image.width} × {preview_image.height} px"
            except Exception as exc:
                st.error(f"Unable to read image file: {exc}")
                st.stop()
        elif active_sample and os.path.exists(active_sample):
            try:
                preview_image = Image.open(active_sample)
                preview_caption = f"Benchmark Sample: {os.path.basename(active_sample)} • Dimensions: {preview_image.width} × {preview_image.height} px"
            except Exception as exc:
                st.error(f"Unable to open sample image: {exc}")

        if preview_image:
            st.image(
                preview_image,
                caption=preview_caption,
                width="stretch",
                alt=translate_text("Uploaded foliar specimen photograph", lang_code),
            )

        st.space("small")
        has_input = img_file is not None or (active_sample and os.path.exists(active_sample))
        run_diagnose = st.button(
            translate_text("Execute Pathology Diagnosis", lang_code),
            type="primary",
            icon=":material/biotech:",
            disabled=not has_input,
        )

with guide_col:
    with st.container(border=True):
        st.markdown(f"#### :material/science: {translate_text('Pathology Diagnostic Protocol', lang_code)}")
        st.markdown(
            translate_text(
                "Accurate identification of fungal, bacterial, viral, and nutritional pathologies requires clear visualization "
                "of diagnostic lesions, halos, and margins. Observe the following photographic standards:",
                lang_code,
            )
        )
        st.space("small")

        with st.container(border=True):
            st.markdown(f"**:material/center_focus_strong: {translate_text('Macro Lesion Framing', lang_code)}**")
            st.markdown(
                translate_text(
                    "Capture clear close-ups of a single leaf showing lesions, chlorotic yellowing, rust pustules, or powdery growth.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/flip_to_back: {translate_text('Inspect Upper & Lower Lamina', lang_code)}**")
            st.markdown(
                translate_text(
                    "Many pathogens and pests (e.g. Downy Mildew, Spider Mites, Mealybugs) cluster on leaf undersides.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/wb_sunny: {translate_text('Uniform Glare-Free Background', lang_code)}**")
            st.markdown(
                translate_text(
                    "Place the leaf against a plain neutral surface in diffuse daylight to maximize classical CV lesion boundary accuracy.",
                    lang_code,
                )
            )

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
current_input_id = (
    f"{getattr(img_file, 'name', 'cam')}-{getattr(img_file, 'size', 0)}-{explicit_host or 'auto'}"
    if img_file
    else f"{active_sample or ''}-{explicit_host or 'auto'}"
)
if st.session_state.get("disease_file_id") != current_input_id:
    st.session_state.disease_result = None
    st.session_state.disease_file_id = current_input_id

if has_input and run_diagnose:
    input_source = img_file if img_file else active_sample
    if hasattr(input_source, "seek"):
        input_source.seek(0)

    with st.status(translate_text("Processing specimen through neural pathology pipeline...", lang_code), expanded=True) as status:
        st.write(translate_text("1. Validating visual morphology, EXIF orientation, and host-crop consistency gate...", lang_code))

        try:
            st.write(
                translate_text(
                    "2. Running dual-stage diagnosis (38-class ViT crop pathology + 7-class ConvNeXt foliar condition & classical CV lesion segmentation)...",
                    lang_code,
                )
            )
            prediction_result = predict_disease(input_source, host_plant=explicit_host)
        except Exception as exc:
            status.update(label="Inference execution halted", state="error")
            st.error(f"Diagnosis failed: {exc}")
            st.stop()

        st.write(translate_text("3. Synthesizing agronomic recommendations, cultural care, and pathogen profile...", lang_code))
        st.session_state.disease_result = prediction_result

        if prediction_result.get("health_status") != "Unsupported":
            upsert_disease_diagnosis({
                "plant_name": prediction_result["plant"],
                "disease_name": prediction_result["disease_name"],
                "health_status": prediction_result["health_status"],
                "confidence": prediction_result["confidence"],
                "confidence_pct": prediction_result["confidence_pct"],
                "pathogen": prediction_result["pathogen"],
                "severity": prediction_result["severity"],
                "symptoms": prediction_result["symptoms"],
                "recommendations": prediction_result["recommendations"],
                "image_bytes": prediction_result["image_bytes"],
                "is_sighting": False,
            })
        elif prediction_result.get("general_condition", {}).get("available"):
            gc = prediction_result["general_condition"]
            upsert_disease_diagnosis({
                "plant_name": gc["plant_name"],
                "disease_name": gc["full_diagnosis_title"],
                "health_status": gc["health_status"],
                "confidence": gc["confidence"],
                "confidence_pct": gc["confidence_pct"],
                "pathogen": gc["pathogen"],
                "severity": prediction_result.get("severity", "N/A"),
                "symptoms": gc["symptoms"],
                "recommendations": gc["recommendations"],
                "image_bytes": prediction_result["image_bytes"],
                "is_sighting": False,
            })

        status.update(label=translate_text("Pathology diagnosis completed successfully!", lang_code), state="complete")

result = st.session_state.get("disease_result")
if not result:
    st.stop()

st.space("large")
section_rule("section-rule-disease-result")
st.space("small")

# ==============================================================================
# QUALITY WARNINGS (Multiple Leaves / Cluttered Background)
# ==============================================================================
for q_warn in result.get("quality_warnings", []):
    st.warning(f":material/photo_camera: {translate_text(q_warn, lang_code)}")

# ==============================================================================
# DETERMINE ACTIVE DIAGNOSTIC MODE:
#   Mode A: Supported 14-Crop Agricultural Pathology (38-class ViT primary)
#   Mode B: Non-Crop Specimen Gate + 7-Class General Foliar Condition & Plant Care
# ==============================================================================
is_unsupported_crop = (
    result.get("health_status") == "Unsupported" or not result.get("is_supported_crop", True)
)
gen_cond: Dict[str, Any] = result.get("general_condition") or {}

if is_unsupported_crop:
    kicker(translate_text("Host Crop Consistency Gate", lang_code), key="kicker-disease-unsupported")
    with st.container(border=True):
        st.info(
            f":material/info: **{translate_text(result.get('uncertainty_message') or UNSUPPORTED_CROP_DISEASE_MESSAGE, lang_code)}** "
            f"— *{translate_text('Detected Specimen', lang_code)}:* **{translate_text(gen_cond.get('plant_name') or result.get('plant', 'Unknown'), lang_code)}**"
        )
        st.caption(
            translate_text(
                "The 38-class agricultural crop ViT classifier (`models/model.onnx`) only evaluates 14 supported crop hosts "
                "(Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato). "
                "Cross-crop predictions are blocked by the Consistency Gate to prevent impossible crop diagnoses.",
                lang_code,
            )
        )
        if not gen_cond.get("available"):
            u_col1, u_col2 = st.columns([0.85, 1.15], gap="large", vertical_alignment="center")
            with u_col1:
                st.image(
                    result["image_bytes"],
                    alt=translate_text("Uploaded specimen photograph", lang_code),
                    width="stretch",
                )
            with u_col2:
                st.warning(
                    translate_text(
                        "No reliable botanical specimen could be identified in this image. "
                        "Please upload a clear close-up photograph of a plant leaf.",
                        lang_code,
                    )
                )
                render_feedback_widget(
                    task_type="disease_detection_unsupported_gate",
                    predicted_label=f"{result.get('plant', 'Unknown')} — {UNSUPPORTED_CROP_DISEASE_MESSAGE}",
                    confidence=0.0,
                    model_name=result.get("model_name", "consistency_gate"),
                    extra_meta={"health_status": "Unsupported"},
                    key_prefix="fb_disease_unsupported",
                )
            st.stop()
        else:
            st.success(
                f":material/neurology: **{translate_text('General Foliar Condition & Plant Care Head Active', lang_code)}:** "
                + translate_text(
                    "Diagnosing foliar health using the 7-class foliar condition head of `models/plant_model.onnx` "
                    "(Healthy, Leaf Spot / Blight, Powdery Mildew, Rust, Sooty Mould, Mosaic / Yellowing, Pest Infestation) "
                    "combined with species-specific pathology & care metadata.",
                    lang_code,
                )
            )
    st.space("medium")

# Build unified display fields for either Mode A (Supported Crop) or Mode B (General Foliar Condition)
if is_unsupported_crop and gen_cond.get("available"):
    disp_plant = gen_cond["plant_name"]
    disp_scientific = gen_cond.get("scientific_name", "")
    disp_hindi = gen_cond.get("hindi_name", "")
    disp_category = gen_cond.get("category", "Botanical Specimen")
    disp_condition = gen_cond["condition_name"]
    disp_full_title = gen_cond["full_diagnosis_title"]
    disp_health_status = gen_cond["health_status"]
    disp_is_healthy = gen_cond["is_healthy"]
    disp_conf = gen_cond["confidence"]
    disp_conf_pct = gen_cond["confidence_pct"]
    disp_conf_tier = gen_cond["confidence_tier"]
    disp_pathogen = gen_cond["pathogen"]
    disp_type = gen_cond.get("type", "General foliar condition")
    disp_urgency = gen_cond.get("urgency", "medium")
    disp_symptoms = gen_cond["symptoms"]
    disp_recs = gen_cond["recommendations"]
    disp_org = gen_cond["organic_treatments"]
    disp_chem = gen_cond["chemical_treatments"]
    disp_prev = gen_cond["preventive_measures"]
    disp_care = gen_cond["care"]
    disp_common_problems = gen_cond.get("common_species_problems", [])
    disp_uses = gen_cond.get("uses", [])
    disp_toxic = gen_cond.get("toxic", "")
    disp_model_name = gen_cond["model_name"]
    disp_raw_label = f"{gen_cond.get('plant_id', 'plant')}___{gen_cond.get('condition_id', 'condition')}"
    disp_kicker = "Model Output • General Foliar Condition & Species Care Diagnosis"
else:
    disp_plant = result["plant"]
    disp_scientific = result.get("scientific_name") or gen_cond.get("scientific_name", "")
    disp_hindi = result.get("hindi_name") or gen_cond.get("hindi_name", "")
    disp_category = gen_cond.get("category", "Agricultural Crop")
    disp_condition = result["disease"]
    disp_full_title = result["disease_name"]
    disp_health_status = result["health_status"]
    disp_is_healthy = disp_health_status == "Healthy"
    disp_conf = result["confidence"]
    disp_conf_pct = result["confidence_pct"]
    disp_conf_tier = result.get("confidence_tier", "medium")
    disp_pathogen = result["pathogen"]
    disp_type = "Crop-Specific Pathology (38-Class ViT)"
    disp_urgency = result.get("agronomic_severity_note", "Medium")
    disp_symptoms = result["symptoms"]
    disp_recs = result.get("recommendations", [])
    disp_org = result.get("organic_treatments", [])
    disp_chem = result.get("chemical_treatments", [])
    disp_prev = result.get("preventive_measures", [])
    disp_care = result.get("care") or gen_cond.get("care", {})
    disp_common_problems = result.get("common_species_problems") or gen_cond.get("common_species_problems", [])
    disp_uses = result.get("uses") or gen_cond.get("uses", [])
    disp_toxic = gen_cond.get("toxic", "")
    disp_model_name = result.get("model_name", "ViT-Base ONNX (models/model.onnx)")
    disp_raw_label = result.get("raw_label", "crop_disease")
    disp_kicker = "Model Output • Neural Pathology Classification"

# ==============================================================================
# SECTION 1: MODEL OUTPUT — PATHOLOGY DOSSIER HERO CARD
# ==============================================================================
kicker(translate_text(disp_kicker, lang_code), key="kicker-disease-result")

with st.container(border=True):
    image_col, diagnosis_col = st.columns([0.92, 1.08], gap="large", vertical_alignment="center")

    with image_col:
        with st.container(key="profile-image"):
            st.image(
                result["image_bytes"],
                alt=translate_text("Diagnosed leaf specimen photograph", lang_code),
                width="stretch",
            )
        st.caption(
            f":material/verified: {translate_text('Evaluated foliar specimen photograph', lang_code)} "
            f"• Model: `{disp_model_name}`"
        )

    with diagnosis_col:
        if disp_conf_tier == "low":
            status_badge = f":orange-badge[? {translate_text('Low Confidence — Tentative Foliar Assessment', lang_code)}]"
        elif disp_is_healthy:
            status_badge = f":green-badge[✓ {translate_text('Healthy Foliage — No Active Pathology', lang_code)}]"
        elif disp_conf_tier == "high":
            status_badge = f":red-badge[⚠ {translate_text('Foliar Condition Detected — High Confidence', lang_code)}]"
        else:
            status_badge = f":orange-badge[⚠ {translate_text('Foliar Condition Detected — Moderate Confidence', lang_code)}]"

        st.markdown(status_badge)

        st.markdown(f"## {translate_text(disp_plant, lang_code)} — {translate_text(disp_condition, lang_code)}")
        subtitle_parts = []
        if disp_scientific and disp_scientific.lower() != disp_plant.lower():
            subtitle_parts.append(f"*{disp_scientific}*")
        if disp_hindi:
            subtitle_parts.append(f"**{disp_hindi}**")
        if disp_category:
            subtitle_parts.append(f"`{translate_text(disp_category, lang_code)}`")
        if subtitle_parts:
            st.markdown(" • ".join(subtitle_parts))

        st.markdown(f"**{translate_text('Causal Agent / Group', lang_code)}:** *{disp_pathogen}*")

        if disp_conf_tier == "low":
            st.warning(
                f":material/warning: **{translate_text(result.get('uncertainty_message') if not is_unsupported_crop else 'Low condition confidence (< 35%): verify leaf symptoms against common species issues below.', lang_code)}**"
            )
        elif not is_unsupported_crop and result.get("is_uncertain") and result.get("uncertainty_message"):
            st.warning(f":material/warning: {translate_text(result['uncertainty_message'], lang_code)}")

        if disp_toxic:
            st.warning(f":material/pets: **{translate_text('Safety / Toxicity Note', lang_code)}:** {translate_text(disp_toxic, lang_code)}")

        st.space("small")
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        with stat_c1:
            st.metric(
                label=translate_text("Condition Confidence", lang_code),
                value=f"{disp_conf_pct:.1f}%",
                help="Softmax probability output by the active neural condition head.",
            )
        with stat_c2:
            st.metric(
                label=translate_text("Classified State", lang_code),
                value=translate_text(disp_health_status, lang_code),
                help="Separation between healthy foliage and pathological/stress condition.",
            )
        with stat_c3:
            st.metric(
                label=translate_text("Condition Type", lang_code),
                value=translate_text(disp_type.split("(")[0].strip(), lang_code),
                help="Etiological category of the detected foliar condition.",
            )

        st.space("small")
        action_row = st.container(horizontal=True, gap="small")
        with action_row:
            if st.button(
                translate_text("Locate Host Species on Map", lang_code),
                icon=":material/map:",
            ):
                st.session_state.ar_query = disp_plant
                st.switch_page("app_pages/ar_explorer.py")
            if st.button(
                translate_text("Inspect Specimen in Library", lang_code),
                icon=":material/menu_book:",
            ):
                st.session_state.library_profile = None
                st.switch_page("app_pages/library.py")

st.space("medium")

# ==============================================================================
# SECTION 2: CLASSICAL CV AFFECTED-AREA & DERIVED METRICS (Visually Separated)
# ==============================================================================
vision_info = result.get("vision_analysis") or {}
kicker(
    translate_text("Classical CV Lesion Segmentation & Project-Defined Derived Metrics", lang_code),
    key="kicker-vision-analysis",
)

with st.container(border=True):
    st.caption(
        f":material/tune: **Method:** {translate_text(vision_info.get('method_label', CLASSICAL_CV_METHOD_LABEL), lang_code)}"
    )
    ov_col, met_col = st.columns([0.92, 1.08], gap="large", vertical_alignment="center")

    is_variegated = bool(vision_info.get("is_variegated_foliage", False))
    show_variegated_raw = False
    if is_variegated and vision_info.get("segmentation_found"):
        show_variegated_raw = st.checkbox(
            translate_text(
                "Preview raw color-threshold spot/yellowing mask on variegated foliage "
                "(Note: may include natural golden/yellow variegation on Money Plant, Croton, or Snake Plant)",
                lang_code,
            ),
            value=True,
            key="toggle_variegated_raw_mask",
        )

    with ov_col:
        overlay_bytes = (
            vision_info.get("overlay_with_lesions_bytes")
            if (is_variegated and show_variegated_raw and vision_info.get("overlay_with_lesions_bytes"))
            else vision_info.get("overlay_image_bytes")
        )
        if overlay_bytes:
            st.image(
                overlay_bytes,
                caption=translate_text(
                    "Classical CV Overlay (Green contour = segmented leaf boundary; Red tint = thresholded chlorotic/necrotic spot mask)",
                    lang_code,
                ),
                width="stretch",
                alt=translate_text("Classical CV leaf and lesion segmentation overlay", lang_code),
            )
        else:
            st.info(translate_text("Lesion overlay unavailable: leaf boundary could not be segmented.", lang_code))

    with met_col:
        affected_pct = result.get("affected_area_pct")
        raw_unfiltered_pct = vision_info.get("raw_unfiltered_severity_pct")
        affected_status = result.get("affected_area_status", "")
        proj_severity = result.get("severity", "Unavailable")
        health_score = result.get("derived_health_score")

        v_c1, v_c2, v_c3 = st.columns(3)
        with v_c1:
            if affected_pct is not None:
                area_val = f"{affected_pct:.1f}%"
            elif is_variegated and show_variegated_raw and raw_unfiltered_pct is not None:
                area_val = f"~{raw_unfiltered_pct:.1f}%*"
            else:
                area_val = "N/A"
            st.metric(
                label=translate_text("Affected Leaf Area", lang_code),
                value=area_val,
                help=vision_info.get("method_label", CLASSICAL_CV_METHOD_LABEL),
            )
        with v_c2:
            st.metric(
                label=translate_text("Severity Category", lang_code),
                value=translate_text(proj_severity, lang_code),
                help=result.get("severity_scheme_label", SEVERITY_SCHEME_LABEL),
            )
        with v_c3:
            st.metric(
                label=translate_text("Foliar Health Score", lang_code),
                value=f"{health_score}/100" if health_score is not None else "N/A",
                help=result.get("health_score_label", DERIVED_HEALTH_SCORE_LABEL),
            )

        if affected_pct is None:
            st.warning(f":material/info: **{translate_text(affected_status or 'Affected area could not be estimated', lang_code)}**")
            if is_variegated and raw_unfiltered_pct is not None:
                st.caption(
                    f":material/visibility: *{translate_text('Raw color-threshold yellow/brown area on leaf', lang_code)}: "
                    f"**{raw_unfiltered_pct:.1f}%** ({vision_info.get('raw_unfiltered_lesion_count', 0)} regions). "
                    f"{translate_text('Excluded from official severity score because natural leaf variegation shares yellow/cream HSV hues.', lang_code)}*"
                )
        else:
            st.caption(f":material/check_circle: {translate_text(affected_status, lang_code)}")

        st.markdown(
            f"- **Severity Scale Notice:** _{translate_text(result.get('severity_scheme_label', SEVERITY_SCHEME_LABEL), lang_code)}_ "
            f"(`Healthy`: 0%, `Early`: <5%, `Moderate`: 5–20%, `Severe`: ≥20% affected leaf area)\n"
            f"- **Health Score Notice:** _{translate_text(result.get('health_score_label', DERIVED_HEALTH_SCORE_LABEL), lang_code)}_"
        )

        with st.expander(translate_text("Known Limitations of Color-Threshold Classical CV Segmentation", lang_code), expanded=False):
            for lim in vision_info.get("limitations", CLASSICAL_CV_LIMITATIONS):
                st.markdown(f"- {translate_text(lim, lang_code)}")

st.space("medium")

# ==============================================================================
# MULTIMODAL PATHOLOGY DETAILS TABS
# ==============================================================================
tab_recs, tab_symp, tab_cand, tab_sight = st.tabs(
    [
        translate_text("💊 Actionable Recommendations & Host Care", lang_code),
        translate_text("🔬 Symptoms & Pathogen Profile", lang_code),
        translate_text("📊 Probability Distributions", lang_code),
        translate_text("📍 Save as Field Sighting", lang_code),
    ]
)

# TAB 1: RECOMMENDATIONS & HOST CARE (Task 7 + Species Care & Common Problems)
with tab_recs:
    st.space("small")
    st.markdown(f"#### {translate_text('Targeted Interventions & Cultural Management', lang_code)}")

    st.info(f":material/verified_user: **{translate_text(result.get('expert_review_notice', EXPERT_REVIEW_NOTICE), lang_code)}**")

    if disp_recs:
        with st.container(border=True):
            st.markdown(f"**{translate_text('Recommended Action Plan & Cultural Practices', lang_code)}**")
            for idx, item in enumerate(disp_recs, 1):
                st.markdown(f"**{idx}.** {translate_text(item, lang_code)}")
    else:
        st.info(translate_text("No specific disease remediation required for this healthy specimen.", lang_code))

    if disp_chem or disp_org:
        st.space("small")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            with st.container(border=True):
                st.markdown(f"**:material/eco: {translate_text('Organic & Biological Controls', lang_code)}**")
                if disp_org:
                    for item in disp_org:
                        st.markdown(f"- {translate_text(item, lang_code)}")
                else:
                    st.caption(translate_text("No organic spray required when foliage is healthy.", lang_code))
        with t_col2:
            with st.container(border=True):
                st.markdown(f"**:material/science: {translate_text('Conventional / Chemical Controls', lang_code)}**")
                if disp_chem:
                    for item in disp_chem:
                        st.markdown(f"- {translate_text(item, lang_code)}")
                else:
                    st.caption(translate_text("No chemical fungicide/pesticide required.", lang_code))

    if disp_common_problems:
        st.space("small")
        with st.container(border=True):
            st.markdown(
                f"**:material/troubleshoot: {translate_text(f'Common Foliar & Root Problems in {disp_plant}', lang_code)}**"
            )
            st.caption(
                translate_text(
                    "Species-specific diagnostic checklist (verify these frequent causes of yellowing, spots, or wilting on this host):",
                    lang_code,
                )
            )
            for prob in disp_common_problems:
                st.markdown(f"- {translate_text(prob, lang_code)}")

    if disp_care:
        st.space("small")
        with st.container(border=True):
            st.markdown(f"**:material/yard: {translate_text('Host Plant Cultural Care Baseline', lang_code)}**")
            c_sun, c_wat, c_soil = st.columns(3)
            with c_sun:
                st.markdown(f"**☀️ {translate_text('Sunlight', lang_code)}:** {translate_text(disp_care.get('sun', 'N/A'), lang_code)}")
            with c_wat:
                st.markdown(f"**💧 {translate_text('Watering', lang_code)}:** {translate_text(disp_care.get('water', 'N/A'), lang_code)}")
            with c_soil:
                st.markdown(f"**🌱 {translate_text('Soil', lang_code)}:** {translate_text(disp_care.get('soil', 'N/A'), lang_code)}")

    st.space("small")
    st.caption(
        ":material/health_and_safety: **" + translate_text("Agronomic Advisory", lang_code) + "**: "
        + translate_text(
            "Recommendations are provided strictly for educational and informational purposes. "
            "Always consult certified agronomists and adhere to manufacturer chemical label requirements before applying chemical treatments.",
            lang_code,
        )
    )

# TAB 2: SYMPTOMS & PATHOGEN PROFILE
with tab_symp:
    st.space("small")
    st.markdown(f"#### {translate_text('Etiology, Host Metadata & Morphological Indicators', lang_code)}")

    with st.container(border=True):
        definition_row(
            translate_text("Host Plant", lang_code),
            translate_text(disp_plant, lang_code),
            key="symp-crop",
        )
        if disp_scientific:
            definition_row(
                translate_text("Scientific Binomial", lang_code),
                f"*{disp_scientific}*",
                key="symp-sci",
            )
        if disp_hindi:
            definition_row(
                translate_text("Hindi Name (हिन्दी नाम)", lang_code),
                disp_hindi,
                key="symp-hindi",
            )
        definition_row(
            translate_text("Diagnosed Condition", lang_code),
            translate_text(disp_condition, lang_code),
            key="symp-cond",
        )
        definition_row(
            translate_text("Condition Category", lang_code),
            translate_text(disp_type, lang_code),
            key="symp-type",
        )
        definition_row(
            translate_text("Causal Agent / Pathogen", lang_code),
            f"*{disp_pathogen}*",
            key="symp-path",
        )
        definition_row(
            translate_text("Project-Defined Lesion Severity", lang_code),
            f"{translate_text(result['severity'], lang_code)} ({translate_text(result.get('severity_scheme_label', SEVERITY_SCHEME_LABEL), lang_code)})",
            key="symp-sev",
        )
        definition_row(
            translate_text("Reference Urgency / Threat Level", lang_code),
            translate_text(str(disp_urgency).capitalize(), lang_code),
            key="symp-agro-sev",
        )
        definition_row(
            translate_text("Diagnostic Symptoms", lang_code),
            translate_text(disp_symptoms, lang_code),
            key="symp-desc",
        )

    if disp_prev:
        st.space("small")
        st.markdown(f"#### {translate_text('Preventative Cultural Controls', lang_code)}")
        with st.container(border=True):
            for p in disp_prev:
                st.markdown(f"- {translate_text(p, lang_code)}")

    if disp_uses:
        st.space("small")
        st.markdown(f"#### {translate_text('Botanical & Traditional Uses of Host Plant', lang_code)}")
        with st.container(border=True):
            for u in disp_uses:
                st.markdown(f"- {translate_text(u, lang_code)}")

# TAB 3: PROBABILITY DISTRIBUTIONS
with tab_cand:
    st.space("small")
    candidates = result.get("top_candidates", [])
    if candidates:
        st.markdown(f"#### {translate_text('38-Class Crop Pathology Probability Distribution (ViT)', lang_code)}")
        st.caption(translate_text("Top consistency-gated crop disease candidates evaluated by the ViT model:", lang_code))
        with st.container(border=True):
            for c in candidates:
                c_score = c.get("confidence_pct", c.get("confidence", 0) * 100)
                col_name, col_bar = st.columns([1.2, 0.8], vertical_alignment="center")
                with col_name:
                    st.markdown(f"**{translate_text(c['label'], lang_code)}**")
                with col_bar:
                    st.progress(min(max(float(c_score) / 100.0, 0.0), 1.0), text=f"{c_score:.1f}%")
        st.space("small")

    top_conds = gen_cond.get("top_conditions", [])
    if top_conds:
        st.markdown(f"#### {translate_text('7-Class General Foliar Condition Distribution (ConvNeXt-Tiny)', lang_code)}")
        st.caption(
            translate_text(
                "General foliar condition probabilities evaluated on the leaf specimen across all 7 pathological categories:",
                lang_code,
            )
        )
        with st.container(border=True):
            for tc in top_conds:
                tc_score = float(tc.get("confidence_pct", tc.get("score", 0.0) * 100))
                col_name, col_bar = st.columns([1.2, 0.8], vertical_alignment="center")
                with col_name:
                    st.markdown(
                        f"**{translate_text(tc.get('name', tc.get('id', '')), lang_code)}** "
                        f"<span style='opacity:0.7;font-size:0.85em;'>({translate_text(tc.get('type', ''), lang_code)})</span>",
                        unsafe_allow_html=True,
                    )
                with col_bar:
                    st.progress(min(max(tc_score / 100.0, 0.0), 1.0), text=f"{tc_score:.1f}%")

    top_sp_cands = gen_cond.get("top_species_candidates", [])
    if top_sp_cands:
        st.space("small")
        st.markdown(f"#### {translate_text('Host Plant Species Identification Candidates', lang_code)}")
        with st.container(border=True):
            for sp in top_sp_cands[:5]:
                sp_score = float(sp.get("confidence_pct", sp.get("score", 0.0) * 100))
                col_name, col_bar = st.columns([1.2, 0.8], vertical_alignment="center")
                with col_name:
                    st.markdown(f"**{translate_text(sp.get('species', ''), lang_code)}** (*{sp.get('scientific_name', '')}*)")
                with col_bar:
                    st.progress(min(max(sp_score / 100.0, 0.0), 1.0), text=f"{sp_score:.1f}%")

# TAB 4: SAVE AS FIELD SIGHTING
with tab_sight:
    st.space("small")
    st.markdown(f"#### {translate_text('Associate Diagnosis with Geographic Sighting', lang_code)}")
    st.markdown(
        translate_text(
            "Optionally anchor this specimen diagnosis to GPS coordinates to map disease outbreak patterns "
            "and record it in your persistent field herbarium.",
            lang_code,
        )
    )
    st.space("small")

    with st.container(border=True):
        loc_method = st.radio(
            translate_text("Coordinate acquisition method", lang_code),
            [
                translate_text("Auto-detect my location via IP", lang_code),
                translate_text("Enter coordinates manually", lang_code),
            ],
            horizontal=True,
            key="sighting_loc_method",
        )

        s_lat = st.session_state.get("ar_lat") or 28.6139
        s_lon = st.session_state.get("ar_lon") or 77.2090

        if "Auto-detect" in loc_method:
            if st.button(translate_text("Acquire GPS Coordinates", lang_code), icon=":material/my_location:"):
                with st.spinner(translate_text("Resolving geolocation...", lang_code)):
                    loc = finder.get_user_location()
                    if loc:
                        s_lat, s_lon = loc
                        st.session_state.ar_lat = s_lat
                        st.session_state.ar_lon = s_lon
                        st.success(f"Coordinates locked: {s_lat:.4f}, {s_lon:.4f}")
                    else:
                        st.warning(translate_text("Could not auto-resolve location. Please input coordinates manually.", lang_code))
        else:
            col_slat, col_slon = st.columns(2)
            with col_slat:
                s_lat = st.number_input(translate_text("Latitude", lang_code), value=float(s_lat), format="%.6f")
            with col_slon:
                s_lon = st.number_input(translate_text("Longitude", lang_code), value=float(s_lon), format="%.6f")

        location_tag = st.text_input(
            translate_text("Location name / Field zone (optional)", lang_code),
            placeholder="e.g., Farm Sector 4, Green Terrace Bed 2, Indore, MP",
            key="sighting_loc_tag",
        )

        observation_notes = st.text_area(
            translate_text("Field Observation Notes (optional)", lang_code),
            placeholder="e.g., Observed early leaf spotting or yellowing on lower canopy.",
            key="sighting_notes",
        )

        st.space("small")
        if st.button(
            translate_text("Save Diagnosis as Sighting", lang_code),
            type="primary",
            icon=":material/bookmark_add:",
        ):
            upsert_disease_diagnosis({
                "plant_name": disp_plant,
                "disease_name": disp_full_title,
                "health_status": disp_health_status,
                "confidence": disp_conf,
                "confidence_pct": disp_conf_pct,
                "pathogen": disp_pathogen,
                "severity": result["severity"],
                "symptoms": disp_symptoms,
                "recommendations": disp_recs,
                "image_bytes": result["image_bytes"],
                "is_sighting": True,
                "latitude": s_lat,
                "longitude": s_lon,
                "location_name": location_tag or f"Location ({s_lat:.4f}, {s_lon:.4f})",
                "notes": observation_notes,
            })

            upsert_library_entry({
                "common_name": f"{disp_plant} ({disp_full_title})",
                "scientific_name": disp_scientific or disp_plant,
                "score": disp_conf,
                "taxonomy": {},
                "medicinal": None,
                "description": f"Field specimen diagnosed with {disp_full_title} ({disp_health_status}) with {disp_conf_pct}% confidence.",
                "image_bytes": result["image_bytes"],
                "latitude": s_lat,
                "longitude": s_lon,
                "location_name": location_tag,
            })

            st.success(
                translate_text(
                    f"Diagnosis successfully cataloged as verified sighting at ({s_lat:.4f}, {s_lon:.4f})!",
                    lang_code,
                )
            )

st.space("medium")

# ==============================================================================
# USER FEEDBACK WIDGET (Task 11)
# ==============================================================================
render_feedback_widget(
    task_type="disease_detection",
    predicted_label=f"{disp_plant} — {disp_condition} ({disp_raw_label})",
    confidence=float(disp_conf),
    model_name=disp_model_name,
    extra_meta={
        "is_supported_crop": not is_unsupported_crop,
        "confidence_tier": disp_conf_tier,
        "affected_area_pct": result.get("affected_area_pct"),
        "project_severity": result.get("severity"),
        "derived_health_score": result.get("derived_health_score"),
    },
    key_prefix=f"fb_disease_{disp_raw_label}",
)

st.space("medium")

# ==============================================================================
# EXPORT REPORT & LICENSE ATTRIBUTION FOOTER
# ==============================================================================
report_data = {
    "report_type": "PlantVerse Plant Disease & Foliar Diagnostic Report",
    "timestamp": st.session_state.get("disease_timestamp", ""),
    "crop_consistency_gate": {
        "is_supported_14_crop": not is_unsupported_crop,
        "gate_status": "Passed (14-crop ViT active)" if not is_unsupported_crop else UNSUPPORTED_CROP_DISEASE_MESSAGE,
    },
    "model_output": {
        "plant": disp_plant,
        "scientific_name": disp_scientific,
        "hindi_name": disp_hindi,
        "condition": disp_condition,
        "full_diagnosis": disp_full_title,
        "raw_label": disp_raw_label,
        "health_status": disp_health_status,
        "confidence_percentage": disp_conf_pct,
        "confidence_tier": disp_conf_tier,
        "model_name": disp_model_name,
        "top_crop_candidates": result.get("top_candidates", []),
        "top_general_conditions": gen_cond.get("top_conditions", []),
    },
    "classical_cv_and_derived_metrics": {
        "method": vision_info.get("method_label", CLASSICAL_CV_METHOD_LABEL),
        "affected_area_pct": result.get("affected_area_pct"),
        "raw_unfiltered_severity_pct": vision_info.get("raw_unfiltered_severity_pct"),
        "estimation_status": result.get("affected_area_status"),
        "project_severity_category": result.get("severity"),
        "severity_scheme_label": result.get("severity_scheme_label", SEVERITY_SCHEME_LABEL),
        "derived_health_score": result.get("derived_health_score"),
        "health_score_label": result.get("health_score_label", DERIVED_HEALTH_SCORE_LABEL),
    },
    "knowledge_base": {
        "pathogen": disp_pathogen,
        "symptoms": disp_symptoms,
        "recommendations": disp_recs,
        "chemical_treatments": disp_chem,
        "organic_treatments": disp_org,
        "preventive_measures": disp_prev,
        "common_species_problems": disp_common_problems,
        "care": disp_care,
        "expert_review_notice": result.get("expert_review_notice", EXPERT_REVIEW_NOTICE),
    },
}

st.download_button(
    label=translate_text("Export Pathology Diagnostic Report (JSON)", lang_code),
    data=json.dumps(report_data, indent=2, ensure_ascii=False),
    file_name=f"plantverse_diagnosis_{disp_plant.lower().replace(' ', '_')}_{disp_condition.lower().replace(' ', '_')}.json",
    mime="application/json",
    icon=":material/download:",
)

st.caption(
    ":material/copyright: **Model Attribution & License Notice:** Primary 38-class ViT crop disease model (`models/model.onnx`) is adapted from "
    "[wambugu71/crop_leaf_diseases_vit](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) (**[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)**); "
    "Dual-head 68-species + 7-condition ConvNeXt-Tiny (`models/plant_model.onnx`) is licensed under `Apache-2.0` / `CC BY-NC-SA 4.0`. See `MODEL_LICENSES.md` for full details."
)
