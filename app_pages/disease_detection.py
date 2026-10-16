import json
import os
from io import BytesIO
from typing import Optional

import streamlit as st
from PIL import Image

from ar_location_plant_map import PlantLocationFinder
from disease_recommendations import get_disease_info
from plant_disease_detection import predict_disease
from plant_identification import LANGUAGES, translate_text
from ui.chrome import definition_row, init_state, kicker, section_rule
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
            <span>Plant Pathology Engine • Leaf Disease Diagnostic Studio</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.header("Plant Disease Detection Studio")
    st.markdown(
        "Screen crops and botanical specimens for foliar infections, blights, rusts, and viruses. "
        "The neural pathology inference pipeline diagnoses conditions across 38 distinct agricultural classes, "
        "evaluates prediction confidence, and supplies validated cultural and treatment recommendations."
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
st.caption(translate_text("Quick Evaluation Benchmark Specimens:", lang_code))
sample_cols = st.container(horizontal=True, gap="small")
sample_choice: Optional[str] = None
with sample_cols:
    if st.button("🍅 " + translate_text("Tomato Early Blight", lang_code), key="sample_eb"):
        sample_choice = "static/sample_leaves/tomato_early_blight.jpg"
    if st.button("🍏 " + translate_text("Apple Scab", lang_code), key="sample_scab"):
        sample_choice = "static/sample_leaves/apple_scab.jpg"
    if st.button("🌿 " + translate_text("Healthy Tomato", lang_code), key="sample_htomato"):
        sample_choice = "static/sample_leaves/healthy_tomato.jpg"
    if st.button("🍎 " + translate_text("Healthy Apple", lang_code), key="sample_happle"):
        sample_choice = "static/sample_leaves/healthy_apple.jpg"

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
                "Upload a high-resolution photograph of the affected leaf or foliage in JPG, JPEG, or PNG format.",
                lang_code,
            )
        )

        with st.container(key="disease-upload"):
            img_file = st.file_uploader(
                translate_text("Leaf photograph", lang_code),
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="disease_uploader",
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
                preview_caption = f"{img_file.name} • Dimensions: {preview_image.width} × {preview_image.height} px"
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
                "Accurate identification of fungal, bacterial, and viral pathologies requires clear visualization "
                "of diagnostic lesions, halos, and margins. Observe the following photographic standards:",
                lang_code,
            )
        )
        st.space("small")

        with st.container(border=True):
            st.markdown(f"**:material/center_focus_strong: {translate_text('Macro Lesion Framing', lang_code)}**")
            st.markdown(
                translate_text(
                    "Capture clear close-ups of distinct lesions, chlorotic rings, rust pustules, or powdery fungal growth.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/flip_to_back: {translate_text('Inspect Upper & Lower Lamina', lang_code)}**")
            st.markdown(
                translate_text(
                    "Many pathogens (e.g. Downy Mildew, Spider Mites) produce signature signs primarily on leaf undersides.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/wb_sunny: {translate_text('Uniform Glare-Free Lighting', lang_code)}**")
            st.markdown(
                translate_text(
                    "Diffuse daylight preserves natural coloration and prevents harsh specular reflections on waxy leaves.",
                    lang_code,
                )
            )

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
current_input_id = f"{img_file.name}-{img_file.size}" if img_file else (active_sample or "")
if st.session_state.get("disease_file_id") != current_input_id:
    st.session_state.disease_result = None
    st.session_state.disease_file_id = current_input_id

if has_input and run_diagnose:
    input_source = img_file if img_file else active_sample
    if hasattr(input_source, "seek"):
        input_source.seek(0)

    with st.status(translate_text("Processing specimen through neural pathology pipeline...", lang_code), expanded=True) as status:
        st.write(translate_text("1. Validating visual morphology and standardizing tensor resolution...", lang_code))

        try:
            st.write(translate_text("2. Computing pathology probability distribution with MobileNetV2 / PlantVillage...", lang_code))
            prediction_result = predict_disease(input_source)
        except Exception as exc:
            status.update(label="Inference execution halted", state="error")
            st.error(f"Diagnosis failed: {exc}")
            st.stop()

        st.write(translate_text("3. Synthesizing agronomic recommendations and biological pathogen profile...", lang_code))
        st.session_state.disease_result = prediction_result

        # Auto-record diagnosis in session state and persistent storage
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

        status.update(label=translate_text("Pathology diagnosis completed successfully!", lang_code), state="complete")

result = st.session_state.get("disease_result")
if not result:
    st.stop()

st.space("large")
section_rule("section-rule-disease-result")
st.space("small")

# ==============================================================================
# PATHOLOGY DOSSIER HERO CARD
# ==============================================================================
kicker(translate_text("Pathology Diagnostic Report", lang_code), key="kicker-disease-result")

with st.container(border=True):
    image_col, diagnosis_col = st.columns([0.92, 1.08], gap="large", vertical_alignment="center")

    with image_col:
        with st.container(key="profile-image"):
            st.image(
                result["image_bytes"],
                alt=translate_text("Diagnosed leaf specimen photograph", lang_code),
                width="stretch",
            )
        st.caption(f":material/verified: {translate_text('Evaluated foliar specimen photograph', lang_code)}")

    with diagnosis_col:
        conf_pct = result["confidence_pct"]
        is_healthy = result["health_status"] == "Healthy"

        if is_healthy:
            status_badge = f":green-badge[✓ {translate_text('Healthy Foliage — No Active Pathology', lang_code)}]"
        elif conf_pct >= 75:
            status_badge = f":red-badge[⚠ {translate_text('Pathology Detected — High Confidence', lang_code)}]"
        elif conf_pct >= 50:
            status_badge = f":orange-badge[⚠ {translate_text('Pathology Detected — Moderate Confidence', lang_code)}]"
        else:
            status_badge = f":orange-badge[? {translate_text('Uncertain Condition — Tentative Candidate', lang_code)}]"

        st.markdown(status_badge)
        st.markdown(f"## {translate_text(result['plant'], lang_code)} — {translate_text(result['disease'], lang_code)}")
        st.markdown(f"### *{result['pathogen']}*")

        if result.get("is_uncertain"):
            st.warning(f":material/warning: {translate_text(result['uncertainty_message'], lang_code)}")

        st.space("small")
        stat_c1, stat_c2 = st.columns(2)
        with stat_c1:
            st.metric(
                label=translate_text("Prediction Confidence", lang_code),
                value=f"{conf_pct:.1f}%",
                help="Softmax probability distribution output by the PlantVillage neural classifier.",
            )
        with stat_c2:
            st.metric(
                label=translate_text("Health Status", lang_code),
                value=translate_text(result["health_status"], lang_code),
                help="Overall physiological assessment of foliar health.",
            )

        st.space("small")
        action_row = st.container(horizontal=True, gap="small")
        with action_row:
            if st.button(
                translate_text("Locate Host Species on Map", lang_code),
                icon=":material/map:",
            ):
                st.session_state.ar_query = result["plant"]
                st.switch_page("app_pages/ar_explorer.py")
            if st.button(
                translate_text("Inspect Specimen in Library", lang_code),
                icon=":material/menu_book:",
            ):
                st.session_state.library_profile = None
                st.switch_page("app_pages/library.py")

st.space("medium")

# ==============================================================================
# MULTIMODAL PATHOLOGY DETAILS TABS
# ==============================================================================
tab_recs, tab_symp, tab_cand, tab_sight = st.tabs(
    [
        translate_text("💊 Actionable Recommendations", lang_code),
        translate_text("🔬 Symptoms & Pathogen Profile", lang_code),
        translate_text("📊 Alternative Candidates", lang_code),
        translate_text("📍 Save as Field Sighting", lang_code),
    ]
)

# TAB 1: RECOMMENDATIONS
with tab_recs:
    st.space("small")
    st.markdown(f"#### {translate_text('Targeted Agronomic Interventions', lang_code)}")
    st.caption(
        translate_text(
            "Evidence-based cultural, mechanical, and treatment practices to mitigate infection spread.",
            lang_code,
        )
    )

    recs = result.get("recommendations", [])
    if recs:
        with st.container(border=True):
            for idx, item in enumerate(recs, 1):
                st.markdown(f"**{idx}.** {translate_text(item, lang_code)}")
    else:
        st.info(translate_text("No specific disease remediation required for this healthy specimen.", lang_code))

    st.space("small")
    st.caption(
        ":material/health_and_safety: **" + translate_text("Agronomic Advisory", lang_code) + "**: "
        + translate_text(
            "Recommendations are provided strictly for educational and informational purposes. "
            "Always consult certified agronomists and adhere to manufacturer chemical label requirements before applying chemical fungicides.",
            lang_code,
        )
    )

# TAB 2: SYMPTOMS & PATHOGEN PROFILE
with tab_symp:
    st.space("small")
    st.markdown(f"#### {translate_text('Etiology & Morphological Indicators', lang_code)}")

    with st.container(border=True):
        definition_row(
            translate_text("Host Crop", lang_code),
            translate_text(result["plant"], lang_code),
            key="symp-crop",
        )
        definition_row(
            translate_text("Pathological Condition", lang_code),
            translate_text(result["disease"], lang_code),
            key="symp-cond",
        )
        definition_row(
            translate_text("Pathogen Agent", lang_code),
            f"*{result['pathogen']}*",
            key="symp-path",
        )
        definition_row(
            translate_text("Pathological Severity", lang_code),
            translate_text(result["severity"], lang_code),
            key="symp-sev",
        )
        definition_row(
            translate_text("Diagnostic Symptoms", lang_code),
            translate_text(result["symptoms"], lang_code),
            key="symp-desc",
        )

    st.space("small")
    st.markdown(f"#### {translate_text('Preventative Cultural Controls', lang_code)}")
    prevs = result.get("preventive_measures", [])
    if prevs:
        with st.container(border=True):
            for p in prevs:
                st.markdown(f"- {translate_text(p, lang_code)}")

# TAB 3: ALTERNATIVE CANDIDATES
with tab_cand:
    st.space("small")
    st.markdown(f"#### {translate_text('Neural Classification Probability Distribution', lang_code)}")
    st.caption(translate_text("Top candidate predictions evaluated by the model softmax layer:", lang_code))

    candidates = result.get("top_candidates", [])
    if candidates:
        with st.container(border=True):
            for c in candidates:
                c_score = c.get("confidence_pct", c.get("confidence", 0) * 100)
                col_name, col_bar = st.columns([1.2, 0.8], vertical_alignment="center")
                with col_name:
                    st.markdown(f"**{translate_text(c['label'], lang_code)}**")
                with col_bar:
                    st.progress(min(max(float(c_score) / 100.0, 0.0), 1.0), text=f"{c_score:.1f}%")
    else:
        st.info(translate_text("No alternative candidate distributions available.", lang_code))

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
            placeholder="e.g., Observed early leaf spotting on lower canopy after heavy rain.",
            key="sighting_notes",
        )

        st.space("small")
        if st.button(
            translate_text("Save Diagnosis as Sighting", lang_code),
            type="primary",
            icon=":material/bookmark_add:",
        ):
            # Save into persistent disease diagnoses store with sighting metadata
            upsert_disease_diagnosis({
                "plant_name": result["plant"],
                "disease_name": result["disease_name"],
                "health_status": result["health_status"],
                "confidence": result["confidence"],
                "confidence_pct": result["confidence_pct"],
                "pathogen": result["pathogen"],
                "severity": result["severity"],
                "symptoms": result["symptoms"],
                "recommendations": result["recommendations"],
                "image_bytes": result["image_bytes"],
                "is_sighting": True,
                "latitude": s_lat,
                "longitude": s_lon,
                "location_name": location_tag or f"Location ({s_lat:.4f}, {s_lon:.4f})",
                "notes": observation_notes,
            })

            # Also add to general library entries so it is mapped and indexed
            upsert_library_entry({
                "common_name": f"{result['plant']} ({result['disease_name']})",
                "scientific_name": result["plant"],
                "score": result["confidence"],
                "taxonomy": {},
                "medicinal": None,
                "description": f"Field specimen diagnosed with {result['disease_name']} ({result['health_status']}) with {result['confidence_pct']}% confidence.",
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

st.space("large")

# ==============================================================================
# EXPORT REPORT
# ==============================================================================
report_data = {
    "report_type": "PlantVerse Plant Disease Diagnostic Report",
    "timestamp": st.session_state.get("disease_timestamp", ""),
    "plant": result["plant"],
    "disease": result["disease"],
    "disease_name": result["disease_name"],
    "health_status": result["health_status"],
    "confidence_percentage": result["confidence_pct"],
    "pathogen": result["pathogen"],
    "severity": result["severity"],
    "symptoms": result["symptoms"],
    "recommendations": result["recommendations"],
    "preventive_measures": result["preventive_measures"],
    "top_candidates": result.get("top_candidates", []),
}

st.download_button(
    label=translate_text("Export Pathology Diagnostic Report (JSON)", lang_code),
    data=json.dumps(report_data, indent=2),
    file_name=f"plantverse_diagnosis_{result['plant'].lower()}_{result['disease'].lower().replace(' ', '_')}.json",
    mime="application/json",
    icon=":material/download:",
)
