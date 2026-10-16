from io import BytesIO

import streamlit as st
from PIL import Image

from ar_location_plant_map import PlantLocationFinder
from plant_identification import (
    LANGUAGES,
    get_medicinal_uses_from_wikipedia,
    get_taxonomy_from_wikidata,
    get_wikipedia_title,
    predict_species,
    translate_text,
)
from plant_disease_detection import predict_disease
from ui.chrome import definition_row, init_state, kicker, section_rule
from ui.presentation import (
    remaining_taxonomy,
    render_taxonomy,
    upsert_disease_diagnosis,
    upsert_library_entry,
)

init_state()

# ==============================================================================
# HEADER & LANGUAGE SELECTOR
# ==============================================================================
top_left, top_right = st.columns([1.35, 0.65], vertical_alignment="top")

with top_left:
    st.markdown(
        """<div class="pv-academic-badge">
            <span>📷</span>
            <span>Computer Vision Inference Module • Specimen Studio</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.header("Plant Identification Studio")
    st.markdown(
        "Upload a high-resolution photograph of a leaf, bloom, or whole botanical specimen. "
        "The system processes the visual morphology using deep neural classification, resolves "
        "the biological identity through Wikidata, and compiles pharmacological and ecological data."
    )

with top_right:
    with st.container(border=True):
        selected_lang = st.selectbox(
            "Evaluation Language",
            list(LANGUAGES.keys()),
            key="identify_language",
            help="Select evaluation language for translation of plant names, taxonomy, and botanical records.",
        )
        lang_code = LANGUAGES[selected_lang]
        st.caption(f":material/translate: Active Language: **{selected_lang}**")

st.space("small")
section_rule("section-rule-input")
st.space("small")

# ==============================================================================
# UPLOAD STUDIO & CAMERA PROTOCOL
# ==============================================================================
upload_col, guide_col = st.columns([1.08, 0.92], gap="large")

with upload_col:
    with st.container(border=True):
        st.markdown(f"#### {translate_text('Specimen Photo Acquisition', lang_code)}")
        st.caption(
            translate_text(
                "Upload a specimen photograph in JPG, JPEG, or PNG format.",
                lang_code,
            )
        )
        with st.container(key="identify-upload"):
            img_file = st.file_uploader(
                translate_text("Plant photograph", lang_code),
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="identify_uploader",
            )

        if img_file:
            try:
                preview = Image.open(img_file)
                st.image(
                    preview,
                    caption=f"{img_file.name} • Dimensions: {preview.width} × {preview.height} px",
                    width="stretch",
                    alt=translate_text("Uploaded plant photograph", lang_code),
                )
            except Exception as exc:
                st.error(f"Unable to read image file: {exc}")
                st.stop()

        st.space("small")
        enable_disease_check = st.checkbox(
            translate_text("Screen for Foliar Diseases & Health Status", lang_code),
            value=True,
            help="Simultaneously evaluate leaf health and classify crop disease pathologies using the PlantVillage neural model.",
        )
        st.space("small")
        run_identify = st.button(
            translate_text("Execute Identification Pipeline", lang_code),
            type="primary",
            icon=":material/auto_awesome:",
            disabled=img_file is None,
        )

with guide_col:
    with st.container(border=True):
        st.markdown(f"#### :material/science: {translate_text('Imaging Protocol & Best Practices', lang_code)}")
        st.markdown(
            translate_text(
                "Classification fidelity depends on diagnostic morphological clarity. "
                "Adhere to the following photographic standards for optimal inference results:",
                lang_code,
            )
        )
        st.space("small")

        with st.container(border=True):
            st.markdown(f"**:material/wb_sunny: {translate_text('Diffuse Natural Illumination', lang_code)}**")
            st.markdown(
                translate_text(
                    "Capture under bright, ambient outdoor lighting. Avoid direct lens glare or harsh shadow occlusions.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/center_focus_strong: {translate_text('Diagnostic Character Macro', lang_code)}**")
            st.markdown(
                translate_text(
                    "Focus sharply on distinguishing anatomical features such as leaf margins, primary venation, or floral whorls.",
                    lang_code,
                )
            )

        with st.container(border=True):
            st.markdown(f"**:material/filter_vintage: {translate_text('Background Isolation', lang_code)}**")
            st.markdown(
                translate_text(
                    "Frame against an unobstructed natural or neutral backdrop to eliminate noise from surrounding foliage.",
                    lang_code,
                )
            )

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
if img_file:
    file_id = f"{img_file.name}-{img_file.size}"
    if st.session_state.get("ident_file_id") != file_id:
        st.session_state.ident_result = None
        st.session_state.ident_file_id = file_id

if img_file and run_identify:
    img_file.seek(0)
    try:
        image = Image.open(img_file)
    except Exception as exc:
        st.error(f"Unable to open image: {exc}")
        st.stop()

    with st.status(translate_text("Processing specimen through identification pipeline...", lang_code), expanded=True) as status:
        st.write(translate_text("1. Computing morphological feature representations with FloraSense...", lang_code))
        try:
            label, score = predict_species(image)
        except Exception as exc:
            status.update(label="Inference execution halted", state="error")
            st.error(f"Plant identification failed: {exc}")
            st.stop()

        st.write(translate_text("2. Resolving taxonomic entity against Wikipedia & Wikidata...", lang_code))
        wiki_title = get_wikipedia_title(label)
        taxonomy = get_taxonomy_from_wikidata(wiki_title)

        st.write(translate_text("3. Extracting pharmacological data & botanical encyclopedia summary...", lang_code))
        medicinal = get_medicinal_uses_from_wikipedia(wiki_title, target_lang="en")
        description = PlantLocationFinder().get_wikipedia_summary(wiki_title)

        disease_res = None
        if enable_disease_check:
            st.write(translate_text("4. Screening foliar health and agricultural disease pathologies...", lang_code))
            try:
                disease_res = predict_disease(image)
                st.session_state.disease_result = disease_res
                upsert_disease_diagnosis({
                    "plant_name": disease_res["plant"],
                    "disease_name": disease_res["disease_name"],
                    "health_status": disease_res["health_status"],
                    "confidence": disease_res["confidence"],
                    "confidence_pct": disease_res["confidence_pct"],
                    "pathogen": disease_res["pathogen"],
                    "severity": disease_res["severity"],
                    "symptoms": disease_res["symptoms"],
                    "recommendations": disease_res["recommendations"],
                    "image_bytes": image_bytes,
                    "is_sighting": False,
                })
            except Exception as d_exc:
                st.warning(f"Disease screening skipped: {d_exc}")

        buffer = BytesIO()
        rgb = image.convert("RGB")
        rgb.save(buffer, format="JPEG", quality=88)
        image_bytes = buffer.getvalue()

        result = {
            "common_name": label,
            "scientific_name": wiki_title,
            "score": float(score),
            "taxonomy": taxonomy,
            "medicinal": medicinal,
            "description": description,
            "image_bytes": image_bytes,
            "disease_diagnosis": disease_res,
        }
        st.session_state.ident_result = result
        upsert_library_entry(result)
        status.update(label=translate_text("Pipeline execution completed successfully!", lang_code), state="complete")

result = st.session_state.get("ident_result")
if not result:
    st.stop()

st.space("large")
section_rule("section-rule-profile")
st.space("small")

# ==============================================================================
# SPECIMEN DOSSIER HERO CARD
# ==============================================================================
kicker(translate_text("Identification Analysis Result", lang_code), key="kicker-result")

with st.container(border=True):
    image_col, identity_col = st.columns([0.92, 1.08], gap="large", vertical_alignment="center")

    with image_col:
        with st.container(key="profile-image"):
            st.image(
                result["image_bytes"],
                alt=translate_text("Identified plant photograph", lang_code),
                width="stretch",
            )
        st.caption(f":material/verified: {translate_text('Analyzed specimen photograph voucher', lang_code)}")

    with identity_col:
        score_pct = result["score"] * 100
        if score_pct >= 85:
            conf_badge = f":green-badge[Certainty: {score_pct:.1f}% — High Confidence]"
        elif score_pct >= 60:
            conf_badge = f":blue-badge[Certainty: {score_pct:.1f}% — Moderate Confidence]"
        else:
            conf_badge = f":orange-badge[Certainty: {score_pct:.1f}% — Tentative Candidate]"

        st.markdown(conf_badge)

        dis_diag = result.get("disease_diagnosis")
        if dis_diag:
            if dis_diag["health_status"] == "Healthy":
                st.markdown(f":green-badge[✓ {translate_text('Foliar Health: Healthy', lang_code)} ({dis_diag['confidence_pct']}%) ]")
            else:
                st.markdown(f":red-badge[⚠ {translate_text('Pathology Detected', lang_code)}: {translate_text(dis_diag['disease_name'], lang_code)} ({dis_diag['confidence_pct']}%) ]")

        st.markdown(f"## {translate_text(result['common_name'], lang_code)}")
        st.markdown(f"### *{result['scientific_name']}*")

        st.space("small")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric(
                label=translate_text("Species Confidence", lang_code),
                value=f"{score_pct:.1f}%",
                help="Softmax probability output by the FloraSense classification model.",
            )
        with stat_col2:
            if dis_diag:
                st.metric(
                    label=translate_text("Health Status", lang_code),
                    value=translate_text(dis_diag["health_status"], lang_code),
                    help="Agricultural leaf disease classification result.",
                )
            else:
                st.metric(
                    label=translate_text("Model Architecture", lang_code),
                    value="FloraSense",
                    help="Deep convolutional image classification pipeline hosted on Hugging Face.",
                )

        st.space("small")
        action_row = st.container(horizontal=True, gap="small")
        with action_row:
            if st.button(
                translate_text("Locate Nearby on Map", lang_code),
                icon=":material/map:",
            ):
                st.session_state.ar_query = result["scientific_name"]
                st.switch_page("app_pages/ar_explorer.py")
            if dis_diag and st.button(
                translate_text("Disease Diagnostic Studio", lang_code),
                icon=":material/biotech:",
            ):
                st.session_state.disease_result = dis_diag
                st.switch_page("app_pages/disease_detection.py")
            if st.button(
                translate_text("Inspect in Library", lang_code),
                icon=":material/menu_book:",
            ):
                st.switch_page("app_pages/library.py")

st.space("medium")

# ==============================================================================
# MULTIMODAL RESEARCH DOSSIER TABS
# ==============================================================================
tab_tax, tab_desc, tab_med, tab_meta, tab_path = st.tabs(
    [
        translate_text("🌿 Taxonomic Hierarchy", lang_code),
        translate_text("📖 Botanical Encyclopedia", lang_code),
        translate_text("💊 Pharmacological Literature", lang_code),
        translate_text("ℹ️ Authority Metadata", lang_code),
        translate_text("🔬 Foliar Pathology & Health", lang_code),
    ]
)

# TAB 1: TAXONOMY
with tab_tax:
    st.space("small")
    st.markdown(f"#### {translate_text('Taxonomic Lineage & Wikidata Ranks', lang_code)}")
    st.markdown(
        translate_text(
            "Complete biological lineage reconstructed dynamically by querying Wikidata entity claims "
            "for taxon name (P225), parent taxon (P171), and rank (P105).",
            lang_code,
        )
    )
    st.space("small")
    render_taxonomy(
        result.get("taxonomy"),
        translate=translate_text,
        lang_code=lang_code,
        featured_only=False,
        key_prefix="class",
    )

# TAB 2: BOTANICAL DESCRIPTION
with tab_desc:
    st.space("small")
    st.markdown(f"#### {translate_text('Morphological & Botanical Abstract', lang_code)}")
    description = result.get("description") or ""
    if description:
        with st.container(border=True):
            st.markdown(translate_text(description, lang_code))
            st.caption(":material/menu_book: Reference Source: Wikipedia Summary REST API")
    else:
        st.info(translate_text("No encyclopedic description is currently indexed for this taxon.", lang_code))

# TAB 3: MEDICINAL INFORMATION
with tab_med:
    st.space("small")
    st.markdown(f"#### {translate_text('Ethnobotanical & Pharmacological Notes', lang_code)}")
    medicinal = result.get("medicinal")
    if medicinal:
        with st.container(border=True):
            for point in medicinal:
                st.markdown(f"- {translate_text(point, lang_code)}")
        st.space("small")
        st.caption(
            ":material/warning: **Academic Advisory**: Pharmacological records are compiled from Wikipedia "
            "for scholarly review and historical reference. They do not constitute clinical guidance or dosage instructions."
        )
    else:
        st.info(translate_text("No verified medicinal or pharmacological sections indexed for this species.", lang_code))

# TAB 4: METADATA & CITATIONS
with tab_meta:
    st.space("small")
    st.markdown(f"#### {translate_text('Authority References & Nomenclature Keys', lang_code)}")
    with st.container(border=True):
        definition_row(
            translate_text("Common Name", lang_code),
            translate_text(result["common_name"], lang_code),
            key="meta-common",
        )
        definition_row(
            translate_text("Scientific Binomial", lang_code),
            f"*{result['scientific_name']}*",
            key="meta-sci",
        )
        definition_row(
            translate_text("Wikipedia Page", lang_code),
            f"[{result['scientific_name']}](https://en.wikipedia.org/wiki/{result['scientific_name'].replace(' ', '_')})",
            key="meta-wiki",
        )
        extra_ranks = remaining_taxonomy(result.get("taxonomy"))
        if extra_ranks:
            for rank, value in extra_ranks:
                definition_row(
                    translate_text(rank, lang_code),
                    f"*{value}*",
                    key=f"meta-{rank}",
                )

# TAB 5: FOLIAR PATHOLOGY & HEALTH
with tab_path:
    st.space("small")
    st.markdown(f"#### {translate_text('Foliar Pathology & Disease Diagnosis', lang_code)}")
    d_info = result.get("disease_diagnosis")
    if d_info:
        with st.container(border=True):
            definition_row(
                translate_text("Health Status", lang_code),
                f"**{translate_text(d_info['health_status'], lang_code)}**",
                key="id-path-health",
            )
            definition_row(
                translate_text("Condition", lang_code),
                f"**{translate_text(d_info['disease_name'], lang_code)}**",
                key="id-path-cond",
            )
            definition_row(
                translate_text("Diagnostic Confidence", lang_code),
                f"**{d_info['confidence_pct']}%**",
                key="id-path-conf",
            )
            definition_row(
                translate_text("Causal Agent", lang_code),
                f"*{d_info['pathogen']}*",
                key="id-path-agent",
            )
            definition_row(
                translate_text("Symptoms", lang_code),
                translate_text(d_info["symptoms"], lang_code),
                key="id-path-symp",
            )

        st.space("small")
        st.markdown(f"#### {translate_text('Targeted Actionable Recommendations', lang_code)}")
        recs = d_info.get("recommendations", [])
        if recs:
            with st.container(border=True):
                for idx, r in enumerate(recs, 1):
                    st.markdown(f"**{idx}.** {translate_text(r, lang_code)}")
        else:
            st.info(translate_text("No corrective interventions required for this healthy specimen.", lang_code))

        st.space("small")
        if st.button(
            translate_text("Open Full Pathology Studio", lang_code),
            icon=":material/biotech:",
            type="primary",
            key="btn_open_disease_studio",
        ):
            st.session_state.disease_result = d_info
            st.switch_page("app_pages/disease_detection.py")
    else:
        st.info(translate_text("Foliar pathology screening was not enabled for this analysis run.", lang_code))
