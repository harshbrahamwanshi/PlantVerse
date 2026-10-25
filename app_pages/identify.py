from io import BytesIO

import streamlit as st
from PIL import Image

from ar_location_plant_map import PlantLocationFinder
from ml_config import LOW_CONFIDENCE_PLANT_MESSAGE, UNSUPPORTED_CROP_DISEASE_MESSAGE
from plant_disease_detection import predict_disease
from plant_identification import (
    LANGUAGES,
    get_medicinal_uses_from_wikipedia,
    get_supported_plants_info,
    get_taxonomy_from_wikidata,
    get_wikipedia_title,
    predict_species,
    predict_species_detailed,
    translate_text,
)
from ui.chrome import definition_row, init_state, kicker, section_rule
from ui.feedback import render_feedback_widget
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
    supported_info = get_supported_plants_info()
    with st.expander(":material/eco: Supported plants & model scope (68 species + non-plant filter)", expanded=False):
        st.markdown(
            f"**Primary Plant Classifier:** `{supported_info.get('model_name', 'ConvNeXt-Tiny ONNX')}`  \n"
            f"**Supported Species Count:** `{supported_info.get('unique_binomials', 68)}` regional & crop species (+ `not_plant` rejection)  \n"
            f"**License:** {supported_info.get('license', 'Apache-2.0 / CC BY-NC-SA 4.0')}"
        )
        sp_list = supported_info.get("supported_species_list", [])
        if sp_list:
            st.markdown("**Supported plants:** " + ", ".join(sp_list))

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
            translate_text("Screen for Foliar Diseases & Health Status (for supported crops)", lang_code),
            value=True,
            help="When the identified plant is one of the 14 supported agricultural crops, runs the 38-class ViT disease classifier gated to that crop.",
        )
        include_florasense_opt = st.checkbox(
            translate_text("Include secondary FloraSense top-5 suggestions (low confidence)", lang_code),
            value=False,
            help="Optionally query the secondary 9,985-taxon FloraSense model for additional low-confidence candidate suggestions.",
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
        st.write(translate_text("1. Computing morphological feature representations with ConvNeXt-Tiny ONNX...", lang_code))
        try:
            plant_pred = predict_species_detailed(
                image,
                include_secondary_florasense=include_florasense_opt,
            )
            label = plant_pred["species"]
            sci_from_model = plant_pred.get("scientific_name", label)
            wiki_query = plant_pred.get("wiki_query") or sci_from_model
            score = plant_pred["score"]
            conf_tier = plant_pred["confidence_tier"]
            is_reliable = plant_pred["is_reliable"]
            is_not_plant = plant_pred.get("is_not_plant", False)
        except Exception as exc:
            status.update(label="Inference execution halted", state="error")
            st.error(f"Plant identification failed: {exc}")
            st.stop()

        buffer = BytesIO()
        rgb = image.convert("RGB")
        rgb.save(buffer, format="JPEG", quality=88)
        image_bytes = buffer.getvalue()

        if is_reliable and not is_not_plant:
            st.write(translate_text("2. Resolving taxonomic entity against Wikipedia & Wikidata...", lang_code))
            wiki_title = get_wikipedia_title(wiki_query)
            taxonomy = get_taxonomy_from_wikidata(sci_from_model or wiki_title)

            st.write(translate_text("3. Extracting pharmacological data & botanical encyclopedia summary...", lang_code))
            medicinal = get_medicinal_uses_from_wikipedia(wiki_title, target_lang="en")
            description = PlantLocationFinder().get_wikipedia_summary(wiki_title)
        else:
            wiki_title = sci_from_model
            taxonomy = {}
            medicinal = None
            description = ""

        disease_res = None
        if enable_disease_check and is_reliable and not is_not_plant:
            st.write(translate_text("4. Evaluating host crop eligibility & foliar pathology...", lang_code))
            try:
                disease_res = predict_disease(image, host_plant=label)
                st.session_state.disease_result = disease_res
                if (
                    disease_res.get("is_supported_crop", False)
                    and disease_res.get("health_status") not in ("Unsupported", "Uncertain")
                    and not disease_res.get("is_uncertain", False)
                ):
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

        result = {
            "common_name": label if (is_reliable and not is_not_plant) else "Unidentified Plant",
            "scientific_name": sci_from_model if (is_reliable and not is_not_plant) else "Unverified",
            "wiki_title": wiki_title,
            "candidate_species": label,
            "score": float(score),
            "confidence_tier": conf_tier,
            "is_reliable": is_reliable,
            "is_not_plant": is_not_plant,
            "warning_message": plant_pred.get("warning_message", ""),
            "top_candidates": plant_pred.get("top_candidates", []),
            "secondary_suggestions": plant_pred.get("secondary_suggestions", []),
            "plant_metadata": plant_pred.get("plant_metadata", {}),
            "model_name": plant_pred.get("model_name", "ConvNeXt-Tiny ONNX"),
            "taxonomy": taxonomy,
            "medicinal": medicinal,
            "description": description,
            "image_bytes": image_bytes,
            "disease_diagnosis": disease_res,
        }
        st.session_state.ident_result = result
        if is_reliable and not is_not_plant:
            upsert_library_entry(result)
        status.update(label=translate_text("Pipeline execution completed!", lang_code), state="complete")

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

if not result.get("is_reliable", True) or result.get("is_not_plant", False):
    st.error(translate_text(result.get("warning_message") or LOW_CONFIDENCE_PLANT_MESSAGE, lang_code))
    with st.container(border=True):
        c1, c2 = st.columns([0.85, 1.15], gap="large")
        with c1:
            st.image(result["image_bytes"], width="stretch")
        with c2:
            if result.get("is_not_plant"):
                st.markdown(":red-badge[Non-Botanical / No Plant Detected]")
            else:
                st.markdown(f":red-badge[Low Confidence — Top-1 Score: {result['score'] * 100:.2f}%]")
            st.markdown(f"### {translate_text(LOW_CONFIDENCE_PLANT_MESSAGE, lang_code)}")
            st.markdown(
                translate_text(
                    "The model's confidence score fell below the minimum reliability threshold (or no plant was detected). "
                    "To prevent false botanical identifications, no species headline is assigned.",
                    lang_code,
                )
            )
            cands = result.get("top_candidates", [])
            if cands:
                with st.expander("View top-5 primary model candidates (low confidence — diagnostic only)"):
                    for c in cands:
                        st.markdown(f"- **{c['species']}** (*{c.get('scientific_name', '')}*) — `{c['confidence_pct']:.2f}%`")
            sec = result.get("secondary_suggestions", [])
            if sec:
                with st.expander("Other possible matches (low confidence — FloraSense secondary index)"):
                    for s in sec:
                        st.markdown(f"- *{s['species']}* — `{s['confidence_pct']:.2f}%`")
    render_feedback_widget(
        task_type="plant_identification",
        predicted_label=result.get("candidate_species", "Unidentified Plant"),
        confidence=result.get("score", 0.0),
        model_name=result.get("model_name", "ConvNeXt-Tiny ONNX"),
        key_prefix="fb_ident_low",
    )
    st.stop()

if result.get("confidence_tier") == "medium" and result.get("warning_message"):
    st.warning(translate_text(result["warning_message"], lang_code))

dis_diag = result.get("disease_diagnosis")
if dis_diag:
    for qw in dis_diag.get("quality_warnings", []):
        st.warning(f":material/warning: {translate_text(qw, lang_code)}")

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
        tier = result.get("confidence_tier", "high")
        if tier == "high":
            conf_badge = f":green-badge[Certainty: {score_pct:.1f}% — High Confidence]"
        else:
            conf_badge = f":orange-badge[Certainty: {score_pct:.1f}% — Medium Confidence (Verify Traits)]"

        st.markdown(conf_badge)

        if dis_diag:
            if not dis_diag.get("is_supported_crop", True) or dis_diag["health_status"] == "Unsupported":
                st.markdown(f":gray-badge[ℹ {translate_text(UNSUPPORTED_CROP_DISEASE_MESSAGE, lang_code)}]")
            elif dis_diag.get("confidence_tier") == "low":
                st.markdown(f":orange-badge[? {translate_text('Disease Diagnosis Uncertain (Low Confidence)', lang_code)}]")
            elif dis_diag["health_status"] == "Healthy":
                st.markdown(f":green-badge[✓ {translate_text('Foliar Health: Healthy', lang_code)} ({dis_diag['confidence_pct']}%) ]")
            else:
                st.markdown(f":red-badge[⚠ {translate_text('Pathology Detected', lang_code)}: {translate_text(dis_diag['disease_name'], lang_code)} ({dis_diag['confidence_pct']}%) ]")

        st.markdown(f"## {translate_text(result['common_name'], lang_code)}")
        st.markdown(f"### *{result['scientific_name']}*")

        p_meta = result.get("plant_metadata") or {}
        if p_meta.get("toxic"):
            st.warning(f":material/warning: **Toxicity Notice:** {translate_text(p_meta['toxic'], lang_code)}")

        st.space("small")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric(
                label=translate_text("Species Confidence (Model Output)", lang_code),
                value=f"{score_pct:.1f}%",
                help="Softmax probability output by the primary ConvNeXt-Tiny ONNX plant classification model.",
            )
        with stat_col2:
            if dis_diag and dis_diag.get("is_supported_crop", False):
                st.metric(
                    label=translate_text("Crop Health Status", lang_code),
                    value=translate_text(dis_diag["health_status"], lang_code),
                    help="Agricultural leaf disease classification result.",
                )
            else:
                st.metric(
                    label=translate_text("Disease Classification", lang_code),
                    value=translate_text("Not Available (Non-Crop)", lang_code),
                    help=UNSUPPORTED_CROP_DISEASE_MESSAGE,
                )

        cands = result.get("top_candidates", [])
        if len(cands) > 1:
            with st.expander("Top-5 species candidates (Primary Model Output)"):
                for c in cands:
                    st.markdown(f"- **{c['species']}** (*{c.get('scientific_name', '')}*) — `{c['confidence_pct']:.2f}%`")

        sec = result.get("secondary_suggestions", [])
        if sec:
            with st.expander("Other possible matches (low confidence — FloraSense 9,985-taxon secondary index)"):
                for s in sec:
                    st.markdown(f"- *{s['species']}* — `{s['confidence_pct']:.2f}%`")

        st.space("small")
        action_row = st.container(horizontal=True, gap="small")
        with action_row:
            if st.button(
                translate_text("Locate Nearby on Map", lang_code),
                icon=":material/map:",
            ):
                st.session_state.ar_query = result["scientific_name"]
                st.switch_page("app_pages/ar_explorer.py")
            if dis_diag and dis_diag.get("is_supported_crop", False) and st.button(
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
        wiki_link_title = result.get("wiki_title") or result["scientific_name"]
        definition_row(
            translate_text("Wikipedia Page", lang_code),
            f"[{wiki_link_title}](https://en.wikipedia.org/wiki/{wiki_link_title.replace(' ', '_')})",
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
        if not d_info.get("is_supported_crop", True) or d_info["health_status"] == "Unsupported":
            st.info(f":material/info: **{translate_text(UNSUPPORTED_CROP_DISEASE_MESSAGE, lang_code)}**")
            st.caption(
                translate_text(
                    "The 38-class agricultural crop ViT classifier supports only 14 crops (Apple, Blueberry, Cherry, Corn, "
                    "Grape, Orange/Citrus, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato). "
                    "Impossible cross-crop combinations are strictly blocked.",
                    lang_code,
                )
            )
            gc = d_info.get("general_condition") or {}
            if gc.get("available"):
                st.space("small")
                st.markdown(f"##### :material/neurology: {translate_text('7-Class General Foliar Condition & Care Assessment', lang_code)}")
                with st.container(border=True):
                    definition_row(
                        translate_text("Foliar Health Status", lang_code),
                        f"**{translate_text(gc['health_status'], lang_code)}**",
                        key="id-gc-health",
                    )
                    definition_row(
                        translate_text("General Condition (7-Class Head)", lang_code),
                        f"**{translate_text(gc['condition_name'], lang_code)}** ({gc['confidence_pct']}%)",
                        key="id-gc-cond",
                    )
                    definition_row(
                        translate_text("Causal Agent / Group", lang_code),
                        f"*{gc['pathogen']}*",
                        key="id-gc-agent",
                    )
                    definition_row(
                        translate_text("Symptoms", lang_code),
                        translate_text(gc["symptoms"], lang_code),
                        key="id-gc-symp",
                    )
                gc_recs = gc.get("recommendations", [])
                if gc_recs:
                    st.space("small")
                    st.markdown(f"#### {translate_text('Targeted Actionable Recommendations', lang_code)}")
                    with st.container(border=True):
                        for idx, r in enumerate(gc_recs, 1):
                            st.markdown(f"**{idx}.** {translate_text(r, lang_code)}")
                st.space("small")
                if st.button(
                    translate_text("Open Full Pathology Studio", lang_code),
                    icon=":material/biotech:",
                    type="primary",
                    key="btn_open_disease_studio_gc",
                ):
                    st.session_state.disease_result = d_info
                    st.switch_page("app_pages/disease_detection.py")
        elif d_info.get("confidence_tier") == "low":
            st.error(translate_text(d_info.get("uncertainty_message", "Disease could not be diagnosed reliably."), lang_code))
        else:
            if d_info.get("is_uncertain"):
                st.warning(translate_text(d_info.get("uncertainty_message", ""), lang_code))
            with st.container(border=True):
                definition_row(
                    translate_text("Health Status (Model Output)", lang_code),
                    f"**{translate_text(d_info['health_status'], lang_code)}**",
                    key="id-path-health",
                )
                definition_row(
                    translate_text("Condition (Model Output)", lang_code),
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
                if d_info.get("expert_review_notice"):
                    st.caption(f":material/warning: {translate_text(d_info['expert_review_notice'], lang_code)}")
            else:
                st.info(translate_text(d_info.get("no_recommendation_message") or "No corrective interventions required.", lang_code))

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


st.space("medium")
render_feedback_widget(
    task_type="plant_identification",
    predicted_label=f"{result['common_name']} ({result['scientific_name']})",
    confidence=result["score"],
    model_name=result.get("model_name", "ConvNeXt-Tiny ONNX"),
    key_prefix="fb_ident_main",
)
st.caption(
    ":material/copyright: Model Attribution: Plant ID via ConvNeXt-Tiny ONNX (`Apache-2.0` / `CC BY-NC-SA 4.0`, iNaturalist credits in `models/inat_attribution.csv`); "
    "Crop Disease ViT (`models/model.onnx`) licensed under **CC BY-SA 3.0** (`MODEL_LICENSES.md`)."
)
