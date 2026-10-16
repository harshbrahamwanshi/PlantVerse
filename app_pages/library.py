import streamlit as st

from ar_location_plant_map import PlantLocationFinder
from ui.chrome import definition_row, init_state, kicker, section_rule
from ui.presentation import extra_taxon_fields, photo_attribution, taxon_photo_url

init_state()
finder = PlantLocationFinder()

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """<div class="pv-academic-badge">
        <span>📖</span>
        <span>Taxonomic Specimen Index & Digital Herbarium</span>
    </div>""",
    unsafe_allow_html=True,
)
st.header("Botanical Library & Specimen Index")
st.markdown(
    "Query global iNaturalist biological records to inspect validated botanical taxons, "
    "or review specimens identified during your current research session. All records synthesize "
    "morphological descriptions, taxonomic ranks, and verified authority citations."
)

st.space("small")
section_rule("section-rule-lib-search")
st.space("small")

# ==============================================================================
# SEARCH WORKSPACE & QUICK SUGGESTIONS
# ==============================================================================
with st.container(border=True):
    st.markdown("#### :material/search: Query Global Species Database")
    st.caption("Search across worldwide plant taxons using common vernacular names or botanical Latin binomials.")

    search_col, btn_col = st.columns([1.5, 0.5], vertical_alignment="bottom")
    with search_col:
        search = st.text_input(
            "Species or vernacular common name",
            placeholder="e.g. Neem, Ginkgo, Rose, Adiantum, Oak...",
            label_visibility="collapsed",
            key="lib_search_input",
        )
    with btn_col:
        search_clicked = st.button("Search Database", type="primary", icon=":material/search:")

    st.space("small")
    st.caption("Quick evaluation shortcuts:")
    quick_cols = st.container(horizontal=True, gap="small")
    with quick_cols:
        if st.button("🌿 Neem"):
            search = "Neem"
            search_clicked = True
        if st.button("🍃 Ginkgo"):
            search = "Ginkgo"
            search_clicked = True
        if st.button("🌱 Fern"):
            search = "Fern"
            search_clicked = True
        if st.button("🌹 Rose"):
            search = "Rose"
            search_clicked = True
        if st.button("🌳 Oak"):
            search = "Oak"
            search_clicked = True

if search_clicked:
    query = (search or "").strip()
    if not query:
        st.warning("Please specify a valid botanical or common name to initiate query.")
    else:
        with st.spinner(f"Querying species records for '{query}' on iNaturalist..."):
            plant_info = finder.search_plant_species(query)
            if not plant_info:
                st.session_state.library_profile = None
                st.warning(f"Could not resolve plant species '{query}'. Please verify spelling or try an alternate binomial.")
            else:
                scientific_name = plant_info.get("name", query)
                description = finder.get_wikipedia_summary(scientific_name)
                st.session_state.library_profile = {
                    "query": query,
                    "plant_info": plant_info,
                    "description": description,
                }

# ==============================================================================
# SEARCH RESULT PROFILE
# ==============================================================================
profile = st.session_state.get("library_profile")
if profile:
    plant_info = profile["plant_info"]
    scientific_name = plant_info.get("name", profile.get("query", ""))
    common_name = plant_info.get("preferred_common_name") or scientific_name
    photo_url = taxon_photo_url(plant_info)
    credit = photo_attribution(plant_info)

    st.space("medium")
    kicker("Species Taxonomic Dossier", key="kicker-lib-profile")

    with st.container(border=True):
        visual_col, meta_col = st.columns([0.88, 1.12], gap="large", vertical_alignment="center")

        with visual_col:
            with st.container(key="library-image"):
                if photo_url:
                    st.image(photo_url, alt=f"Photograph of {common_name}", width="stretch")
                    if credit:
                        st.caption(credit)
                else:
                    st.image(
                        "static/herbarium.jpg",
                        alt="Herbarium sheet",
                        width="stretch",
                    )
                    st.caption("Reference herbarium voucher plate via Wikimedia Commons.")

        with meta_col:
            rank = plant_info.get("rank")
            if rank:
                st.markdown(f":green-badge[{str(rank).replace('_', ' ').upper()}]")
            st.markdown(f"## {common_name}")
            st.markdown(f"### *{scientific_name}*")

            st.space("small")
            with st.container(border=True):
                definition_row("Common Vernacular", str(common_name), key="lib-common")
                definition_row("Scientific Binomial", f"*{scientific_name}*", key="lib-sci")
                if rank:
                    definition_row("Taxonomic Rank", str(rank).replace("_", " ").title(), key="lib-rank")

            st.space("small")
            if st.button("Locate Nearby in AR Explorer", icon=":material/map:", type="primary"):
                st.session_state.ar_query = scientific_name
                st.switch_page("app_pages/ar_explorer.py")

    st.space("medium")
    tab_desc, tab_tax = st.tabs([":material/description: Botanical Abstract", ":material/info: Taxonomy & Source References"])

    with tab_desc:
        st.space("small")
        with st.container(border=True):
            st.markdown(profile.get("description") or "Description not available from Wikipedia.")
            st.caption(":material/menu_book: Reference Source: Wikipedia Summary API")

    with tab_tax:
        st.space("small")
        extras = extra_taxon_fields(plant_info)
        if extras:
            with st.container(border=True):
                for label, value in extras:
                    definition_row(label, value, key=f"lib-extra-{label}")
        else:
            st.info("No auxiliary taxon fields recorded for this entry.")

st.space("large")
section_rule("section-rule-session")
st.space("small")

# ==============================================================================
# SESSION HERBARIUM & PATHOLOGY HISTORY
# ==============================================================================
kicker("Field Voucher & Pathology Log", key="kicker-session")

tab_vouchers, tab_path_hist = st.tabs(
    [
        ":material/eco: Botanical Species Vouchers",
        ":material/biotech: Pathology & Disease Diagnoses",
    ]
)

with tab_vouchers:
    entries = st.session_state.get("library_entries") or []

    header_col, count_col = st.columns([1.5, 0.5], vertical_alignment="center")
    with header_col:
        st.subheader("Vouchers identified in this session")
    with count_col:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.caption(f":material/inventory_2: **{len(entries)} specimens cataloged**")

    if not entries:
        with st.container(border=True):
            empty_col, plate_col = st.columns([1.25, 0.75], gap="large", vertical_alignment="center")
            with empty_col:
                st.markdown("#### No specimen vouchers recorded yet")
                st.markdown(
                    "When you identify botanical specimens in the Identification Studio, "
                    "their photographic vouchers, classification metrics, and taxonomy are "
                    "automatically cataloged in this digital herbarium for subsequent research."
                )
                st.space("small")
                if st.button("Initiate First Specimen Identification", type="primary", icon=":material/document_scanner:", key="lib_init_ident"):
                    st.switch_page("app_pages/identify.py")
            with plate_col:
                st.image(
                    "static/herbarium.jpg",
                    alt="Pressed herbarium specimen",
                    width="stretch",
                )
                st.caption("Reference botanical plate • Wikimedia Commons")
    else:
        for index in range(0, len(entries), 3):
            row = st.columns(3, gap="medium")
            for offset, (col, entry) in enumerate(zip(row, entries[index : index + 3])):
                with col:
                    with st.container(border=True):
                        if entry.get("image_bytes"):
                            with st.container(key=f"session-image-{index}-{offset}"):
                                st.image(
                                    entry["image_bytes"],
                                    alt=entry.get("common_name") or "Identified plant",
                                    width="stretch",
                                )
                        st.space("small")
                        st.caption(f":material/eco: {entry.get('scientific_name') or 'Botanical Record'}")
                        st.markdown(f"#### {entry.get('common_name', 'Unknown')}")
                        score = entry.get("score")
                        if score is not None:
                            st.markdown(f":green-badge[Certainty: {score * 100:.1f}%]")
                        st.space("small")
                        if st.button(
                            "Map Habitat",
                            key=f"btn-map-{index}-{offset}",
                            icon=":material/map:",
                        ):
                            st.session_state.ar_query = entry.get("scientific_name") or entry.get("common_name")
                            st.switch_page("app_pages/ar_explorer.py")

with tab_path_hist:
    diagnoses = st.session_state.get("disease_diagnoses") or []

    d_header_col, d_count_col = st.columns([1.5, 0.5], vertical_alignment="center")
    with d_header_col:
        st.subheader("Foliar Pathology & Disease Diagnoses History")
    with d_count_col:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.caption(f":material/healing: **{len(diagnoses)} diagnoses recorded**")

    if not diagnoses:
        with st.container(border=True):
            d_empty_col, d_plate_col = st.columns([1.25, 0.75], gap="large", vertical_alignment="center")
            with d_empty_col:
                st.markdown("#### No disease diagnoses recorded yet")
                st.markdown(
                    "When you screen plant specimens in the Disease Detection Studio or Identification Studio, "
                    "their pathological diagnoses, health status evaluations, and remediation records are "
                    "saved here for comparative agronomic tracking."
                )
                st.space("small")
                if st.button("Initiate Foliar Disease Screening", type="primary", icon=":material/biotech:", key="lib_init_disease"):
                    st.switch_page("app_pages/disease_detection.py")
            with d_plate_col:
                st.image(
                    "static/fern-plate.jpg",
                    alt="Botanical foliar specimen",
                    width="stretch",
                )
                st.caption("Reference foliar plate • Wikimedia Commons")
    else:
        for d_idx in range(0, len(diagnoses), 3):
            d_row = st.columns(3, gap="medium")
            for d_offset, (d_col, diag) in enumerate(zip(d_row, diagnoses[d_idx : d_idx + 3])):
                with d_col:
                    with st.container(border=True):
                        if diag.get("image_bytes"):
                            with st.container(key=f"diag-image-{d_idx}-{d_offset}"):
                                st.image(
                                    diag["image_bytes"],
                                    alt=diag.get("disease_name") or "Diagnosed leaf",
                                    width="stretch",
                                )
                        st.space("small")
                        st.caption(f":material/calendar_today: {diag.get('created_at', 'Session Record')}")

                        d_status = diag.get("health_status", "Diseased")
                        if d_status == "Healthy":
                            st.markdown(":green-badge[✓ Healthy Foliage]")
                        else:
                            st.markdown(":red-badge[⚠ Pathology Detected]")

                        st.markdown(f"#### {diag.get('plant_name', 'Plant')} — {diag.get('disease_name', 'Condition')}")
                        d_conf = diag.get("confidence_pct", diag.get("confidence", 0) * 100)
                        st.caption(f"Certainty: **{d_conf:.1f}%** • Pathogen: *{diag.get('pathogen', 'N/A')}*")

                        if diag.get("is_sighting"):
                            st.caption(f":material/pin_drop: Locked Sighting: **{diag.get('location_name', 'Field Location')}**")

                        st.space("small")
                        action_c1, action_c2 = st.columns(2)
                        with action_c1:
                            if st.button("Inspect", key=f"btn-inspect-diag-{d_idx}-{d_offset}", icon=":material/zoom_in:"):
                                st.session_state.disease_result = diag
                                st.switch_page("app_pages/disease_detection.py")
                        with action_c2:
                            if st.button("Map", key=f"btn-map-diag-{d_idx}-{d_offset}", icon=":material/map:"):
                                st.session_state.ar_query = diag.get("plant_name", "Plant")
                                st.switch_page("app_pages/ar_explorer.py")

