import streamlit as st

from ui.chrome import kicker, section_rule

# ==============================================================================
# MAJOR PROJECT HEADER RIBBON
# ==============================================================================
ribbon_left, ribbon_right = st.columns([1.4, 0.6], vertical_alignment="center")
with ribbon_left:
    st.markdown(
        """<div class="pv-academic-badge">
            <span>🌿</span>
            <span>Major Project Submission • Deep Learning & Bio-Informatics</span>
        </div>""",
        unsafe_allow_html=True,
    )
with ribbon_right:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.markdown(":green-badge[● FloraSense Model Online] :blue-badge[v1.0 Research]")

# ==============================================================================
# HERO SECTION
# ==============================================================================
hero_copy, hero_image = st.columns([1.18, 0.82], gap="large", vertical_alignment="center")

with hero_copy:
    with st.container(key="hero-title"):
        st.title("Know your plant.")
        st.title("Understand nature.")

    st.space("small")
    with st.container(key="hero-copy"):
        st.markdown(
            "PlantVerse is an end-to-end multimodal botanical intelligence framework developed "
            "to bridge deep computer vision with open biodiversity knowledge graphs. The system "
            "identifies plant specimens from field photographs, traverses formal taxonomic hierarchies "
            "via Wikidata, extracts pharmacological literature from Wikipedia, and maps wild "
            "geospatial observations using verified citizen science records."
        )

    st.space("medium")
    actions = st.container(horizontal=True, gap="small")
    with actions:
        if st.button("Launch Identifier", type="primary", icon=":material/document_scanner:"):
            st.switch_page("app_pages/identify.py")
        if st.button("Disease Detection", icon=":material/healing:"):
            st.switch_page("app_pages/disease_detection.py")
        if st.button("Explore Plant Library", icon=":material/menu_book:"):
            st.switch_page("app_pages/library.py")
        if st.button("Geospatial Field Map", icon=":material/location_on:"):
            st.switch_page("app_pages/ar_explorer.py")

with hero_image:
    with st.container(key="hero-image"):
        st.image(
            "static/hero.jpg",
            alt="Maidenhair fern fronds against a dark ground",
            width="stretch",
        )
    st.caption(
        "**Adiantum pedatum** *(Northern Maidenhair Fern)* • Specimen record via Wikimedia Commons."
    )

st.space("medium")

# ==============================================================================
# SYSTEM BENCHMARKS & METRICS STRIP
# ==============================================================================
m1, m2, m3, m4, m5 = st.columns(5, gap="small")
with m1:
    st.metric(label="Species Scope", value="1,000+", help="Classes covered by the FloraSense classification model.")
with m2:
    st.metric(label="Pathology Scope", value="38 Diseases", help="Foliar diseases diagnosed across major agricultural crops.")
with m3:
    st.metric(label="Vision Pipeline", value="Dual Engine", help="FloraSense (Species) + MobileNetV2 / CNN (Pathology).")
with m4:
    st.metric(label="Integrated Sources", value="5 Ontologies", help="Hugging Face, Wikidata, Wikipedia REST, PlantVillage, and iNaturalist.")
with m5:
    st.metric(label="Language Coverage", value="4 Indic & Eng", help="English, Hindi, Telugu, and Tamil translation support.")

st.space("large")
section_rule("section-rule-features")
st.space("medium")

# ==============================================================================
# FOUR CORE RESEARCH CAPABILITIES
# ==============================================================================
kicker("System Architecture", key="kicker-features")
st.subheader("Four pillars of botanical intelligence")
st.caption("A multi-tier architecture connecting visual pattern recognition, foliar pathology, ontological knowledge, and spatial ecology.")
st.space("small")

one, two = st.columns(2, gap="medium")
with one:
    with st.container(border=True):
        st.caption(":material/document_scanner: **PILLAR 01 — NEURAL SPECIES VISION**")
        st.markdown("### Species Classification")
        st.markdown(
            "Analyzes morphological structures of leaf margins, venation, flower blossoms, and canopy "
            "silhouettes. The FloraSense vision model generates confidence probabilities to disambiguate "
            "closely related botanical species."
        )

with two:
    with st.container(border=True):
        st.caption(":material/healing: **PILLAR 02 — FOLIAR PATHOLOGY & DIAGNOSTICS**")
        st.markdown("### Plant Disease Detection")
        st.markdown(
            "Screens leaves for bacterial spots, early and late blights, powdery mildews, and viral curl complexes. "
            "Diagnoses conditions across 38 crop classes, estimates diagnostic certainty, and provides validated "
            "agronomic recommendations."
        )

st.space("small")
three, four = st.columns(2, gap="medium")
with three:
    with st.container(border=True):
        st.caption(":material/account_tree: **PILLAR 03 — KNOWLEDGE ONTOLOGY**")
        st.markdown("### Taxonomic Hierarchy")
        st.markdown(
            "Integrates Wikidata SPARQL and claim entity structures to synthesize complete botanical lineages "
            "from Kingdom down to Genus. Cross-references Wikipedia articles to isolate verified "
            "ethnobotanical and medicinal notes."
        )

with four:
    with st.container(border=True):
        st.caption(":material/explore: **PILLAR 04 — FIELD GIS & OBSERVATIONS**")
        st.markdown("### Spatial Observations & Vouchers")
        st.markdown(
            "Connects to the global iNaturalist repository and local disease logs to map research-grade wild sightings "
            "within user-defined geographic radii, computing geodesic distances and presenting interactive "
            "mapping layers."
        )

st.space("large")
section_rule("section-rule-gallery")
st.space("medium")

# ==============================================================================
# CURATED HERBARIUM ARCHIVES SHOWCASE
# ==============================================================================
kicker("Herbarium Archives", key="kicker-gallery")
st.subheader("Reference botanical specimens")
st.caption("Curated botanical plates illustrating gymnosperm, pteridophyte, and angiosperm morphology.")
st.space("small")

g_col1, g_col2, g_col3 = st.columns(3, gap="medium")

with g_col1:
    with st.container(border=True):
        st.image("static/ginkgo.jpg", alt="Ginkgo biloba foliage", width="stretch")
        st.markdown("#### Ginkgo biloba")
        st.caption(":material/eco: Ginkgoaceae • Living Gymnosperm Fossil")
        st.markdown(
            "An ancient deciduous gymnosperm lineage unchanged for over 200 million years. Notable for "
            "its fan-shaped leaves, dichotomous venation, and resistance to environmental stress."
        )

with g_col2:
    with st.container(border=True):
        st.image("static/herbarium.jpg", alt="Pressed herbarium specimen sheet", width="stretch")
        st.markdown("#### Herbarium Plate")
        st.caption(":material/menu_book: Vascular Archive • Morphological Voucher")
        st.markdown(
            "Preserved botanical voucher documenting diagnostic characters including leaf attachment, "
            "stem nodes, and reproductive inflorescence for comparative taxonomic verification."
        )

with g_col3:
    with st.container(border=True):
        st.image("static/fern-plate.jpg", alt="Fern fronds specimen", width="stretch")
        st.markdown("#### Polypodiopsida")
        st.caption(":material/forest: Pteridophyta • Spore Morphology")
        st.markdown(
            "Non-flowering vascular plants reproducing via microscopic spores housed in sporangia. "
            "Features compound pinnate fronds and primitive vascular bundles."
        )

st.space("large")
section_rule("section-rule-workflow")
st.space("medium")

# ==============================================================================
# PIPELINE & WORKFLOW
# ==============================================================================
kicker("Methodology", key="kicker-workflow")
st.subheader("Automated identification pipeline")
st.space("small")

w1, w2, w3 = st.columns(3, gap="medium")

with w1:
    with st.container(border=True):
        st.caption(":material/camera: **PHASE 1**")
        st.markdown("##### 1. Image Acquisition")
        st.markdown(
            "Field photograph is captured and preprocessed into normalized tensor dimensions with RGB "
            "channel standardization."
        )

with w2:
    with st.container(border=True):
        st.caption(":material/memory: **PHASE 2**")
        st.markdown("##### 2. Deep Inference")
        st.markdown(
            "Deep neural classifier computes top predictions, probability distributions, and validates "
            "taxonomic entity keys."
        )

with w3:
    with st.container(border=True):
        st.caption(":material/biotech: **PHASE 3**")
        st.markdown("##### 3. Knowledge Synthesis")
        st.markdown(
            "Wikidata taxonomy, Wikipedia pharmacology, and iNaturalist geospatial observations are "
            "compiled into a complete botanical profile."
        )

st.space("large")

# ==============================================================================
# MAJOR PROJECT CTA & ATTRIBUTION
# ==============================================================================
with st.container(border=True):
    cta_col1, cta_col2 = st.columns([1.35, 0.65], vertical_alignment="center")
    with cta_col1:
        st.markdown("### Ready to evaluate specimen identification?")
        st.markdown(
            "Experience the neural identification pipeline in real time. Upload any clear plant photograph "
            "to generate an authoritative botanical dossier complete with taxonomy and local sightings."
        )
    with cta_col2:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button("Start Identification", type="primary", icon=":material/arrow_forward:"):
                st.switch_page("app_pages/identify.py")
