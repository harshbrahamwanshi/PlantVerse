import streamlit as st

from ui.chrome import kicker, section_rule

# ==============================================================================
# HEADER HERO & PROJECT ABSTRACT
# ==============================================================================
intro, image = st.columns([1.18, 0.82], gap="large", vertical_alignment="center")

with intro:
    st.markdown(
        """<div class="pv-academic-badge">
            <span>🌿</span>
            <span>Major Project Specification • Research Documentation</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.header("About PlantVerse")
    st.markdown(
        "PlantVerse is a comprehensive computer vision and bio-computational framework developed "
        "to democratize botanical research and biodiversity conservation. The project bridges "
        "deep neural image classification with open global knowledge ontologies, enabling instant species "
        "disambiguation, dynamic biological taxonomy reconstruction, pharmacological text synthesis, and "
        "georeferenced spatial habitat mapping."
    )
    st.space("small")
    st.caption(
        ":material/verified: Academic Major Project Submission • Multimodal AI & Computational Biodiversity"
    )

with image:
    with st.container(key="plate-image"):
        st.image(
            "static/ginkgo.jpg",
            alt="Ginkgo biloba botanical specimen plate",
            width="stretch",
        )
    st.caption("**Ginkgo biloba** *(Maidenhair Tree)* • Preserved reference plate via Wikimedia Commons.")

st.space("large")
section_rule("section-rule-about")
st.space("medium")

# ==============================================================================
# ==============================================================================
# 5-LAYER SYSTEM ARCHITECTURE
# ==============================================================================
kicker("System Architecture", key="kicker-arch")
st.subheader("Multi-tier computational architecture")
st.caption("PlantVerse coordinates five specialized data, computer vision, and machine learning tiers in real time:")
st.space("small")

a1, a2 = st.columns(2, gap="medium")

with a1:
    with st.container(border=True):
        st.caption(":material/memory: **LAYER 01 — NEURAL SPECIES VISION**")
        st.markdown("#### Morphological Inference Engine")
        st.markdown(
            "Powered by the FloraSense classification model hosted on Hugging Face. The model evaluates "
            "subtle morphological patterns across leaf margins, cellular venation grids, and floral organ "
            "symmetry to compute class probabilities across thousands of plant taxa."
        )

with a2:
    with st.container(border=True):
        st.caption(":material/healing: **LAYER 02 — FOLIAR PATHOLOGY CLASSIFIER**")
        st.markdown("#### Crop Disease Diagnostic Engine")
        st.markdown(
            "Based on the Plant Disease Prediction architecture (referenced from "
            "[animesh1012/machineLearning](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction)), "
            "evaluating 38 distinct agricultural pathology classes across crops including Apple, Tomato, Potato, Corn, "
            "Grape, and Citrus with calibrated confidence estimation."
        )

st.space("small")
b1, b2, b3 = st.columns(3, gap="medium")

with b1:
    with st.container(border=True):
        st.caption(":material/account_tree: **LAYER 03 — KNOWLEDGE ONTOLOGY**")
        st.markdown("#### Wikidata Graph Resolution")
        st.markdown(
            "Dynamically parses Wikidata entity claim networks. Traverses biological relationships "
            "including Taxon Name (P225), Parent Taxon (P171), and Taxonomic Rank (P105) to reconstruct the "
            "complete phylogenetic tree from Kingdom down to Genus."
        )

with b2:
    with st.container(border=True):
        st.caption(":material/menu_book: **LAYER 04 — PHARMACOLOGICAL & AGRONOMIC NLP**")
        st.markdown("#### Botanical Information Synthesis")
        st.markdown(
            "Extracts botanical summaries, medicinal history, and targeted agronomic recommendations for "
            "identified foliar diseases, synthesizing cultural practices, mechanical aeration, and biological treatments."
        )

with b3:
    with st.container(border=True):
        st.caption(":material/explore: **LAYER 05 — SPATIAL BIODIVERSITY & SIGHTINGS GIS**")
        st.markdown("#### Field Observations & Geospatial Vouchers")
        st.markdown(
            "Connects to the global iNaturalist citizen science API and local researcher sightings log. "
            "Calculates geodesic proximity using the Haversine formula and maps observation vouchers in real time."
        )

st.space("large")
section_rule("section-rule-specs")
st.space("medium")

# ==============================================================================
# TECHNICAL SPECIFICATIONS & STACK
# ==============================================================================
kicker("Technical Stack", key="kicker-specs")
st.subheader("Implementation & system specifications")
st.space("small")

spec_col1, spec_col2 = st.columns(2, gap="medium")

with spec_col1:
    with st.container(border=True):
        st.markdown("#### Core Software Libraries & ML Pipeline")
        st.markdown(
            "- **Framework**: Streamlit 1.65 (Component v2 Architecture)\n"
            "- **Primary Species Identification Pipeline**: ConvNeXt-Tiny ONNX (`models/plant_model.onnx`, 68 species + `not_plant` + 7 foliar conditions)\n"
            "- **Secondary Taxonomic Index**: `Sisigoks/FloraSense` (9,985 classes, optional low-confidence suggestions)\n"
            "- **Primary Crop Disease Engine**: Vision Transformer ONNX (`models/model.onnx`, 38 PlantVillage + PlantDoc classes, gated by host crop)\n"
            "- **Classical CV Lesion & Severity Analysis**: OpenCV HSV + Excess-Green ($2G - R - B > 25$) + CIE LAB border background segmentation\n"
            "- **Standalone ML Microservice**: ASGI / Starlette & Uvicorn REST Service\n"
            "- **Knowledge Parsing**: Wikidata SPARQL & REST API, Wikipedia REST Summary API\n"
            "- **Geospatial & Geodesy**: iNaturalist API + Haversine Spherical Distance Algorithm"
        )

with spec_col2:
    with st.container(border=True):
        st.markdown("#### Dataset & Model Licenses (Attribution & ShareAlike Notice)")
        st.markdown(
            "- **Primary Disease Model (`models/model.onnx`)**: Fine-tuned from [`kimcomehome/plantvillage-vit-leaf-disease`](https://huggingface.co/kimcomehome/plantvillage-vit-leaf-disease) (`google/vit-base-patch16-224`, Apache-2.0) on PlantVillage + PlantDoc field photos. Licensed under **[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)** — any adaptations or redistributions of these weights must retain attribution and be shared under CC BY-SA 3.0.\n"
            "- **Primary Plant Model (`models/plant_model.onnx`)**: ConvNeXt-Tiny (`timm`, Apache-2.0) trained on openly licensed iNaturalist observations (`CC0`, `CC BY`, `CC BY-SA`, `CC BY-NC`, `CC BY-NC-SA`), PlantVillage, PlantDoc, and Imagenette. Distributed for non-commercial academic/educational use under **CC BY-NC-SA 4.0**; per-photo photographer credits for all 26,946 observations are included in `models/inat_attribution.csv`.\n"
            "- **Benchmark & Training Datasets**: PlantVillage (Hughes & Salathé, 2015, CC0 / CC BY-SA 3.0) and PlantDoc (Singh et al., 2020, ACM CoDS COMAD, CC BY 4.0).\n"
            "- **Full License Registry**: See `MODEL_LICENSES.md` in the repository root."
        )

st.space("large")
section_rule("section-rule-ethics")
st.space("medium")

# ==============================================================================
# ETHICAL & CLINICAL ADVISORY
# ==============================================================================
kicker("Scientific Governance", key="kicker-ethics")

with st.container(border=True):
    st.markdown("#### :material/health_and_safety: Academic & Clinical Safety Advisory")
    st.markdown(
        "PlantVerse is engineered strictly as an educational research framework and taxonomic inquiry tool. "
        "It does not operate as a clinical diagnostic instrument, agricultural safety certification, or foraging authority.\n\n"
        "- **Never consume wild plants or fungi** relying solely on neural network identification.\n"
        "- Numerous lethal or toxic species exhibit close morphological mimicry with common edible herbs.\n"
        "- All pharmacological notes are drawn from open literature for historical scholarship.\n"
        "- Always consult certified botanists, toxicologists, or healthcare specialists before interacting with unfamiliar flora."
    )
    st.space("small")
    st.caption(
        "Curated and compiled for Major Project Evaluation • Models licensed under CC BY-SA 3.0 / CC BY-NC-SA 4.0 (see MODEL_LICENSES.md)."
    )

