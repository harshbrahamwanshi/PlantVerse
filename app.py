import streamlit as st

from ui.chrome import apply_chrome, init_state

st.set_page_config(
    page_title="PlantVerse — Botanical Intelligence",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.logo("static/logo.svg", size="large")
apply_chrome()
init_state()

explore = st.Page(
    "app_pages/explore.py",
    title="Explore",
    icon=":material/forest:",
    url_path="explore",
    default=True,
)
identify = st.Page(
    "app_pages/identify.py",
    title="Identify",
    icon=":material/document_scanner:",
    url_path="identify",
)
disease_detection = st.Page(
    "app_pages/disease_detection.py",
    title="Disease Detection",
    icon=":material/healing:",
    url_path="disease-detection",
)
library = st.Page(
    "app_pages/library.py",
    title="Plant Library",
    icon=":material/menu_book:",
    url_path="library",
)
ar_explorer = st.Page(
    "app_pages/ar_explorer.py",
    title="AR Explorer",
    icon=":material/explore:",
    url_path="ar-explorer",
)
about = st.Page(
    "app_pages/about.py",
    title="About",
    icon=":material/info:",
    url_path="about",
)

page = st.navigation(
    [explore, identify, disease_detection, library, ar_explorer, about],
    position="top",
)
page.run()
