from pathlib import Path
from typing import Optional

import streamlit as st

_CSS_PATH = Path(__file__).with_name("styles.css")


def apply_chrome() -> None:
    """Inject the PlantVerse stylesheet once per run."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def init_state() -> None:
    st.session_state.setdefault("library_entries", [])
    if "disease_diagnoses" not in st.session_state or not st.session_state.disease_diagnoses:
        from ui.presentation import load_persistent_diagnoses
        st.session_state.disease_diagnoses = load_persistent_diagnoses()
    st.session_state.setdefault("ident_result", None)
    st.session_state.setdefault("ident_file_id", None)
    st.session_state.setdefault("disease_result", None)
    st.session_state.setdefault("disease_file_id", None)
    st.session_state.setdefault("ar_lat", None)
    st.session_state.setdefault("ar_lon", None)
    st.session_state.setdefault("ar_results", None)
    st.session_state.setdefault("ar_plant_info", None)
    st.session_state.setdefault("ar_description", None)
    st.session_state.setdefault("ar_query", None)
    st.session_state.setdefault("ar_radius", None)
    st.session_state.setdefault("library_profile", None)


def kicker(text: str, key: str = "kicker") -> None:
    with st.container(key=key):
        st.caption(f":material/eco: {text.upper()}")


def pill_badge(text: str, icon: str = ":material/eco:") -> None:
    st.markdown(f"{icon} **{text}**")


def section_rule(key: str = "rule") -> None:
    with st.container(key=key):
        st.markdown("---")


def page_intro(
    label: str,
    title: str,
    body: Optional[str] = None,
    key: str = "intro",
) -> None:
    kicker(label, key=f"kicker-{key}")
    st.header(title)
    if body:
        with st.container(key=f"lede-{key}"):
            st.markdown(body)


def definition_row(label: str, value: str, key: str = "row") -> None:
    with st.container(key=f"def-row-{key}"):
        left, right = st.columns([1, 2.2], vertical_alignment="center")
        with left:
            st.caption(f"**{label.upper()}**")
        with right:
            st.markdown(value)
