"""
PlantVerse — Local-Only User Feedback Logger (`ui/feedback.py`)
===============================================================
Provides an optional, privacy-preserving "Was this correct?" feedback widget
(with explicit local-storage consent notice) that appends user-confirmed
verifications or corrections to `data/user_feedback_log.jsonl`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional

import streamlit as st

from ml_config import USER_FEEDBACK_LOG_PATH


def log_user_feedback(entry: Dict[str, Any]) -> None:
    """Append a single feedback record to the local JSONL file (`data/user_feedback_log.jsonl`)."""
    USER_FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with open(USER_FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_feedback_widget(
    task_type: str,
    predicted_label: str,
    confidence: float,
    model_name: str,
    key_prefix: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the optional, local-only 'Was this correct?' feedback UI block with
    an explicit consent notice so real-world accuracy can be measured over time.
    """
    with st.container(border=True):
        st.markdown("##### :material/fact_check: Was this prediction correct? *(Optional Local Feedback)*")
        st.caption(
            ":material/lock: **Local-Only Consent Notice:** Submitting feedback appends the predicted label, "
            "confidence score, and your optional correction strictly to a local file (`data/user_feedback_log.jsonl`) "
            "on this machine to help measure real-world accuracy. No images or personal data are transmitted externally."
        )
        consent = st.checkbox(
            "I consent to logging this verification locally to `data/user_feedback_log.jsonl`.",
            value=False,
            key=f"{key_prefix}_consent",
        )
        c1, c2 = st.columns([0.45, 0.55], vertical_alignment="bottom")
        with c1:
            verdict = st.radio(
                "Verification outcome",
                ["Yes, prediction was correct", "No, prediction was incorrect"],
                key=f"{key_prefix}_verdict",
                horizontal=True,
            )
        with c2:
            user_correction = st.text_input(
                "Expected plant / disease (if incorrect)",
                placeholder="e.g., Tulsi (Ocimum tenuiflorum) or Potato Late Blight",
                key=f"{key_prefix}_correction",
            )

        if st.button(
            "Log Verification Locally",
            icon=":material/save:",
            disabled=not consent,
            key=f"{key_prefix}_submit",
        ):
            is_correct = verdict.startswith("Yes")
            log_user_feedback(
                {
                    "task_type": task_type,
                    "predicted_label": predicted_label,
                    "confidence": round(float(confidence), 4),
                    "model_name": model_name,
                    "user_confirmed_correct": is_correct,
                    "user_expected_label": user_correction.strip() if not is_correct else predicted_label,
                    **(extra_meta or {}),
                }
            )
            st.success("Thank you — your feedback was logged locally to `data/user_feedback_log.jsonl`.")
