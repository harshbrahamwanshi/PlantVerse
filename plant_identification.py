# ============================================================
# PlantVerse - Plant Identification
# ============================================================

from functools import lru_cache
import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageOps
import requests
import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ml_config import (
    FLORASENSE_HIGH_CONF_THRESHOLD,
    FLORASENSE_INPUT_SIZE,
    FLORASENSE_MEAN,
    FLORASENSE_MEDIUM_CONF_THRESHOLD,
    FLORASENSE_MIN_TOP1_TOP2_RATIO,
    FLORASENSE_MODEL_NAME,
    FLORASENSE_STD,
    LOW_CONFIDENCE_PLANT_MESSAGE,
    SUPPORTED_PLANT_SUMMARY,
)


# ============================================================
# 1. Language Setup
# ============================================================

LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
}


# ============================================================
# 2. Translation Function
# ============================================================

@st.cache_data
def translate_text(text, target_lang):
    """
    Translate English text into the selected language.

    If Google Translate is unavailable, the original English
    text is returned instead of crashing the application.
    """

    # No translation required
    if not text or target_lang == "en":
        return text

    try:
        url = "https://translate.googleapis.com/translate_a/single"

        params = {
            "client": "gtx",
            "sl": "en",
            "tl": target_lang,
            "dt": "t",
            "q": text
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        response.raise_for_status()

        data = response.json()

        if data and data[0]:
            return "".join(
                part[0]
                for part in data[0]
                if part and len(part) > 0
            )

        return text

    except requests.exceptions.RequestException as e:
        print(f"Translation request failed: {e}")
        return text

    except (ValueError, IndexError, TypeError) as e:
        print(f"Translation returned invalid data: {e}")
        return text

    except Exception as e:
        print(f"Translation error: {e}")
        return text


# ============================================================
# 3. Plant Classifier & Preprocessing
# ============================================================

_CROP_BOTANICAL_MAP = {
    "Squash": "Cucurbita",
    "Orange": "Orange (fruit)",
    "Corn (Maize)": "Maize",
    "Corn": "Maize",
    "Cherry (including sour)": "Prunus cerasus",
    "Cherry": "Prunus avium",
    "Pepper, bell": "Bell pepper",
    "Bell Pepper": "Bell pepper",
}


@st.cache_resource
def load_classifier() -> Dict[str, Any]:
    """
    Load the FloraSense Vision Transformer model and processor in eval mode.
    Raises RuntimeError with a clear message if the model or weights cannot be loaded.
    """
    try:
        model = AutoModelForImageClassification.from_pretrained(FLORASENSE_MODEL_NAME)
        model.eval()
        processor = AutoImageProcessor.from_pretrained(FLORASENSE_MODEL_NAME)
        return {
            "model": model,
            "processor": processor,
            "id2label": model.config.id2label,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load plant identification model '{FLORASENSE_MODEL_NAME}'. "
            f"Verify model weights are available locally or check network connection: {exc}"
        ) from exc


def _clean_florasense_label(raw_label: str) -> str:
    """
    Convert raw FloraSense labels like 'Adiantum_pedatum_L_' or
    'Valerianella_locusta__L___Laterr_' into a clean binomial 'Adiantum pedatum'.
    """
    if not raw_label:
        return "Unknown"
    if "_" not in raw_label:
        return raw_label.strip()
    parts = [p for p in raw_label.replace("__", "_").split("_") if p]
    if len(parts) >= 2 and parts[0][0].isupper() and parts[1][0].islower():
        return f"{parts[0]} {parts[1]}"
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def preprocess_for_florasense(image: Image.Image) -> torch.Tensor:
    """
    Exact training-matched preprocessing for Sisigoks/FloraSense (google/vit-base-patch16-224):
    1. Apply EXIF orientation transpose so portrait mobile photos are upright.
    2. Convert to 3-channel RGB.
    3. Squash-resize to (224, 224) using BILINEAR interpolation (matching ViTImageProcessor default_to_square=True).
    4. Scale pixel intensities to [0.0, 1.0] (rescale_factor = 1/255).
    5. Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5].
    6. Transpose HWC -> CHW and add batch dimension [1, 3, 224, 224].
    """
    if image is None:
        raise ValueError("No image provided for plant identification.")
    img = ImageOps.exif_transpose(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width < 32 or img.height < 32:
        raise ValueError(f"Image is too small ({img.width}x{img.height} px). Minimum dimension is 32x32 px.")

    resized = img.resize(FLORASENSE_INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array(FLORASENSE_MEAN, dtype=np.float32)
    std = np.array(FLORASENSE_STD, dtype=np.float32)
    norm = (arr - mean) / std
    chw = np.transpose(norm, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def evaluate_plant_confidence_policy(
    top1_score: float,
    top2_score: float,
    high_threshold: float = FLORASENSE_HIGH_CONF_THRESHOLD,
    medium_threshold: float = FLORASENSE_MEDIUM_CONF_THRESHOLD,
    min_ratio: float = FLORASENSE_MIN_TOP1_TOP2_RATIO,
) -> Tuple[str, str]:
    """
    Evaluate confidence tier ('high', 'medium', 'low') using measured top-1 probability
    and top-1 / top-2 probability ratio.
    """
    ratio = top1_score / max(top2_score, 1e-7)
    if top1_score >= high_threshold and ratio >= min_ratio:
        return "high", ""
    if top1_score >= medium_threshold and ratio >= min_ratio:
        return (
            "medium",
            "Moderate confidence prediction: fine-grained visual similarity detected across related taxa. Verify diagnostic traits below.",
        )
    return "low", LOW_CONFIDENCE_PLANT_MESSAGE


def predict_species_detailed(image: Image.Image) -> Dict[str, Any]:
    """
    Run genuine plant species identification using FloraSense without any cross-model
    hijacking or fake fallbacks. Aggregates probability mass across duplicate synonym
    authorities in FloraSense's 9,985-class head and applies the confidence policy.
    """
    pixel_values = preprocess_for_florasense(image)
    bundle = load_classifier()
    model = bundle["model"]
    id2label = bundle["id2label"]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Aggregate top-50 raw classes by cleaned binomial so duplicate authority strings
    # (e.g. 'Adiantum_raddianum_C_Presl' and 'Adiantum_raddianum_C__Presl') sum together
    top_indices = np.argsort(probs)[::-1][:50]
    species_scores: Dict[str, float] = {}
    species_raw_map: Dict[str, str] = {}
    for idx in top_indices:
        raw_lbl = id2label[int(idx)]
        clean_lbl = _clean_florasense_label(raw_lbl)
        species_scores[clean_lbl] = species_scores.get(clean_lbl, 0.0) + float(probs[int(idx)])
        if clean_lbl not in species_raw_map:
            species_raw_map[clean_lbl] = raw_lbl

    ranked = sorted(species_scores.items(), key=lambda kv: -kv[1])
    top1_label, top1_score = ranked[0] if ranked else ("Unknown", 0.0)
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0

    tier, message = evaluate_plant_confidence_policy(top1_score, top2_score)
    top_candidates = [
        {
            "species": sp,
            "raw_label": species_raw_map.get(sp, sp),
            "score": float(sc),
            "confidence_pct": round(float(sc) * 100, 2),
        }
        for sp, sc in ranked[:5]
    ]

    return {
        "species": top1_label,
        "raw_label": species_raw_map.get(top1_label, top1_label),
        "score": float(top1_score),
        "top2_score": float(top2_score),
        "margin_ratio": round(float(top1_score) / max(float(top2_score), 1e-7), 2),
        "confidence_tier": tier,
        "is_reliable": tier in ("high", "medium"),
        "warning_message": message,
        "top_candidates": top_candidates,
        "model_name": FLORASENSE_MODEL_NAME,
    }


def predict_species(image: Image.Image) -> Tuple[str, float]:
    """
    Predict plant species from uploaded image.
    Returns (species_name, confidence_score).
    Note: Does NOT hijack predictions with the crop disease model.
    """
    res = predict_species_detailed(image)
    return res["species"], res["score"]


def get_supported_plants_info() -> Dict[str, Any]:
    """Return metadata on supported plant classes for config/UI display."""
    return SUPPORTED_PLANT_SUMMARY["florasense"]


# ============================================================
# 4. Wikipedia Search
# ============================================================

WIKIMEDIA_HEADERS = {
    "User-Agent": "PlantVerse/1.0 (https://github.com/plantverse/app; contact@plantverse.org) python-requests/2.31"
}


@st.cache_data
def get_wikipedia_title(name: str):
    """
    Search Wikipedia for the predicted plant.

    If Wikipedia is unavailable, return the original
    model prediction instead of crashing.
    """

    clean_query = _clean_florasense_label(name) if "_" in name else name

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_query,
        "format": "json",
        "utf8": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers=WIKIMEDIA_HEADERS
        )

        response.raise_for_status()

        # Make sure the server actually returned JSON
        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "json" not in content_type:
            print("Wikipedia returned a non-JSON response.")
            print(response.text[:300])
            return clean_query

        data = response.json()

        results = data.get(
            "query",
            {}
        ).get(
            "search",
            []
        )

        if results:
            return results[0].get("title", clean_query)

        return clean_query

    except requests.exceptions.RequestException as e:
        print(f"Wikipedia request failed: {e}")
        return clean_query

    except (ValueError, KeyError, TypeError) as e:
        print(f"Wikipedia returned invalid JSON/data: {e}")
        return clean_query

    except Exception as e:
        print(f"Wikipedia error: {e}")
        return clean_query


# ============================================================
# 5. Wikidata Taxonomy
# ============================================================

_KNOWN_RANKS = {
    "Q7432": "Species",
    "Q34740": "Genus",
    "Q227936": "Tribe",
    "Q164280": "Subfamily",
    "Q35409": "Family",
    "Q5867959": "Suborder",
    "Q36602": "Order",
    "Q5867051": "Subclass",
    "Q37517": "Class",
    "Q235536": "Subdivision",
    "Q334460": "Division",
    "Q38348": "Phylum",
    "Q2752679": "Subkingdom",
    "Q36732": "Kingdom",
    "Q146481": "Domain",
    "Q19858692": "Superdomain",
    "Q713623": "Clade",
}


@st.cache_data
def get_taxonomy_from_wikidata(label: str):
    """
    Get plant taxonomy from Wikidata.

    If Wikidata fails, return an error dictionary instead
    of crashing the application.
    """

    clean_label = _clean_florasense_label(label) if "_" in label else label
    search_url = "https://www.wikidata.org/w/api.php"

    try:
        eid = None

        # 1. Try resolving exact Wikidata QID from the Wikipedia page first
        try:
            wp_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "pageprops",
                    "titles": clean_label,
                    "redirects": 1,
                    "format": "json",
                },
                timeout=8,
                headers=WIKIMEDIA_HEADERS,
            )
            if wp_resp.status_code == 200:
                pages = wp_resp.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    qid = p.get("pageprops", {}).get("wikibase_item")
                    if qid:
                        eid = qid
                        break
        except Exception:
            pass

        # 2. Fall back to Wikidata wbsearchentities if not found via Wikipedia pageprops
        if not eid:
            response = requests.get(
                search_url,
                params={
                    "action": "wbsearchentities",
                    "search": clean_label,
                    "language": "en",
                    "format": "json",
                },
                timeout=10,
                headers=WIKIMEDIA_HEADERS,
            )
            response.raise_for_status()
            results = response.json().get("search", [])
            if results:
                eid = results[0].get("id")

        if not eid:
            return {"error": f"No Wikidata entity found for '{clean_label}'."}

        # 3. Single-request Wikidata SPARQL query for the complete taxonomy chain
        sparql_query = f"""
        SELECT ?rankLabel ?taxonName ?taxonLabel WHERE {{
          wd:{eid} (wdt:P1582|wdt:P31|wdt:P279)? / wdt:P171* ?taxon .
          ?taxon wdt:P105 ?rank .
          OPTIONAL {{ ?taxon wdt:P225 ?taxonName . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        try:
            sparql_resp = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql_query, "format": "json"},
                timeout=10,
                headers=WIKIMEDIA_HEADERS,
            )
            if sparql_resp.status_code == 200:
                bindings = (
                    sparql_resp.json()
                    .get("results", {})
                    .get("bindings", [])
                )
                taxonomy = {}
                for b in bindings:
                    rank_val = b.get("rankLabel", {}).get("value", "").strip()
                    sci_val = (
                        b.get("taxonName", {}).get("value")
                        or b.get("taxonLabel", {}).get("value")
                        or ""
                    ).strip()
                    if rank_val and sci_val and rank_val.lower() != "clade":
                        rank_key = rank_val.capitalize()
                        if rank_key not in taxonomy:
                            taxonomy[rank_key] = sci_val
                if taxonomy:
                    return taxonomy
        except Exception as sparql_err:
            print(f"SPARQL taxonomy fallback notice: {sparql_err}")

        # 4. Fallback: lightweight wbgetentities traversal if SPARQL is unavailable
        @lru_cache(None)
        def fetch(e):
            try:
                r = requests.get(
                    search_url,
                    params={
                        "action": "wbgetentities",
                        "ids": e,
                        "props": "claims|labels",
                        "languages": "en",
                        "format": "json",
                    },
                    timeout=8,
                    headers=WIKIMEDIA_HEADERS,
                )
                r.raise_for_status()
                return r.json().get("entities", {}).get(e, {})
            except Exception as err:
                print(f"Failed to fetch Wikidata entity {e}: {err}")
                return {}

        def find_taxon(e, depth=0):
            if depth > 4:
                return None
            ent = fetch(e)
            if not ent:
                return None
            claims = ent.get("claims", {})
            if "P225" in claims or "P171" in claims:
                return e
            for prop in ("P1582", "P31", "P279"):
                for claim in claims.get(prop, []):
                    try:
                        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                        if isinstance(val, dict) and val.get("id"):
                            res = find_taxon(val["id"], depth + 1)
                            if res:
                                return res
                    except Exception:
                        continue
            return None

        taxon_e = find_taxon(eid)
        if not taxon_e:
            return {"error": f"No taxonomic information found for '{clean_label}'."}

        taxonomy = {}
        visited = set()

        def collect(e, lvl=0):
            if lvl > 14 or e in visited:
                return
            visited.add(e)
            ent = fetch(e)
            if not ent:
                return
            claims = ent.get("claims", {})
            sci = ""
            try:
                p225 = claims.get("P225", [])
                if p225:
                    sci = p225[0].get("mainsnak", {}).get("datavalue", {}).get("value", "")
                if not sci:
                    sci = ent.get("labels", {}).get("en", {}).get("value", "")
            except Exception:
                sci = ""

            rank_id = ""
            try:
                p105 = claims.get("P105", [])
                if p105:
                    rank_id = p105[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")
            except Exception:
                rank_id = ""

            rank = _KNOWN_RANKS.get(rank_id, "")
            if sci and rank and rank.lower() != "clade":
                taxonomy[rank.capitalize()] = sci

            try:
                p171 = claims.get("P171", [])
                if p171:
                    parent = p171[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                    if parent:
                        collect(parent, lvl + 1)
            except Exception:
                pass

        collect(taxon_e)
        return taxonomy

    except requests.exceptions.RequestException as e:
        print(f"Wikidata request failed: {e}")
        return {"error": "Unable to connect to Wikidata."}
    except Exception as e:
        print(f"Wikidata error: {e}")
        return {"error": "Unable to retrieve taxonomy information."}


# ============================================================
# 6. Medicinal Uses from Wikipedia
# ============================================================

def get_medicinal_uses_from_wikipedia(
    title: str,
    target_lang="en"
):
    """
    Extract medicinal/traditional medicine/uses information
    from the Wikipedia page using a single lightweight API call.

    If Wikipedia is unavailable, return None.
    """

    clean_title = _clean_florasense_label(title) if "_" in title else title

    try:
        url = "https://en.wikipedia.org/w/api.php"

        response = requests.get(
            url,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": clean_title,
                "redirects": 1,
                "format": "json",
            },
            timeout=10,
            headers=WIKIMEDIA_HEADERS,
        )
        response.raise_for_status()

        pages = response.json().get("query", {}).get("pages", {})
        extract = ""
        for p in pages.values():
            if p.get("extract"):
                extract = p["extract"]
                break

        if not extract:
            return None

        priority_keywords = (
            "medicinal",
            "traditional medicine",
            "pharmacology",
            "ethnobotan",
            "folk medicine",
            "herbal",
            "therapeutic",
            "uses",
            "culinary",
            "phytochem",
            "chemical",
            "nutrition",
            "toxicity",
            "cultivation",
            "description",
        )

        # Split plain-text extract into sections by "== Heading =="
        sections = []
        current_heading = "lead"
        current_lines = []
        for line in extract.splitlines():
            m = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
            if m:
                sections.append((current_heading.lower(), "\n".join(current_lines).strip()))
                current_heading = m.group(1)
                current_lines = []
            else:
                current_lines.append(line)
        sections.append((current_heading.lower(), "\n".join(current_lines).strip()))

        target_text = ""
        for kw in priority_keywords:
            for heading, body in sections:
                if heading != "lead" and kw in heading and len(body.split()) >= 8:
                    target_text = body
                    break
            if target_text:
                break

        if not target_text:
            # Fall back to lead section
            for heading, body in sections:
                if heading == "lead" and body:
                    target_text = body
                    break

        if not target_text:
            return None

        bullet_points = []
        for paragraph in target_text.split("\n"):
            text = re.sub(r"\[\d+\]", "", paragraph.strip())
            if len(text.split()) < 5:
                continue
            sentences = re.split(r"(?<!\b[A-Z])\.\s+", text)
            for sentence in sentences:
                sentence = sentence.strip().strip(".")
                if len(sentence.split()) < 5:
                    continue
                if sentence.lower().startswith("there is insufficient"):
                    continue
                if target_lang != "en":
                    sentence = translate_text(sentence, target_lang)
                bullet_points.append(f"{sentence}.")
                if len(bullet_points) >= 5:
                    return bullet_points

        return bullet_points[:5] if bullet_points else None

    except requests.exceptions.RequestException as e:
        print(f"Wikipedia medicinal-use request failed: {e}")
        return None
    except Exception as e:
        print(f"Medicinal-use extraction error: {e}")
        return None


# Identification UI lives in app_pages/identify.py and reuses the functions above.
