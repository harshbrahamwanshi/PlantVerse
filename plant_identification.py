# ============================================================
# PlantVerse - Plant Identification
# ============================================================

import streamlit as st
import requests
from PIL import Image
from transformers import pipeline
from functools import lru_cache
from bs4 import BeautifulSoup
import re


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
# 3. Plant Classifier
# ============================================================

@st.cache_resource
def load_classifier():
    """
    Load the FloraSense image classification model.
    """

    return pipeline(
        "image-classification",
        model="Sisigoks/FloraSense"
    )


def predict_species(image: Image.Image):
    """
    Predict plant species from uploaded image.
    """

    classifier = load_classifier()

    if image.mode != "RGB":
        image = image.convert("RGB")

    preds = classifier(image)

    if not preds:
        return "Unknown", 0.0

    return preds[0]["label"], preds[0]["score"]


# ============================================================
# 4. Wikipedia Search
# ============================================================

@st.cache_data
def get_wikipedia_title(name: str):
    """
    Search Wikipedia for the predicted plant.

    If Wikipedia is unavailable, return the original
    model prediction instead of crashing.
    """

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "utf8": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0 (educational project)"
            }
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
            return name

        data = response.json()

        results = data.get(
            "query",
            {}
        ).get(
            "search",
            []
        )

        if results:
            return results[0].get("title", name)

        return name

    except requests.exceptions.RequestException as e:
        print(f"Wikipedia request failed: {e}")
        return name

    except (ValueError, KeyError, TypeError) as e:
        print(f"Wikipedia returned invalid JSON/data: {e}")
        return name

    except Exception as e:
        print(f"Wikipedia error: {e}")
        return name


# ============================================================
# 5. Wikidata Taxonomy
# ============================================================

@st.cache_data
def get_taxonomy_from_wikidata(label: str):
    """
    Get plant taxonomy from Wikidata.

    If Wikidata fails, return an error dictionary instead
    of crashing the application.
    """

    search_url = "https://www.wikidata.org/w/api.php"

    search_params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json"
    }

    try:

        # ----------------------------------------------------
        # Search Wikidata
        # ----------------------------------------------------

        response = requests.get(
            search_url,
            params=search_params,
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        response.raise_for_status()

        if "json" not in response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return {
                "error": "Wikidata returned a non-JSON response."
            }

        search = response.json()

        results = search.get("search", [])

        if not results:
            return {
                "error": f"No Wikidata entity found for '{label}'."
            }

        eid = results[0].get("id")

        if not eid:
            return {
                "error": f"No valid Wikidata ID found for '{label}'."
            }


        # ----------------------------------------------------
        # Fetch Wikidata Entity
        # ----------------------------------------------------

        @lru_cache(None)
        def fetch(e):

            try:

                entity_url = (
                    f"https://www.wikidata.org/"
                    f"wiki/Special:EntityData/{e}.json"
                )

                response = requests.get(
                    entity_url,
                    timeout=10,
                    headers={
                        "User-Agent": "PlantVerse/1.0"
                    }
                )

                response.raise_for_status()

                if "json" not in response.headers.get(
                    "Content-Type",
                    ""
                ).lower():
                    print(
                        f"Wikidata entity {e} "
                        f"returned non-JSON response."
                    )
                    return {}

                data = response.json()

                return data.get(
                    "entities",
                    {}
                ).get(
                    e,
                    {}
                )

            except requests.exceptions.RequestException as err:
                print(
                    f"Failed to fetch Wikidata entity "
                    f"{e}: {err}"
                )
                return {}

            except (ValueError, KeyError, TypeError) as err:
                print(
                    f"Invalid Wikidata data for "
                    f"{e}: {err}"
                )
                return {}

            except Exception as err:
                print(
                    f"Unexpected Wikidata error "
                    f"for {e}: {err}"
                )
                return {}


        # ----------------------------------------------------
        # Find Taxonomic Entity
        # ----------------------------------------------------

        def find_taxon(e, depth=0):

            if depth > 5:
                return None

            ent = fetch(e)

            if not ent:
                return None

            claims = ent.get("claims", {})

            # P225 = taxon scientific name
            # P171 = parent taxon

            if (
                "P225" in claims
                and "P171" in claims
            ):
                return e

            # Try instance of / subclass of
            for prop in ("P31", "P279"):

                for claim in claims.get(prop, []):

                    try:

                        mainsnak = claim.get(
                            "mainsnak",
                            {}
                        )

                        datavalue = mainsnak.get(
                            "datavalue",
                            {}
                        )

                        value = datavalue.get(
                            "value",
                            {}
                        )

                        if isinstance(value, dict):
                            nid = value.get("id")

                            if nid:
                                result = find_taxon(
                                    nid,
                                    depth + 1
                                )

                                if result:
                                    return result

                    except Exception:
                        continue

            return None


        taxon_e = find_taxon(eid)

        if not taxon_e:

            return {
                "error": (
                    f"No taxonomic information "
                    f"found for '{label}'."
                )
            }


        # ----------------------------------------------------
        # Collect Taxonomy
        # ----------------------------------------------------

        taxonomy = {}

        def collect(e, lvl=0):

            if lvl > 20:
                return

            ent = fetch(e)

            if not ent:
                return

            claims = ent.get(
                "claims",
                {}
            )

            # ------------------------------------------------
            # Scientific Name - P225
            # ------------------------------------------------

            sci = ""

            try:

                p225 = claims.get(
                    "P225",
                    []
                )

                if p225:

                    sci = (
                        p225[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", "")
                    )

            except Exception:
                sci = ""


            # ------------------------------------------------
            # Taxonomic Rank - P105
            # ------------------------------------------------

            rank_id = ""

            try:

                p105 = claims.get(
                    "P105",
                    []
                )

                if p105:

                    rank_id = (
                        p105[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("id", "")
                    )

            except Exception:
                rank_id = ""


            # ------------------------------------------------
            # Get Rank Name
            # ------------------------------------------------

            rank = ""

            if rank_id:

                rank_entity = fetch(rank_id)

                rank = (
                    rank_entity
                    .get("labels", {})
                    .get("en", {})
                    .get("value", "")
                )


            if sci:

                taxonomy[
                    rank.capitalize()
                    if rank
                    else f"Rank{lvl}"
                ] = sci


            # ------------------------------------------------
            # Parent Taxon - P171
            # ------------------------------------------------

            parent = None

            try:

                p171 = claims.get(
                    "P171",
                    []
                )

                if p171:

                    parent = (
                        p171[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("id")
                    )

            except Exception:
                parent = None


            if parent:
                collect(
                    parent,
                    lvl + 1
                )


        collect(taxon_e)

        return taxonomy


    except requests.exceptions.RequestException as e:

        print(
            f"Wikidata request failed: {e}"
        )

        return {
            "error": "Unable to connect to Wikidata."
        }


    except (ValueError, KeyError, TypeError) as e:

        print(
            f"Wikidata returned invalid data: {e}"
        )

        return {
            "error": "Wikidata returned invalid data."
        }


    except Exception as e:

        print(
            f"Wikidata error: {e}"
        )

        return {
            "error": "Unable to retrieve taxonomy information."
        }


# ============================================================
# 6. Medicinal Uses from Wikipedia
# ============================================================

def get_medicinal_uses_from_wikipedia(
    title: str,
    target_lang="en"
):
    """
    Extract medicinal/traditional medicine information
    from the Wikipedia page.

    If Wikipedia is unavailable, return None.
    """

    try:

        url = "https://en.wikipedia.org/w/api.php"

        # ----------------------------------------------------
        # Get Wikipedia sections
        # ----------------------------------------------------

        sections_response = requests.get(
            url,
            params={
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json"
            },
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        sections_response.raise_for_status()

        if "json" not in sections_response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return None

        sections_data = sections_response.json()

        sections = (
            sections_data
            .get("parse", {})
            .get("sections", [])
        )

        # ----------------------------------------------------
        # Find medicinal section
        # ----------------------------------------------------

        section_index = next(
            (
                sec.get("index")
                for sec in sections
                if (
                    "medicinal"
                    in sec.get(
                        "line",
                        ""
                    ).lower()
                    or
                    "traditional medicine"
                    in sec.get(
                        "line",
                        ""
                    ).lower()
                )
            ),
            None
        )

        if not section_index:
            return None


        # ----------------------------------------------------
        # Get section text
        # ----------------------------------------------------

        text_response = requests.get(
            url,
            params={
                "action": "parse",
                "page": title,
                "format": "json",
                "prop": "text",
                "section": section_index
            },
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        text_response.raise_for_status()

        if "json" not in text_response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return None

        text_data = text_response.json()

        html = (
            text_data
            .get("parse", {})
            .get("text", {})
            .get("*", "")
        )

        if not html:
            return None


        # ----------------------------------------------------
        # Parse HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        bullet_points = []


        # ----------------------------------------------------
        # Extract useful sentences
        # ----------------------------------------------------

        for paragraph in paragraphs:

            text = paragraph.get_text().strip()

            # Remove citation numbers
            text = re.sub(
                r"\[\d+\]",
                "",
                text
            )

            if len(text.split()) < 5:
                continue


            sentences = re.split(
                r"\.\s+",
                text
            )

            for sentence in sentences:

                sentence = (
                    sentence
                    .strip()
                    .strip(".")
                )

                if len(sentence.split()) < 5:
                    continue

                if sentence.lower().startswith(
                    "there is insufficient"
                ):
                    continue


                # Translate if necessary
                if target_lang != "en":

                    sentence = translate_text(
                        sentence,
                        target_lang
                    )


                bullet_points.append(
                    f"{sentence}."
                )


        if bullet_points:

            return bullet_points[:5]

        return None


    except requests.exceptions.RequestException as e:

        print(
            f"Wikipedia medicinal-use request failed: {e}"
        )

        return None


    except (ValueError, KeyError, TypeError) as e:

        print(
            f"Wikipedia medicinal-use data error: {e}"
        )

        return None


    except Exception as e:

        print(
            f"Medicinal-use extraction error: {e}"
        )

        return None


# Identification UI lives in app_pages/identify.py and reuses the functions above.
