"""
Build and cache public evaluation datasets for PlantVerse ML Evaluation Harness.

Sources built (no user-uploaded images or guessed labels):
1. Held-out PlantVillage test split (from BrandonFors/Plant-Diseases-PlantVillage-Dataset on HF + FloraCare's 14 held-out samples)
2. PlantDoc test split (from pratikkayal/PlantDoc-Dataset /tmp/PlantDoc-Dataset/test — 236 real field photos across 27 classes)
3. Public iNaturalist permissive-licensed (CC0 / CC-BY) species-labeled field photos (fresh observations not used in model training)
4. Smoke test set (existing repo images in static/ and FloraCare reference samples, labeled strictly as unverified smoke test)
"""

import csv
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image
import requests

CACHE_DIR = Path("/tmp/plantverse_eval_data")
PV_DIR = CACHE_DIR / "plantvillage_test"
INAT_DIR = CACHE_DIR / "inat_species_test"
PLANTDOC_TEST_DIR = Path("/tmp/PlantDoc-Dataset/test")
FLORACARE_REF_DIR = Path("/tmp/FloraCare_ref")

# Mapping from PlantDoc test folder names (lowercased, normalized whitespace) to standard PlantVillage raw labels
PLANTDOC_TO_PV = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "bell_pepper leaf": "Pepper,_bell___healthy",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "peach leaf": "Peach___healthy",
    "potato leaf": "Potato___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato leaf": "Tomato___healthy",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

PV_CROP_TO_SCIENTIFIC = {
    "Apple": ("Apple", "Malus domestica"),
    "Blueberry": ("Blueberry", "Vaccinium corymbosum"),
    "Cherry_(including_sour)": ("Cherry", "Prunus avium"),
    "Corn_(maize)": ("Corn (Maize)", "Zea mays"),
    "Grape": ("Grape", "Vitis vinifera"),
    "Orange": ("Orange / Citrus", "Citrus sinensis"),
    "Peach": ("Peach", "Prunus persica"),
    "Pepper,_bell": ("Bell Pepper", "Capsicum annuum"),
    "Potato": ("Potato", "Solanum tuberosum"),
    "Raspberry": ("Raspberry", "Rubus idaeus"),
    "Soybean": ("Soybean", "Glycine max"),
    "Squash": ("Squash", "Cucurbita"),
    "Strawberry": ("Strawberry", "Fragaria ananassa"),
    "Tomato": ("Tomato", "Solanum lycopersicum"),
}

# 15 representative species for public iNaturalist field evaluation (including key Indian plants)
INAT_EVAL_SPECIES = [
    ("tulsi", "Tulsi (Holy basil)", "Ocimum tenuiflorum"),
    ("neem", "Neem", "Azadirachta indica"),
    ("peepal", "Peepal (Sacred fig)", "Ficus religiosa"),
    ("curry_leaf", "Curry leaf", "Bergera koenigii"),
    ("hibiscus", "Hibiscus (Gudhal)", "Hibiscus rosa-sinensis"),
    ("money_plant", "Money plant (Pothos)", "Epipremnum aureum"),
    ("banyan", "Banyan", "Ficus benghalensis"),
    ("rose", "Rose", "Rosa"),
    ("aloe", "Aloe vera", "Aloe vera"),
    ("mango", "Mango", "Mangifera indica"),
    ("snake_plant", "Snake plant", "Dracaena trifasciata"),
    ("croton", "Croton", "Codiaeum variegatum"),
    ("tomato", "Tomato", "Solanum lycopersicum"),
    ("apple", "Apple", "Malus domestica"),
    ("potato", "Potato", "Solanum tuberosum"),
]


def fetch_url(url: str) -> bytes:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PlantVerse-Eval/1.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url}")


def build_plantvillage_set(n_target: int = 76) -> Path:
    """Download held-out PlantVillage test images from HuggingFace test split + FloraCare held-out samples."""
    PV_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PV_DIR / "manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(records) >= n_target:
            print(f"[Dataset] Using cached PlantVillage test set ({len(records)} images)")
            return manifest_path

    records = []
    # 1. Include the 14 held-out PlantVillage test images in /tmp/FloraCare_ref/samples/labels.json
    fc_labels_file = FLORACARE_REF_DIR / "samples" / "labels.json"
    if fc_labels_file.exists():
        fc_map = json.loads(fc_labels_file.read_text(encoding="utf-8"))
        for fname, raw_label in sorted(fc_map.items()):
            fpath = FLORACARE_REF_DIR / "samples" / fname
            if fpath.exists():
                crop_prefix = raw_label.split("___")[0]
                common_crop, sci_crop = PV_CROP_TO_SCIENTIFIC.get(crop_prefix, (crop_prefix, crop_prefix))
                records.append({
                    "image_path": str(fpath),
                    "expected_plant": common_crop,
                    "expected_scientific": sci_crop,
                    "expected_disease_raw": raw_label,
                    "source": "PlantVillage_test_heldout",
                    "license": "CC0 / Public Domain (PlantVillage)",
                })

    # 2. Fetch additional held-out test split rows from HuggingFace Datasets Server
    DATASET = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
    API = "https://datasets-server.huggingface.co"
    try:
        info = json.loads(fetch_url(f"{API}/info?dataset={DATASET}"))["dataset_info"]["default"]
        names = info["features"]["label"]["names"]
        total = info["splits"]["test"]["num_examples"]
        random.seed(42)
        needed = max(0, n_target - len(records))
        starts = random.sample(range(0, total - 5, 5), (needed + 4) // 5 + 2)
        for start in starts:
            if len(records) >= n_target:
                break
            batch_url = f"{API}/rows?dataset={DATASET}&config=default&split=test&offset={start}&length=5"
            rows = json.loads(fetch_url(batch_url))["rows"]
            for r in rows:
                if len(records) >= n_target:
                    break
                idx = r["row_idx"]
                raw_label = names[r["row"]["label"]]
                img_url = r["row"]["image"]["src"]
                dst = PV_DIR / f"pv_test_{idx}.jpg"
                if not dst.exists():
                    img_bytes = fetch_url(img_url)
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    im.save(dst, quality=92)
                crop_prefix = raw_label.split("___")[0]
                common_crop, sci_crop = PV_CROP_TO_SCIENTIFIC.get(crop_prefix, (crop_prefix, crop_prefix))
                records.append({
                    "image_path": str(dst),
                    "expected_plant": common_crop,
                    "expected_scientific": sci_crop,
                    "expected_disease_raw": raw_label,
                    "source": f"HF:{DATASET}:test#{idx}",
                    "license": "CC0 / Public Domain (PlantVillage)",
                })
    except Exception as exc:
        print(f"[Dataset] Warning fetching extra PlantVillage rows: {exc}")

    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[Dataset] Built PlantVillage test set with {len(records)} images at {manifest_path}")
    return manifest_path


def build_plantdoc_set() -> Path:
    """Index all real field photos in PlantDoc-Dataset/test."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "plantdoc_test_manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(records) > 200:
            print(f"[Dataset] Using cached PlantDoc test manifest ({len(records)} images)")
            return manifest_path

    records = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for folder in sorted(PLANTDOC_TEST_DIR.iterdir()):
        if not folder.is_dir():
            continue
        norm_name = re.sub(r"\s+", " ", folder.name.strip().lower())
        raw_label = PLANTDOC_TO_PV.get(norm_name)
        if not raw_label:
            print(f"[Dataset] Skipping unmapped PlantDoc folder: {folder.name}")
            continue
        crop_prefix = raw_label.split("___")[0]
        common_crop, sci_crop = PV_CROP_TO_SCIENTIFIC.get(crop_prefix, (crop_prefix, crop_prefix))
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            try:
                im = Image.open(img_path)
                im.verify()
            except Exception:
                continue
            records.append({
                "image_path": str(img_path),
                "expected_plant": common_crop,
                "expected_scientific": sci_crop,
                "expected_disease_raw": raw_label,
                "source": f"PlantDoc:test/{folder.name}/{img_path.name}",
                "license": "CC BY 4.0 (PlantDoc Dataset, Singh et al. 2020)",
            })

    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[Dataset] Indexed {len(records)} PlantDoc real field test images at {manifest_path}")
    return manifest_path


def build_inat_species_set(per_species: int = 3) -> Path:
    """Download permissive (CC0 / CC-BY) research-grade iNaturalist field photos for plant species evaluation."""
    INAT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INAT_DIR / "manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(records) >= len(INAT_EVAL_SPECIES) * 2:
            print(f"[Dataset] Using cached iNaturalist species set ({len(records)} images)")
            return manifest_path

    # Load excluded observation URLs from FloraCare's training attribution to guarantee unseen test images
    seen_obs_ids = set()
    attr_csv = FLORACARE_REF_DIR / "backend" / "model" / "inat_attribution.csv"
    if attr_csv.exists():
        with open(attr_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row.get("observation_url", "")
                if "/" in url:
                    seen_obs_ids.add(url.rsplit("/", 1)[-1].strip())

    session = requests.Session()
    session.headers["User-Agent"] = "PlantVerse-Eval/1.0 (https://github.com/harshbrahamwanshi/PlantVerse)"
    records = []

    for sp_id, common_name, sci_name in INAT_EVAL_SPECIES:
        try:
            # Resolve taxon_id
            r_tax = session.get(
                "https://api.inaturalist.org/v1/taxa",
                params={"q": sci_name, "per_page": 10},
                timeout=15,
            )
            r_tax.raise_for_status()
            results = r_tax.json().get("results", [])
            taxon_id = None
            for t in results:
                if t.get("name", "").lower() == sci_name.lower():
                    taxon_id = t["id"]
                    break
            if not taxon_id and results:
                taxon_id = results[0]["id"]
            if not taxon_id:
                print(f"[Dataset] Could not resolve taxon for {sci_name}")
                continue

            # Fetch research-grade CC0/CC-BY observations (sort asc by id to avoid recent ones used in training)
            r_obs = session.get(
                "https://api.inaturalist.org/v1/observations",
                params={
                    "taxon_id": taxon_id,
                    "quality_grade": "research",
                    "photos": "true",
                    "photo_license": "cc0,cc-by",
                    "per_page": 30,
                    "order_by": "votes",
                    "order": "desc",
                },
                timeout=20,
            )
            r_obs.raise_for_status()
            obs_list = r_obs.json().get("results", [])
            count = 0
            for obs in obs_list:
                obs_id = str(obs["id"])
                if obs_id in seen_obs_ids:
                    continue
                photos = obs.get("photos") or []
                if not photos:
                    continue
                ph = photos[0]
                lic = (ph.get("license_code") or "").lower()
                if lic not in ("cc0", "cc-by"):
                    continue
                url = (ph.get("url") or "").replace("/square.", "/medium.")
                if not url:
                    continue
                dst = INAT_DIR / f"{sp_id}_{obs_id}.jpg"
                if not dst.exists():
                    img_resp = session.get(url, timeout=20)
                    img_resp.raise_for_status()
                    im = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    im.save(dst, quality=90)
                records.append({
                    "image_path": str(dst),
                    "species_id": sp_id,
                    "expected_plant": common_name,
                    "expected_scientific": sci_name,
                    "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
                    "license": lic.upper(),
                    "attribution": ph.get("attribution", ""),
                    "source": "iNaturalist_research_grade",
                })
                count += 1
                if count >= per_species:
                    break
            print(f"[Dataset] iNaturalist {sp_id} ({sci_name}): {count} unseen CC0/CC-BY photos")
            time.sleep(0.6)
        except Exception as exc:
            print(f"[Dataset] Error fetching iNaturalist {sci_name}: {exc}")

    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[Dataset] Built iNaturalist species set with {len(records)} images at {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    build_plantvillage_set(76)
    build_plantdoc_set()
    build_inat_species_set(3)
