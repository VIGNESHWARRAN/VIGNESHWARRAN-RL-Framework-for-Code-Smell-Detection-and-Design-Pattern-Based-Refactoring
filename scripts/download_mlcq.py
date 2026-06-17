"""
Download the MLCQ dataset from Zenodo and convert it to the format expected by SmellRL.

Usage:
  python scripts/download_mlcq.py

The script downloads the CSV from Zenodo record 3590102 and saves a unified
mlcq.csv under data/mlcq/.

If you already have the CSV, place it at data/mlcq/mlcq.csv and skip this script.
Expected columns (at minimum):
  smell_type (or kind/type)
  severity   (optional — 'none' rows become NoSmell)
  wmc, dit, noc, cbo, rfc, loc  (optional CK metrics)
"""

import os
import sys
import json
import logging
import requests
import zipfile
import io
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download_mlcq")

ZENODO_RECORD  = "3590102"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
OUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "data", "mlcq")

SMELL_NORM = {
    "GodClass": "GodClass", "godclass": "GodClass", "god_class": "GodClass",
    "FeatureEnvy": "FeatureEnvy", "featureenvy": "FeatureEnvy",
    "LongMethod": "LongMethod", "longmethod": "LongMethod",
    "DataClass": "DataClass", "dataclass": "DataClass",
}


def fetch_zenodo_files():
    log.info(f"Fetching file list from Zenodo record {ZENODO_RECORD}...")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    meta = resp.json()
    return meta.get("files", [])


def download_file(url: str, filename: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        log.info(f"  Already exists: {dest}")
        return dest
    log.info(f"  Downloading {filename} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"  Saved to {dest}")
    return dest


def process_zenodo_csv(path: str) -> pd.DataFrame:
    """
    Parse Zenodo MLCQ CSV into a unified DataFrame.
    Handles multiple possible formats.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    log.info(f"Loaded {len(df)} rows from {path}. Columns: {list(df.columns)}")

    # Detect smell column
    for col in ["kind", "smell_type", "type", "codesmell"]:
        if col in df.columns:
            df["smell_type"] = df[col]
            break

    if "smell_type" not in df.columns:
        raise ValueError(f"Cannot find smell-type column in {path}")

    # Normalise severity=none → NoSmell
    if "severity" in df.columns:
        df.loc[df["severity"].str.strip().str.lower() == "none", "smell_type"] = "NoSmell"

    # Normalise smell names
    df["smell_type"] = df["smell_type"].map(
        lambda x: SMELL_NORM.get(str(x).strip(), str(x).strip())
    )

    # Keep only known smells
    known = set(SMELL_NORM.values()) | {"NoSmell"}
    df    = df[df["smell_type"].isin(known)].copy()
    log.info(f"After filtering: {len(df)} rows. Distribution:\n{df['smell_type'].value_counts().to_string()}")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "mlcq.csv")

    if os.path.exists(out_csv):
        log.info(f"mlcq.csv already exists at {out_csv}. Delete it to re-download.")
        return

    try:
        files = fetch_zenodo_files()
    except Exception as e:
        log.error(f"Failed to fetch Zenodo record: {e}")
        log.info("Manual download instructions:")
        log.info("  1. Go to: https://zenodo.org/record/3590102")
        log.info("  2. Download the CSV file(s)")
        log.info("  3. Place the CSV at: data/mlcq/mlcq.csv")
        log.info("  4. Ensure it has columns: smell_type (or kind), severity")
        log.info("  5. Optionally: wmc, dit, noc, cbo, rfc, loc")
        sys.exit(1)

    csv_files = [f for f in files if f["key"].endswith(".csv")]
    zip_files = [f for f in files if f["key"].endswith(".zip")]

    all_dfs = []
    raw_dir = os.path.join(OUT_DIR, "raw")

    # Download and process CSV files
    for fmeta in csv_files:
        local = download_file(fmeta["links"]["self"], fmeta["key"], raw_dir)
        try:
            df = process_zenodo_csv(local)
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"Skipping {fmeta['key']}: {e}")

    # Download and extract ZIP files
    for fmeta in zip_files:
        local = download_file(fmeta["links"]["self"], fmeta["key"], raw_dir)
        with zipfile.ZipFile(local) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        content = f.read()
                    tmp_path = os.path.join(raw_dir, os.path.basename(name))
                    with open(tmp_path, "wb") as tf:
                        tf.write(content)
                    try:
                        df = process_zenodo_csv(tmp_path)
                        all_dfs.append(df)
                    except Exception as e:
                        log.warning(f"Skipping {name}: {e}")

    if not all_dfs:
        log.error("No CSV data could be processed from the download. Use manual download.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    combined.to_csv(out_csv, index=False)
    log.info(f"Combined MLCQ CSV saved: {out_csv} ({len(combined)} instances)")


if __name__ == "__main__":
    main()
