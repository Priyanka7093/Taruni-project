"""
ADNI -> Multi-Modal XAI Framework clinical_data.csv converter
================================================================
Converts a downloaded ADNIMERGE.csv file into the 7-feature clinical
schema your project expects (ASF dropped - no ADNI equivalent exists).

USAGE:
    python adni_preprocessor.py --input ADNIMERGE.csv --output data/clinical_data_adni.csv

OUTPUT COLUMNS:
    Age, Gender, MMSE, CDR, Educ, eTIV, nWBV, Group
    (Image_Path is left blank here - filled in later once MRI scans
     are matched to subjects in Step 2)
"""

import argparse
import pandas as pd
import numpy as np

# ---- 1. Mapping: ADNIMERGE column -> framework column ----
# NOTE: ASF has been dropped (OASIS-pipeline-specific, no ADNI equivalent)
COLUMN_MAP = {
    "RID": "RID",          # subject identifier, used for split - not a model feature
    "AGE": "Age",
    "PTGENDER": "Gender",     # 'Male' -> 1, 'Female' -> 0
    "MMSE": "MMSE",
    "CDRSB": "CDR",          # ADNI's CDR-Sum-of-Boxes (closest available equivalent)
    "PTEDUCAT": "Educ",
    "ICV": "eTIV",         # Intracranial volume (ADNI's eTIV equivalent)
    "WholeBrain": "WholeBrain",  # used only to compute nWBV, then dropped
    "DX": "Group",         # CN / MCI / Dementia -> 0 / 1 / 2 (real diagnosis label, not CDR-derived)
}

DX_MAP = {"CN": 0, "MCI": 1, "Dementia": 2}


def convert(input_path: str, output_path: str):
    print(f"[1/6] Reading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"      Loaded {len(df)} rows (all visits, all subjects)")

    missing_cols = [c for c in COLUMN_MAP if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"These expected ADNIMERGE columns were not found: {missing_cols}\n"
            f"Available columns include: {list(df.columns)[:30]} ..."
        )

    # ---- 2. Keep only baseline visit per subject (subject-level, no leakage) ----
    print("[2/6] Filtering to baseline visit per subject (VISCODE == 'bl') ...")
    if "VISCODE" in df.columns:
        df = df[df["VISCODE"] == "bl"].copy()
    print(f"      {len(df)} rows remain after baseline filter")

    # ---- 3. Drop rows missing a diagnosis or RID ----
    df = df.dropna(subset=["DX", "RID"])
    print(f"[3/6] {len(df)} rows remain after dropping missing DX/RID")

    # ---- 4. One row per subject (RID) - dedup safeguard ----
    df = df.drop_duplicates(subset="RID", keep="first")
    print(f"[4/6] {len(df)} unique subjects after RID dedup")

    # ---- 5. Build output dataframe ----
    print("[5/6] Mapping columns and computing nWBV ...")
    out = pd.DataFrame()
    out["RID"] = df["RID"]
    out["Age"] = df["AGE"]
    out["Gender"] = df["PTGENDER"].map({"Male": 1, "Female": 0})
    out["MMSE"] = df["MMSE"]
    out["CDR"] = df["CDRSB"]
    out["Educ"] = df["PTEDUCAT"]
    out["eTIV"] = df["ICV"]
    # nWBV = WholeBrain volume / Intracranial volume (same meaning as OASIS's nWBV)
    out["nWBV"] = df["WholeBrain"] / df["ICV"]
    out["Group"] = df["DX"].map(DX_MAP)

    # ---- 6. Drop rows with any missing values in final feature set ----
    before = len(out)
    out = out.dropna(subset=["Age", "Gender", "MMSE", "CDR", "Educ", "eTIV", "nWBV", "Group"])
    out["Group"] = out["Group"].astype(int)
    print(f"[6/6] Dropped {before - len(out)} rows with missing values -> {len(out)} final rows")

    print("\nClass distribution (Group: 0=CN, 1=MCI, 2=AD):")
    print(out["Group"].value_counts().sort_index())

    out.to_csv(output_path, index=False)
    print(f"\n[+] Saved cleaned clinical data to: {output_path}")
    print("    NOTE: 'Image_Path' column is NOT included yet.")
    print("    That gets added in Step 2, once MRI scans are matched to these RIDs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADNIMERGE.csv to framework clinical schema")
    parser.add_argument("--input", required=True, help="Path to downloaded ADNIMERGE.csv")
    parser.add_argument("--output", required=True, help="Path to save converted clinical_data_adni.csv")
    args = parser.parse_args()
    convert(args.input, args.output)
