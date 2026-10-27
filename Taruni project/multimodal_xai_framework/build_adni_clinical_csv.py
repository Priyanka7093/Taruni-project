"""
Builds the final clinical_data_adni.csv for the Multi-Modal XAI Framework.

This ADNI download uses the newer CDISC/SDTM-style ADSL.rda table, which
already contains ALL clinical/cognitive fields in one place (age, sex,
education, diagnosis, MMSE, CDR, MoCA, FAQ).

NOTE: eTIV/nWBV (brain volume) are DROPPED in this version. The UCSFFSX51
table in this ADNI export only contains ~165 individual FreeSurfer region
codes (e.g. ST102CV, ST103CV...) rather than one clean "whole brain volume"
/ "intracranial volume" summary column like OASIS provided. Computing those
correctly would require summing many specific region codes, which risks
silent errors - so instead we rely on the real cognitive/clinical features
(MMSE, CDR, MoCA, FAQ, Age, Gender, Educ), which carry most of the predictive
signal for CN/MCI/AD classification anyway.

OUTPUT COLUMNS:
    RID, Age, Gender, MMSE, CDR, Educ, MOCA, FAQ, Group

USAGE:
    python build_adni_clinical_csv.py --raw_dir data/raw/adni_tables --output data/clinical_data_adni.csv
"""

import argparse
import os
import pyreadr
import pandas as pd


def load_rda(path):
    result = pyreadr.read_r(path)
    key = list(result.keys())[0]
    return result[key]


def main(raw_dir, output_path):
    print("=" * 60)
    print("BUILDING ADNI CLINICAL DATASET (ADSL only)")
    print("=" * 60)

    print("\n[1/1] Loading ADSL.rda ...")
    adsl = load_rda(os.path.join(raw_dir, "ADSL.rda"))
    print(f"      {len(adsl)} subjects total")
    print(f"      Unique DX values: {sorted(adsl['DX'].dropna().unique().tolist())}")
    print(f"      Unique SEX values: {sorted(adsl['SEX'].dropna().unique().tolist())}")

    adsl = adsl.drop_duplicates(subset="SUBJID", keep="first").copy()
    print(f"      {len(adsl)} unique subjects after SUBJID dedup")

    core = pd.DataFrame()
    core["RID"] = adsl["SUBJID"]
    core["Age"] = adsl["AGE"]
    core["Gender"] = adsl["SEX"]
    core["Educ"] = adsl["EDUC"]
    core["MMSE"] = adsl["MMSCORE"]
    core["CDR"] = adsl["CDRSB"] if "CDRSB" in adsl.columns else adsl["CDGLOBAL"]
    core["MOCA"] = adsl.get("MOCA")
    core["FAQ"] = adsl.get("FAQTOTAL")
    core["Group_raw"] = adsl["DX"]

    sex_map = {"M": 1, "F": 0, "Male": 1, "Female": 0, 1: 1, 2: 0}
    core["Gender"] = core["Gender"].map(sex_map)

    # Includes 'DEM' - the label actually used in this ADNI export for dementia/AD
    dx_map = {
        "CN": 0, "Dementia": 2, "DEM": 2, "MCI": 1, "AD": 2,
        "EMCI": 1, "LMCI": 1, "SMC": 0,
    }
    core["Group"] = core["Group_raw"].map(dx_map)
    unmapped = core[core["Group"].isna() & core["Group_raw"].notna()]["Group_raw"].unique()
    if len(unmapped) > 0:
        print(f"      [!] WARNING: unmapped DX values found: {unmapped} - these rows will be dropped")
    core = core.drop(columns=["Group_raw"])

    final_cols = ["RID", "Age", "Gender", "MMSE", "CDR", "Educ", "MOCA", "FAQ", "Group"]
    core = core[[c for c in final_cols if c in core.columns]]

    before = len(core)
    core = core.dropna(subset=["Age", "Gender", "MMSE", "CDR", "Educ", "Group"])
    core["Group"] = core["Group"].astype(int)
    print(f"\n[CLEAN] Dropped {before - len(core)} rows with missing core values -> {len(core)} final rows")

    print("\nClass distribution (Group: 0=CN, 1=MCI, 2=AD):")
    print(core["Group"].value_counts().sort_index())

    core.to_csv(output_path, index=False)
    print(f"\n[+] Saved final clinical dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.raw_dir, args.output)
