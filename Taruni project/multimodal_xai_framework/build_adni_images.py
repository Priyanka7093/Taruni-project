"""
Converts downloaded ADNI MRI DICOM scans into PNG images (matching the
format your project's dataset.py expects), matches each scan to a
subject's clinical row by RID, and produces a final merged CSV with
an Image_Path column ready for multimodal training.

How subject ID matching works:
    ADNI image folders are named like "031_S_0830" (Site_S_RID).
    The numeric RID at the end ("0830" -> 830) matches the "RID"
    column already in your clinical_data_adni.csv.

For each subject folder, we pick the MIDDLE slice of their MPRAGE
scan (most informative axial/sagittal cross-section of the brain)
and save it as a 224x224 RGB PNG, named "<RID>.png".

USAGE:
    python build_adni_images.py --scans_dir "C:/Users/10844369/Downloads/MyMRIScans/ADNI" --clinical_csv data/clinical_data_adni.csv --images_out data/images_adni --output data/clinical_data_adni_full.csv
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import pydicom
from PIL import Image


def extract_rid_from_folder(folder_name: str):
    """'031_S_0830' -> 830"""
    parts = folder_name.split("_S_")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def find_dicom_series_dir(subject_dir: str):
    """
    Subject folder structure: <subject>/MPRAGE/<timestamp>/<series_id>/*.dcm
    Returns the folder containing the most .dcm files (the main series),
    in case there are multiple visits/series for a subject.
    """
    dcm_dirs = {}
    for root, dirs, files in os.walk(subject_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if dcm_files:
            dcm_dirs[root] = dcm_files
    if not dcm_dirs:
        return None, []
    # Pick the series with the most slices (most complete scan)
    best_dir = max(dcm_dirs, key=lambda d: len(dcm_dirs[d]))
    return best_dir, sorted(dcm_dirs[best_dir])


def dicom_slice_to_png(dicom_dir: str, dcm_files: list, out_path: str, image_size=(224, 224)):
    """Pick the middle slice, normalize pixel values, save as RGB PNG."""
    mid_idx = len(dcm_files) // 2
    mid_file = os.path.join(dicom_dir, dcm_files[mid_idx])

    ds = pydicom.dcmread(mid_file)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255
    pixel_array -= pixel_array.min()
    if pixel_array.max() > 0:
        pixel_array = pixel_array / pixel_array.max() * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    img = Image.fromarray(pixel_array).convert("L")
    img = img.resize(image_size)
    rgb_img = Image.merge("RGB", (img, img, img))
    rgb_img.save(out_path)


def main(scans_dir, clinical_csv, images_out, output_csv):
    os.makedirs(images_out, exist_ok=True)

    print("=" * 60)
    print("MATCHING MRI SCANS TO CLINICAL DATA")
    print("=" * 60)

    clinical = pd.read_csv(clinical_csv)
    print(f"\n[1/3] Loaded clinical data: {len(clinical)} subjects")

    subject_dirs = [d for d in glob.glob(os.path.join(scans_dir, "*_S_*")) if os.path.isdir(d)]
    print(f"[2/3] Found {len(subject_dirs)} subject folders in MRI scans directory")

    matched_rows = []
    skipped_no_rid = 0
    skipped_no_dicom = 0
    skipped_no_clinical = 0

    for subject_dir in subject_dirs:
        folder_name = os.path.basename(subject_dir)
        rid = extract_rid_from_folder(folder_name)
        if rid is None:
            skipped_no_rid += 1
            continue

        clinical_row = clinical[clinical["RID"] == rid]
        if len(clinical_row) == 0:
            skipped_no_clinical += 1
            continue

        dicom_dir, dcm_files = find_dicom_series_dir(subject_dir)
        if dicom_dir is None or len(dcm_files) == 0:
            skipped_no_dicom += 1
            continue

        png_filename = f"{rid}.png"
        png_path = os.path.join(images_out, png_filename)
        try:
            dicom_slice_to_png(dicom_dir, dcm_files, png_path)
        except Exception as e:
            print(f"      [!] Failed to convert RID {rid}: {e}")
            continue

        row = clinical_row.iloc[0].to_dict()
        row["Image_Path"] = png_filename
        matched_rows.append(row)

    print(f"\n[3/3] Conversion results:")
    print(f"      Successfully matched + converted: {len(matched_rows)}")
    print(f"      Skipped (folder name not parseable): {skipped_no_rid}")
    print(f"      Skipped (no matching clinical row for RID): {skipped_no_clinical}")
    print(f"      Skipped (no DICOM files found): {skipped_no_dicom}")

    final_df = pd.DataFrame(matched_rows)
    final_df.to_csv(output_csv, index=False)
    print(f"\n[+] Saved final multimodal dataset to: {output_csv}")
    print(f"[+] Images saved to: {images_out}")

    if len(final_df) > 0:
        print("\nClass distribution in final matched dataset (0=CN, 1=MCI, 2=AD):")
        print(final_df["Group"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans_dir", required=True, help="Path to .../MyMRIScans/ADNI folder")
    parser.add_argument("--clinical_csv", required=True, help="Path to clinical_data_adni.csv")
    parser.add_argument("--images_out", required=True, help="Folder to save converted PNG images")
    parser.add_argument("--output", required=True, help="Path to save final merged CSV")
    args = parser.parse_args()
    main(args.scans_dir, args.clinical_csv, args.images_out, args.output)
