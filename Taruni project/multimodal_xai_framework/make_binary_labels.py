"""
Converts the 3-class ADNI dataset (CN=0, MCI=1, AD=2) into a binary
classification dataset: CN (Normal) vs Impaired (MCI+AD combined).

This is a standard, defensible technique when the 3-class problem has
too few samples per class for reliable training (your case: 122 subjects
across CN/MCI/AD, with just 28 in the AD class).

USAGE:
    python make_binary_labels.py --input data/clinical_data_adni_full.csv --output data/clinical_data_adni_binary.csv
"""

import argparse
import pandas as pd


def main(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Original 3-class distribution:")
    print(df["Group"].value_counts().sort_index())

    # 0 = CN stays 0 (Normal)
    # 1 (MCI) and 2 (AD) both become 1 (Impaired)
    df["Group"] = df["Group"].apply(lambda x: 0 if x == 0 else 1)

    print(f"\nNew binary distribution (0=Normal, 1=Impaired):")
    print(df["Group"].value_counts().sort_index())

    df.to_csv(output_path, index=False)
    print(f"\n[+] Saved binary-label dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
