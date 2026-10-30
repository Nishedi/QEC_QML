#!/usr/bin/env python3
"""Compare a standard repetition-code decoder with a classical ML decoder.

Reads a CSV previously created by generate_calibrated_training_data.py.
It DOES NOT generate new samples or access Qiskit.

Install: pip install scikit-learn
Run:     python korekcja_klasyczna_vs_ml.py --input training_data.csv

ML inputs:  s1, s2 (syndrome only)
ML target:  additional_x (whether to flip the standard decoder's result)
The known sent bit is used to verify labels and to score test predictions,
NEVER as an input feature for ML.
"""

import argparse
import csv
import json
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier


def syndrome(received):
    """Compare the data bit with each ancillary bit."""
    return (received[0] ^ received[1], received[0] ^ received[2])


def standard_correction(received):
    """Decode the repetition code by majority voting."""
    return received[0] ^ int(syndrome(received) == (1, 1))


def binary_value(text, column, row_number):
    if str(text).strip() not in ("0", "1"):
        raise ValueError(f"Row {row_number}: '{column}' must be 0 or 1, got {text!r}")
    return int(text)


def load_data(csv_path):
    """Read and validate the CSV produced by the calibrated data generator."""
    train, test = [], []
    required = {"split", "sent", "received", "s1", "s2", "additional_x"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            split = row["split"].strip().lower()
            if split not in ("train", "test"):
                raise ValueError(f"Row {row_number}: split must be 'train' or 'test'")

            sent = binary_value(row["sent"], "sent", row_number)
            bit_string = row["received"].strip()
            if len(bit_string) != 3 or any(bit not in "01" for bit in bit_string):
                raise ValueError(f"Row {row_number}: invalid received bits: {bit_string!r}")
            received = tuple(int(bit) for bit in bit_string)
            s = (
                binary_value(row["s1"], "s1", row_number),
                binary_value(row["s2"], "s2", row_number),
            )
            additional_x = binary_value(row["additional_x"], "additional_x", row_number)

            # Detect accidental changes in bit ordering or target definition.
            if s != syndrome(received):
                raise ValueError(f"Row {row_number}: syndrome does not match received bits")
            if additional_x != int(standard_correction(received) != sent):
                raise ValueError(f"Row {row_number}: additional_x label is inconsistent")

            sample = {"sent": sent, "received": received,
                      "syndrome": s, "additional_x": additional_x}
            (train if split == "train" else test).append(sample)

    if not train or not test:
        raise ValueError("CSV must contain both training and test rows")
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="training_data.csv",
                        help="CSV from generate_calibrated_training_data.py")
    args = parser.parse_args()
    csv_path = Path(args.input)
    if not csv_path.is_file():
        parser.error(f"CSV file not found: {csv_path}")

    train, test = load_data(csv_path)
    print(f"Loaded {len(train)} training and {len(test)} test samples from {csv_path}")

    metadata_path = csv_path.with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print("Data source:", metadata.get("source_type", "unknown"))

    # VERSION 2: Learn the extra correction using ONLY two syndrome bits.
    model = DecisionTreeClassifier(max_depth=2, random_state=0)
    X_train = [sample["syndrome"] for sample in train]
    y_train = [sample["additional_x"] for sample in train]
    model.fit(X_train, y_train)

    # Both approaches are evaluated on EXACTLY the same held-out test rows.
    X_test = [sample["syndrome"] for sample in test]
    ml_corrections = model.predict(X_test)

    correct_standard = sum(
        standard_correction(sample["received"]) == sample["sent"]
        for sample in test
    )
    correct_ml = sum(
        (standard_correction(sample["received"]) ^ int(extra_x)) == sample["sent"]
        for sample, extra_x in zip(test, ml_corrections)
    )

    standard_accuracy = correct_standard / len(test)
    ml_accuracy = correct_ml / len(test)
    print("\nTest results:")
    print(f"  Standard correction: {standard_accuracy:.2%} ({correct_standard}/{len(test)})")
    print(f"  Standard + ML:       {ml_accuracy:.2%} ({correct_ml}/{len(test)})")
    print(f"  Difference:          {(ml_accuracy - standard_accuracy) * 100:+.2f} pp")

    print("\nLearned additional X by syndrome:")
    for s in ((0, 0), (1, 0), (0, 1), (1, 1)):
        print(f"  {s[0]}{s[1]} -> {int(model.predict([s])[0])}")

    print("\nNote: the generator's CSV contains CALIBRATION-BASED SIMULATED shots,"
          " not measurements from physical Odra 5.")
    print(train.groupby(["s1", "s2"])["additional_x"].agg(["mean", "count"]))

if __name__ == "__main__":
    main()