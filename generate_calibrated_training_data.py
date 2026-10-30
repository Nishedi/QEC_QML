#!/usr/bin/env python3
"""Generate classical decoder training data from an Odra-5 calibration JSON.

Compatible with Qiskit 1.2.4 + qiskit-aer 0.15.1.

Only CALIBRATION-BASED SIMULATED measurements are generated; this does NOT
connect to a physical QPU or measure the actual underlying quantum errors.

Example:
    python generate_calibrated_training_data.py --calibration calibration_data.json --output training_data.csv --shots 5000

Each shot records the known input bit, three measured bits, the syndrome,
the result of a majority-vote decoder, and the correct supervision label for
an ADDITIONAL X after that decoder. Train the classifier using only s1 and
s2 as features, and additional_x as target (not the known input bit!).
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

# Physical indices: QB1=0, QB2=1, QB3=2 (the center of the star).
DATA, ANC_A, ANC_B = 2, 0, 1
USED_QUBITS = (DATA, ANC_A, ANC_B)
USED_EDGES = {(0, 2), (1, 2)}

COLUMNS = [
    "split", "sent", "received", "received_d", "received_a", "received_b",
    "s1", "s2", "standard_decoded", "additional_x", "observed_difference",
]


def validate_probability(value, field):
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean calibration value: {field}")
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid calibration value in {field}: {value!r}") from exc
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"Calibration value outside [0,1]: {field} -> {value!r}")
    return v


def insert_unique(target, key, value, field):
    if key in target and not math.isclose(target[key], value, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"Conflicting calibration values for {key}: {field}")
    target[key] = value


def load_calibration(path):
    """Parse the metric fields used in the user's original backend.py."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ValueError("Expected JSON object with an 'observations' list")

    one_qubit = {}           # q -> depolarizing parameter
    two_qubit = {}           # (q1, q2), sorted -> depolarizing parameter
    readout = {}             # q -> {p01, p10}

    for item in data["observations"]:
        if not isinstance(item, dict):
            continue
        field = item.get("dut_field", "")
        if not isinstance(field, str):
            continue

        is_oneq = "metrics.rb.clifford.xy" in field
        is_cz = "metrics.irb.cz" in field
        is_readout = "metrics.ssro.measure.constant" in field
        if not (is_oneq or is_cz or is_readout):
            continue

        qubits = [int(n) - 1 for n in re.findall(r"QB(\d+)(?!\d)", field)]
        if not qubits or any(q not in range(5) for q in qubits):
            raise ValueError(f"Unexpected qubit identifier: {field}")

        if is_oneq:
            if len(qubits) != 1:
                raise ValueError(f"Unexpected single-qubit metric: {field}")
            q = qubits[0]
            if q in USED_QUBITS:
                value = validate_probability(item.get("value"), field)
                # EXACTLY the original backend.py convention, not an inferred
                # conversion from RB fidelity to gate error probability.
                insert_unique(one_qubit, q, 1.0 - value, field)

        elif is_cz:
            if len(qubits) != 2 or qubits[0] == qubits[1]:
                raise ValueError(f"Unexpected CZ metric: {field}")
            edge = tuple(sorted(qubits))
            if edge in USED_EDGES:
                value = validate_probability(item.get("value"), field)
                insert_unique(two_qubit, edge, 1.0 - value, field)

        else:
            if len(qubits) != 1:
                raise ValueError(f"Unexpected readout metric: {field}")
            q = qubits[0]
            if q not in USED_QUBITS:
                continue
            if "error_0_to_1" in field:
                key = "p01"
            elif "error_1_to_0" in field:
                key = "p10"
            else:
                continue
            insert_unique(readout.setdefault(q, {}), key,
                          validate_probability(item.get("value"), field), field)

    missing = []
    for q in USED_QUBITS:
        if q not in one_qubit:
            missing.append(f"metrics.rb.clifford.xy for QB{q + 1}")
        for direction in ("p01", "p10"):
            if direction not in readout.get(q, {}):
                missing.append(f"readout {direction} for QB{q + 1}")
    for edge in USED_EDGES:
        if edge not in two_qubit:
            missing.append(f"metrics.irb.cz for QB{edge[0] + 1}__QB{edge[1] + 1}")
    if missing:
        raise ValueError("Incomplete calibration:\n- " + "\n- ".join(missing))

    return {
        "created_timestamp": data.get("created_timestamp"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "one_qubit": one_qubit,
        "two_qubit": two_qubit,
        "readout": readout,
    }


def create_simulator(calibration):
    """Apply 1q/CZ depolarizing and 1q readout errors, as in backend.py."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies: pip install qiskit==1.2.4 qiskit-aer==0.15.1"
        ) from exc

    noise = NoiseModel()
    for q, probability in calibration["one_qubit"].items():
        noise.add_quantum_error(
            depolarizing_error(probability, 1), ["h", "rx", "ry", "rz"], [q]
        )
    for (a, b), probability in calibration["two_qubit"].items():
        error = depolarizing_error(probability, 2)
        # Although CZ is symmetric, Aer assigns errors by ordered qubit indices.
        noise.add_quantum_error(error, "cz", [a, b])
        noise.add_quantum_error(error, "cz", [b, a])
    for q, rates in calibration["readout"].items():
        p01, p10 = rates["p01"], rates["p10"]
        noise.add_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q]
        )

    return AerSimulator(method="density_matrix", noise_model=noise)


def make_circuit(sent, stress_pairs):
    """Encode logical 0/1 in 3 connected qubits, then measure all three.

    Only H, RX and CZ are used, so the gate noise can be attached explicitly.
    The identity CZ-CZ pairs increase exposure to calibrated gate noise.
    Without compiler cancellation, they are identity in the ideal circuit.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5, 3)
    if sent == 1:
        qc.rx(math.pi, DATA)  # prepare |1>, including calibrated 1q noise

    for anc in (ANC_A, ANC_B):
        # CNOT(DATA, anc) = H(anc) CZ(DATA, anc) H(anc)
        qc.h(anc)
        qc.cz(DATA, anc)
        qc.h(anc)

    for _ in range(stress_pairs):
        for anc in (ANC_A, ANC_B):
            qc.cz(DATA, anc)
            qc.cz(DATA, anc)

    # Classical bit 0 = DATA; 1 = ANC_A; 2 = ANC_B.
    # Qiskit prints bitstrings in REVERSE classical-bit order (c2 c1 c0).
    qc.measure([DATA, ANC_A, ANC_B], [0, 1, 2])
    return qc


def syndrome(received):
    return (received[0] ^ received[1], received[0] ^ received[2])


def standard_decode(received):
    return received[0] ^ int(syndrome(received) == (1, 1))


def make_row(sent, memory):
    # Memory example '011' means c2=0, c1=1, c0=1 -> received=(1,1,0).
    compact = memory.replace(" ", "")
    if len(compact) != 3 or any(bit not in "01" for bit in compact):
        raise ValueError(f"Unexpected 3-bit measurement result: {memory!r}")
    received = tuple(int(b) for b in compact[::-1])  # DATA, ANC_A, ANC_B
    s1, s2 = syndrome(received)
    standard = standard_decode(received)
    return {
        "sent": sent,
        "received": "".join(map(str, received)),
        "received_d": received[0],
        "received_a": received[1],
        "received_b": received[2],
        "s1": s1,
        "s2": s2,
        "standard_decoded": standard,
        "additional_x": int(standard != sent),
        # NOT the true quantum error! Includes readout/preparation effects.
        "observed_difference": "".join(str(bit ^ sent) for bit in received),
    }


def generate(calibration, shots, stress_pairs, train_fraction, seed):
    simulator = create_simulator(calibration)
    circuits = [make_circuit(0, stress_pairs), make_circuit(1, stress_pairs)]
    result = simulator.run(circuits, shots=shots, memory=True,
                           seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(f"Aer simulation failed: {result.status}")

    rng = random.Random(seed)
    rows = []
    for sent in (0, 1):
        data = [make_row(sent, memory) for memory in result.get_memory(sent)]
        if len(data) != shots:
            raise RuntimeError(f"Expected {shots} shots for sent={sent}; got {len(data)}")
        rng.shuffle(data)
        train_count = int(shots * train_fraction)
        for index, row in enumerate(data):
            row["split"] = "train" if index < train_count else "test"
        rows.extend(data)
    rng.shuffle(rows)
    return rows, circuits


def write_results(output, rows, calibration, shots, stress_pairs, train_fraction, seed, circuits):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    test_rows = [row for row in rows if row["split"] == "test"]
    baseline_accuracy = sum(row["additional_x"] == 0 for row in test_rows) / len(test_rows)
    grouped = {}
    for sent in (0, 1):
        grouped[str(sent)] = dict(sorted(Counter(
            row["received"] for row in rows if row["sent"] == sent
        ).items()))

    summary = {
        "source_type": "SIMULATED_USING_CALIBRATION_NOT_PHYSICAL_QPU_DATA",
        "calibration_sha256": calibration["sha256"],
        "calibration_timestamp": calibration["created_timestamp"],
        "qubits": {"data": "QB3", "ancilla_a": "QB1", "ancilla_b": "QB2"},
        "noise_convention": "depolarizing_parameter = 1 - metric_value (as in backend.py)",
        "noise_model": {
            "one_qubit_parameters": {f"QB{q+1}": v for q, v in calibration["one_qubit"].items()},
            "cz_parameters": {f"QB{a+1}__QB{b+1}": v for (a, b), v in calibration["two_qubit"].items()},
            "readout": {f"QB{q+1}": v for q, v in calibration["readout"].items()},
        },
        "shots_per_prepared_bit": shots,
        "total_rows": len(rows),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "test_rows": len(test_rows),
        "train_fraction": train_fraction,
        "seed": seed,
        "extra_identity_cz_pairs_per_edge": stress_pairs,
        "standard_decoder_test_accuracy": baseline_accuracy,
        "measured_bitstrings_by_prepared_bit": grouped,
        "feature_columns": ["s1", "s2"],
        "target_column": "additional_x",
        "circuit_gate_counts": {str(sent): dict(circuits[sent].count_ops()) for sent in (0, 1)},
        "limitations": [
            "Generated by Qiskit Aer simulation from calibration metrics; not physical Odra shots.",
            "observed_difference includes readout and preparation errors and is NOT the real gate error pattern.",
            "Supervised labels are derived from the known prepared computational basis bit.",
            "Training on basis bits does not validate quantum coherence or full quantum error correction.",
            "No T1/T2, idling, leakage, crosstalk or arbitrary correlated noise is modeled.",
            "CZ-CZ pairs are an identity only before noise; a hardware compiler might cancel them.",
            "A single calibration and random shot split do not establish transfer to a different calibration.",
        ],
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path, baseline_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", default="calibration_data.json",
                        help="Path to calibration JSON (default: calibration_data.json)")
    parser.add_argument("--output", default="training_data.csv",
                        help="Output CSV path (default: training_data.csv)")
    parser.add_argument("--shots", type=int, default=5000,
                        help="Number of shots PER prepared bit (default: 5000)")
    parser.add_argument("--stress-pairs", type=int, default=2,
                        help="Extra CZ-CZ identity pairs PER edge (default: 2)")
    parser.add_argument("--train-fraction", type=float, default=0.8,
                        help="Fraction of rows per prepared bit for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.shots < 2:
        parser.error("--shots must be >= 2")
    if args.stress_pairs < 0:
        parser.error("--stress-pairs must be >= 0")
    if not (0 < args.train_fraction < 1):
        parser.error("--train-fraction must be between 0 and 1")
    if not (1 <= int(args.shots * args.train_fraction) < args.shots):
        parser.error("Selected --train-fraction leaves an empty train or test split")

    calibration = load_calibration(args.calibration)
    rows, circuits = generate(calibration, args.shots,
                               args.stress_pairs, args.train_fraction, args.seed)
    metadata_path, baseline = write_results(args.output, rows, calibration,
                                            args.shots, args.stress_pairs,
                                            args.train_fraction, args.seed, circuits)
    print(f"Saved {len(rows):,} CALIBRATION-SIMULATED samples to {args.output}")
    print(f"Metadata: {metadata_path}")
    print(f"Standard decoder test accuracy: {baseline:.2%}")
    print("ML features: ['s1', 's2']; label: 'additional_x'; filter split=='train' / 'test'")
    print("These are NOT measurements from the physical Odra-5 processor.")


if __name__ == "__main__":
    main()