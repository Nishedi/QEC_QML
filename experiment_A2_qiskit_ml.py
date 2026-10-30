#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment A2: real quantum circuits + classical ML decoder.

Qiskit 1.2.4 / qiskit-aer 0.15.1 / scikit-learn.

We prepare the 3-qubit repetition code with CNOT = H-CZ-H, add a genuine
TWO-QUBIT quantum noise channel X(DATA) X(ANC_A) *before* decoding,
learn a binary syndrome correction from separate basis-state experiments,
and compare two coherent quantum recoveries on a Bell pair:
    (1) fixed majority-vote recovery,
    (2) the same recovery circuit with a syndrome rule learned by classical ML.

The learned rule is a CLASSICAL lookup compiled into a fixed quantum circuit;
no QML is used. No intermediate data-qubit measurement, postselection or
mid-circuit feed-forward is required during Bell recovery.

The calibrated gate/readout noise model is adapted from the supplied backend.py.
The additional correlated X channel is synthetic and NOT inferred from the
calibration. Aer data are SIMULATED, never physical Odra measurements.

Examples:
    python experiment_A2_qiskit_ml.py --calibration calibration_data.json
    python experiment_A2_qiskit_ml.py --demo --train-shots 500 --test-shots 200

Outputs in results_A2/: summary.csv, learned_rules.csv, train_samples.csv,
test_samples.csv, example_circuits.txt, run_metadata.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Odra-style five-qubit star. Python/Qiskit indices start at 0.
ANC_A, ANC_B, DATA, REF, N_QUBITS = 0, 1, 2, 3, 5
USED_QUBITS = (ANC_A, ANC_B, DATA, REF)
USED_CZ = {(0, 2), (1, 2), (2, 3)}
SYNDROMES = ((0, 0), (0, 1), (1, 0), (1, 1))


def validate_probability(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Not a number: {field}={value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Invalid probability {field}={value!r}")
    return number


def remember(target, key, value):
    if key in target and not math.isclose(target[key], value, abs_tol=1e-12):
        raise ValueError(f"Conflicting calibration metric for {key!r}")
    target[key] = value


def read_calibration(path: Path):
    """Read the exact metric families used by the original backend.py.

    Caveat: 1-metric is mapped to Aer depolarizing parameter 1-metric,
    reproducing the ORIGINAL backend convention; it is not automatically
    an exact physical fidelity-to-channel conversion.
    """
    raw = path.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("observations"), list):
        raise ValueError("Expected JSON with an 'observations' list")

    one, cz, readout_flat = {}, {}, {}
    for obs in obj["observations"]:
        if not isinstance(obs, dict):
            continue
        field = obs.get("dut_field", "")
        if not isinstance(field, str):
            continue
        oneq = "metrics.rb.clifford.xy" in field
        twoq = "metrics.irb.cz" in field
        ro = "metrics.ssro.measure.constant" in field
        if not (oneq or twoq or ro):
            continue
        indices = [int(n) - 1 for n in re.findall(r"QB(\d+)(?!\d)", field)]
        if not indices or any(q not in range(N_QUBITS) for q in indices):
            raise ValueError(f"Unknown qubit in metric: {field}")
        if oneq:
            if len(indices) != 1:
                raise ValueError(f"Expected one qubit in: {field}")
            if indices[0] in USED_QUBITS:
                remember(one, indices[0], 1 - validate_probability(obs.get("value"), field))
        elif twoq:
            if len(indices) != 2 or indices[0] == indices[1]:
                raise ValueError(f"Expected 2 distinct qubits in: {field}")
            pair = tuple(sorted(indices))
            if pair in USED_CZ:
                remember(cz, pair, 1 - validate_probability(obs.get("value"), field))
        else:
            if len(indices) != 1 or indices[0] not in USED_QUBITS:
                continue
            direction = ("p01" if "error_0_to_1" in field else
                         "p10" if "error_1_to_0" in field else None)
            if direction is not None:
                remember(readout_flat, (indices[0], direction),
                         validate_probability(obs.get("value"), field))

    missing = []
    for q in USED_QUBITS:
        if q not in one:
            missing.append(f"single-qubit RB for QB{q+1}")
        for kind in ("p01", "p10"):
            if (q, kind) not in readout_flat:
                missing.append(f"readout {kind} for QB{q+1}")
    for edge in USED_CZ:
        if edge not in cz:
            missing.append(f"CZ IRB for QB{edge[0]+1}__QB{edge[1]+1}")
    if missing:
        raise ValueError("Incomplete calibration:\n  " + "\n  ".join(missing))

    return {
        "one": one, "cz": cz,
        "readout": {q: (readout_flat[(q, "p01")], readout_flat[(q, "p10")])
                    for q in USED_QUBITS},
        "sha256": hashlib.sha256(raw).hexdigest(),
        "timestamp": obj.get("created_timestamp"),
        "source": str(path), "synthetic": False,
    }


def synthetic_calibration():
    """For an explicit --demo only; NOT an Odra calibration."""
    return {
        "one": {q: 0.002 for q in USED_QUBITS},
        "cz": {edge: 0.01 for edge in USED_CZ},
        "readout": {q: (0.008, 0.012) for q in USED_QUBITS},
        "sha256": None, "timestamp": None,
        "source": "EXPLICIT_SYNTHETIC_DEMO", "synthetic": True,
    }


def make_simulator(cal):
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise SystemExit("Install: pip install qiskit==1.2.4 qiskit-aer==0.15.1") from exc

    noise = NoiseModel()
    for q, p in cal["one"].items():
        noise.add_quantum_error(depolarizing_error(p, 1),
                                ["h", "rx", "ry", "rz"], [q])
    for (a, b), p in cal["cz"].items():
        error = depolarizing_error(p, 2)
        # Aer 2-qubit errors depend on the ORDER of the instruction qubits.
        noise.add_quantum_error(error, "cz", [a, b])
        noise.add_quantum_error(error, "cz", [b, a])
    for q, (p01, p10) in cal["readout"].items():
        noise.add_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q]
        )
    return AerSimulator(method="density_matrix", noise_model=noise,
                        fusion_enable=False)


def cx_via_cz(qc, control, target):
    """CNOT(control,target) decomposed into calibrated H-CZ-H."""
    assert tuple(sorted((control, target))) in USED_CZ
    qc.h(target)
    qc.cz(control, target)
    qc.h(target)


def add_correlated_fault(qc, p_corr):
    """CPTP channel: with p_corr apply X on D AND A before decoding."""
    if not p_corr:
        return
    # This Kraus operator is simulator-only: it models a specified physical
    # fault channel, not an additional device gate subject to RB gate errors.
    from qiskit.quantum_info import Kraus
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(4, dtype=complex)
    XX = np.kron(X, X)
    qc.append(Kraus([math.sqrt(1 - p_corr) * I,
                     math.sqrt(p_corr) * XX]).to_instruction(), [DATA, ANC_A])


def encode(qc):
    cx_via_cz(qc, DATA, ANC_A)
    cx_via_cz(qc, DATA, ANC_B)


def decode(qc):
    cx_via_cz(qc, DATA, ANC_B)
    cx_via_cz(qc, DATA, ANC_A)


def syndrome_correction_bits(s):
    """Fixed majority rule: after decode flip D for syndrome 11 only."""
    return int(s == (1, 1))


def angles_for_policy(extra_x):
    """Logical controlled-RX angles, in order (00, 10, 01, 11)."""
    return [math.pi * (syndrome_correction_bits(s) ^ int(extra_x.get(s, 0)))
            for s in ((0, 0), (1, 0), (0, 1), (1, 1))]


def betas(theta):
    """Convert syndrome-conditioned RX angles to four physical RX gates.

    For syndrome (a,b), resulting angle is:
        beta0 + (-1)^a beta1 + (-1)^(a+b) beta2 + (-1)^b beta3.
    Only CZ(A,D) and CZ(B,D) are required (star connectivity).
    """
    t00, t10, t01, t11 = theta
    return [
        (t00 + t10 + t01 + t11) / 4,
        (t00 - t10 + t01 - t11) / 4,
        (t00 - t10 - t01 + t11) / 4,
        (t00 + t10 - t01 - t11) / 4,
    ]


def add_coherent_recovery(qc, extra_x):
    """Fixed four-RX/four-CZ circuit; no intermediate syndrome measurement.

    Learned policy is classical. It is compiled into the SAME physical gate
    skeleton as the standard majority policy for a fair comparison.
    """
    b0, b1, b2, b3 = betas(angles_for_policy(extra_x))
    qc.rx(b0, DATA)
    qc.cz(ANC_A, DATA)
    qc.rx(b1, DATA)
    qc.cz(ANC_B, DATA)
    qc.rx(b2, DATA)
    qc.cz(ANC_A, DATA)
    qc.rx(b3, DATA)
    qc.cz(ANC_B, DATA)


def training_probe(sent, p_corr):
    """Known logical 0/1 samples: measure D and the DECODED syndrome bits."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS, 3)
    if sent:
        qc.rx(math.pi, DATA)
    encode(qc)
    add_correlated_fault(qc, p_corr)
    decode(qc)
    # c0=D, c1=A=s1, c2=B=s2; Qiskit bitstrings are c2 c1 c0.
    qc.measure([DATA, ANC_A, ANC_B], [0, 1, 2])
    return qc


def parse_probe_memory(sent, memory, split, p_corr):
    bits = str(memory).replace(" ", "")
    if len(bits) != 3 or any(x not in "01" for x in bits):
        raise ValueError(f"Unexpected 3-bit measurement result: {memory!r}")
    d, s1, s2 = (int(bit) for bit in bits[::-1])
    standard = d ^ int(s1 == 1 and s2 == 1)
    return {"p_corr": p_corr, "split": split, "sent": sent,
            "measured_d": d, "s1": s1, "s2": s2,
            "standard_decoded": standard,
            "additional_x": int(standard != sent)}


def generate_examples(sim, p_corr, shots, seed, split):
    if shots < 1:
        raise ValueError("shots must be >= 1")
    circuits = [training_probe(0, p_corr), training_probe(1, p_corr)]
    result = sim.run(circuits, memory=True, shots=shots,
                     seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(f"Probe simulation failed: {result.status}")
    examples = []
    for sent in (0, 1):
        examples.extend(parse_probe_memory(sent, m, split, p_corr)
                        for m in result.get_memory(sent))
    return examples


def learn_decoder(train):
    ml = DecisionTreeClassifier(max_depth=2, random_state=42)
    ml.fit([[r["s1"], r["s2"]] for r in train],
           [r["additional_x"] for r in train])
    learned = {s: int(ml.predict([list(s)])[0]) for s in SYNDROMES}
    counts = defaultdict(Counter)
    for r in train:
        counts[(r["s1"], r["s2"])][r["additional_x"]] += 1
    # The optimal empirical per-syndrome lookup is a necessary sanity baseline.
    lookup = {s: int(counts[s][1] > counts[s][0]) for s in SYNDROMES}
    return learned, lookup, counts


def binary_accuracy(test, additional_rule):
    correct = sum((r["standard_decoded"] ^ additional_rule.get(
        (r["s1"], r["s2"]), 0)) == r["sent"] for r in test)
    return correct / len(test)


def bell_circuit(p_corr, extra_x):
    """Bell (R,D), encode D, corrupt D/A, decode, recover on D/A/B only."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS)
    qc.h(REF)
    cx_via_cz(qc, REF, DATA)
    encode(qc)
    add_correlated_fault(qc, p_corr)
    decode(qc)
    add_coherent_recovery(qc, extra_x)
    return qc


def exact_bell_fidelity(sim, p_corr, rule, seed):
    """Bell overlap before readout; Aer diagnostic, not direct QPU observable."""
    import qiskit_aer  # noqa: F401: registers QuantumCircuit.save_density_matrix
    base = bell_circuit(p_corr, rule)
    qc = base.copy()
    qc.save_density_matrix([DATA, REF], label="rho_RD")
    result = sim.run(qc, shots=1, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(f"Density-matrix simulation failed: {result.status}")
    rho = np.asarray(result.data(0)["rho_RD"], dtype=complex)
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    return float(np.real(np.vdot(bell, rho @ bell)))


def measured_bell_fidelity(sim, p_corr, rule, shots, seed):
    """Raw estimate (1+XX-YY+ZZ)/4, including readout/basis-gate errors."""
    from qiskit import QuantumCircuit
    base = bell_circuit(p_corr, rule)
    circuits = []
    for basis in ("X", "Y", "Z"):
        qc = QuantumCircuit(N_QUBITS, 2)
        qc.compose(base, inplace=True)
        for q in (DATA, REF):
            if basis == "X":
                qc.h(q)
            elif basis == "Y":
                qc.rz(-math.pi / 2, q)
                qc.h(q)
        qc.measure([DATA, REF], [0, 1])
        circuits.append(qc)
    result = sim.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(f"Bell-shot simulation failed: {result.status}")
    means, vars_ = [], []
    for i in range(3):
        counts = result.get_counts(i)
        mean = sum((1 if int(k.replace(" ", ""), 2).bit_count() % 2 == 0 else -1) * v
                   for k, v in counts.items()) / shots
        means.append(mean)
        vars_.append(max(0.0, 1 - mean**2) / (shots - 1))
    return (1 + means[0] - means[1] + means[2]) / 4, math.sqrt(sum(vars_)) / 4


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_rates(s):
    numbers = [validate_probability(x.strip(), "p_corr") for x in s.split(",") if x.strip()]
    if not numbers:
        raise ValueError("At least one correlated error rate is required")
    return sorted(set(numbers))


def basic_self_tests():
    """Tests requiring only Python + NumPy, no Qiskit installation."""
    for desired in ((0, 0, 0, math.pi), (0, 0, math.pi, math.pi),
                    (math.pi, math.pi, 0, 0), (0.2, -0.5, 0.7, 2.8)):
        b = betas(desired)
        actual = (
            b[0] + b[1] + b[2] + b[3],
            b[0] - b[1] - b[2] + b[3],
            b[0] + b[1] - b[2] - b[3],
            b[0] - b[1] + b[2] - b[3],
        )
        assert np.allclose(actual, desired, atol=1e-12), (actual, desired)
    assert angles_for_policy({}) == [0, 0, 0, math.pi]
    assert angles_for_policy({(0, 1): 1}) == [0, 0, math.pi, math.pi]
    assert parse_probe_memory(0, "111", "train", 0.15)["additional_x"] == 0
    assert parse_probe_memory(0, "101", "train", 0.15)["additional_x"] == 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, default=Path("calibration_data.json"))
    parser.add_argument("--demo", action="store_true",
                        help="Use explicit synthetic demo calibration, NOT Odra")
    parser.add_argument("--rates", default="0,0.03,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--train-shots", type=int, default=3000,
                        help="Training shots PER prepared logical bit (0 and 1)")
    parser.add_argument("--test-shots", type=int, default=1000,
                        help="Held-out shots PER prepared bit")
    parser.add_argument("--bell-shots", type=int, default=3000,
                        help="Bell shots PER basis (XX, YY, ZZ) AND method")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results_A2"))
    parser.add_argument("--self-test", action="store_true",
                        help="Check syndrome mappings (requires only numpy)")
    args = parser.parse_args(argv)
    basic_self_tests()
    if args.self_test:
        print("Self-tests passed: syndrome, angle decomposition, label logic")
        return 0
    if min(args.train_shots, args.test_shots) < 1 or args.bell_shots < 2:
        parser.error("--train-shots and --test-shots >= 1, --bell-shots >= 2")
    try:
        rates = parse_rates(args.rates)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.demo and not args.calibration.is_file():
        parser.error(f"Calibration file not found: {args.calibration}")
    calibration = synthetic_calibration() if args.demo else read_calibration(args.calibration)
    sim = make_simulator(calibration)
    args.out.mkdir(parents=True, exist_ok=True)
    print("EXPERIMENT A2: QUANTUM CNOT CIRCUITS + CLASSICAL ML")
    print("Calibration:", calibration["source"])
    print("Warning: additional correlated X noise is SYNTHETIC, inserted before decoding.")
    print("All circuit outcomes come from Aer simulation, NOT the physical QPU.\n")

    summaries, rule_rows, train_rows, test_rows = [], [], [], []
    shown_rule = None
    print(f"{'p_corr':>7}  {'bit STD':>9} {'bit ML':>9} {'bit gain':>9}"
          f" {'Bell STD':>10} {'Bell ML':>10} {'Bell gain':>10}  {'lookup':>8}")
    for index, p in enumerate(rates):
        train = generate_examples(sim, p, args.train_shots,
                                  args.seed + index * 101 + 1, "train")
        test = generate_examples(sim, p, args.test_shots,
                                 args.seed + index * 101 + 2, "test")
        ml_rules, lookup, counts = learn_decoder(train)
        standard_rules = {s: 0 for s in SYNDROMES}
        bit_std = binary_accuracy(test, standard_rules)
        bit_ml = binary_accuracy(test, ml_rules)
        bit_lookup = binary_accuracy(test, lookup)
        f_std = exact_bell_fidelity(sim, p, standard_rules, args.seed + index * 101 + 3)
        f_ml = exact_bell_fidelity(sim, p, ml_rules, args.seed + index * 101 + 4)
        measured_std, se_std = measured_bell_fidelity(
            sim, p, standard_rules, args.bell_shots, args.seed + index * 101 + 5)
        measured_ml, se_ml = measured_bell_fidelity(
            sim, p, ml_rules, args.bell_shots, args.seed + index * 101 + 6)
        summaries.append({
            "p_corr": p,
            "bit_accuracy_standard": bit_std, "bit_accuracy_ml": bit_ml,
            "bit_accuracy_lookup": bit_lookup,
            "bit_gain_pp": (bit_ml - bit_std) * 100,
            "bell_pre_readout_standard": f_std,
            "bell_pre_readout_ml": f_ml,
            "bell_gain_pp": (f_ml - f_std) * 100,
            "bell_raw_measured_standard": measured_std,
            "bell_raw_measured_ml": measured_ml,
            "bell_raw_se_standard": se_std, "bell_raw_se_ml": se_ml,
            "bell_raw_gain_pp": (measured_ml - measured_std) * 100,
            "train_n": len(train), "test_n": len(test),
        })
        for s in SYNDROMES:
            tally = counts[s]
            rule_rows.append({"p_corr": p, "s1": s[0], "s2": s[1],
                              "train_n": tally[0] + tally[1],
                              "rate_standard_wrong": (tally[1] / (tally[0] + tally[1])
                                                      if tally[0] + tally[1] else ""),
                              "ml_additional_x": ml_rules[s],
                              "lookup_additional_x": lookup[s]})
        train_rows.extend(train)
        test_rows.extend(test)
        if shown_rule is None and (p >= 0.15 or index == len(rates) - 1):
            shown_rule = (p, ml_rules)
        print(f"{p:7.2f}  {bit_std:9.2%} {bit_ml:9.2%} {(bit_ml-bit_std)*100:+8.2f}pp"
              f" {f_std:10.5f} {f_ml:10.5f} {(f_ml-f_std)*100:+9.2f}pp"
              f" {bit_lookup:8.2%}", flush=True)
        print("     ML extra-X:", " ".join(f"{s[0]}{s[1]}:{ml_rules[s]}" for s in SYNDROMES))

    save_csv(args.out / "summary.csv", summaries, list(summaries[0]))
    save_csv(args.out / "learned_rules.csv", rule_rows, list(rule_rows[0]))
    columns = ["p_corr", "split", "sent", "measured_d", "s1", "s2",
               "standard_decoded", "additional_x"]
    save_csv(args.out / "train_samples.csv", train_rows, columns)
    save_csv(args.out / "test_samples.csv", test_rows, columns)
    p_example, r_example = shown_rule
    with (args.out / "example_circuits.txt").open("w", encoding="utf-8") as f:
        f.write(f"Example p_corr = {p_example}\n")
        f.write("Standard correction circuit:\n")
        f.write(str(bell_circuit(p_example, {s: 0 for s in SYNDROMES}).draw(output="text")))
        f.write("\n\nClassical-ML-guided correction circuit:\n")
        f.write(str(bell_circuit(p_example, r_example).draw(output="text")))
        f.write("\n")
    manifest = {
        "purpose": "A2: quantum circuit correction + classical ML decoder",
        "source": calibration["source"], "calibration_sha256": calibration["sha256"],
        "calibration_timestamp": calibration["timestamp"],
        "calibration_synthetic": calibration["synthetic"],
        "noise_model_convention": "1 - reported RB/IRB metric as Aer depolarizing parameter",
        "correlated_fault": "Kraus channel (1-p)*I + p*(X_DATA X_ANC_A), inserted before decode",
        "note": "Synthetic correlated fault is NOT inferred from Odra calibration."
                " Training syndrome reads are noisy; Bell recovery uses coherent controls."
                " Qiskit instruction kraus is simulator-only.",
        "rates": rates, "train_shots_per_logical_bit": args.train_shots,
        "test_shots_per_logical_bit": args.test_shots,
        "bell_shots_per_basis_per_method": args.bell_shots,
        "seed": args.seed,
        "qubit_map": {"ANC_A": ANC_A, "ANC_B": ANC_B, "DATA": DATA, "REF": REF},
    }
    (args.out / "run_metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nSaved outputs in:", args.out.resolve())
    print("Interpretation: bit accuracy is on held-out basis-state shots; Bell fidelity")
    print("is from a separate entangled-state experiment with coherent corrections.")
    print("The ML model uses ONLY s1,s2 and is trained separately for each p_corr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())