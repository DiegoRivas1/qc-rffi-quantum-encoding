"""Verificacion opcional con Qiskit: Swap Test real (17 qubits) vs fidelidad exacta.

Uso: python scripts/qiskit_swap_test.py --pkl data/ManySig.pkl --kind fft_mag --alpha 0.9
(Primero valida con run_experiment.py; esto solo confirma con un circuito.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import build_abc  # noqa: E402
from src import data as D, encoding as E  # noqa: E402


def swap_test_circuit(a, b):
    n = E.n_qubits(len(a))
    anc, ra, rb = QuantumRegister(1, "anc"), QuantumRegister(n, "a"), QuantumRegister(n, "b")
    c = ClassicalRegister(1, "m")
    qc = QuantumCircuit(anc, ra, rb, c)
    qc.append(StatePreparation(a), ra)
    qc.append(StatePreparation(b), rb)
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], ra[i], rb[i])
    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/ManySig.pkl")
    ap.add_argument("--tx-a", type=int, default=0)
    ap.add_argument("--tx-c", type=int, default=1)
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--kind", default="fft_mag")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--n-avg", type=int, default=50)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = D.index_signals(D.load_pkl(args.pkl))
    datums, _ = build_abc(index, args.kind, args.alpha, args.n_avg,
                          args.tx_a, args.tx_c, args.rx, args.day, args.seed)
    s = [E.amplitude_encode(v) for v in datums]
    sim = AerSimulator()
    for (i, j, name) in [(0, 1, "A-B"), (0, 2, "A-C"), (1, 2, "B-C")]:
        qc = transpile(swap_test_circuit(s[i], s[j]), sim)
        counts = sim.run(qc, shots=args.shots).result().get_counts()
        p0 = counts.get("0", 0) / args.shots
        f_swap = max(0.0, 2 * p0 - 1)
        f_exact = state_fidelity(Statevector(s[i]), Statevector(s[j]))
        print(f"{name}: swap={f_swap:.4f}  exacta={f_exact:.4f}")
