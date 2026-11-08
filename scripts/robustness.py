"""Robustez de una configuracion: todos los pares (tx_a, tx_c) y varias semillas.

Uso: python scripts/robustness.py --pkl data/ManySig.pkl --kind amp --alpha 0.5 --n-avg 1 --eq 0
"""
import argparse
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import build_abc  # noqa: E402
from src import data as D, distances as M, encoding as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="data/ManySig.pkl")
ap.add_argument("--kind", default="amp")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--n-avg", type=int, default=1)
ap.add_argument("--rx", type=int, default=0)
ap.add_argument("--day", type=int, default=0)
ap.add_argument("--eq", type=int, default=0)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

index = D.index_signals(D.load_pkl(a.pkl))
txs = D.available(index)[0]
fab, fac = [], []
rows = []
for seed in range(a.seeds):
    for tx_a, tx_c in permutations(txs, 2):
        d, _ = build_abc(index, a.kind, a.alpha, a.n_avg, tx_a, tx_c, a.rx, a.day, seed, eq=a.eq)
        s = [E.amplitude_encode(v) for v in d]
        fab.append(M.fidelity(s[0], s[1]))
        fac.append((M.fidelity(s[0], s[2]) + M.fidelity(s[1], s[2])) / 2)
        rows.append((tx_a, tx_c, fab[-1], fac[-1]))
fab, fac = np.array(fab), np.array(fac)
print(f"{a.kind} alpha={a.alpha} n_avg={a.n_avg} eq={a.eq}  ({len(fab)} tripletas)")
print(f"F(A,B)      media={fab.mean():.3f} std={fab.std():.3f} min={fab.min():.3f}")
print(f"F(A,C),F(B,C) media={fac.mean():.3f} std={fac.std():.3f} max={fac.max():.3f}")
print(f"tripletas con F_AB > F_AC: {np.mean(fab > fac) * 100:.1f}%")

print("\nF(A,B) medio por Tx de A:")
for t in txs:
    v = [r[2] for r in rows if r[0] == t]
    print(f"  tx {t}: {np.mean(v):.3f}")
print("\nPares con F(A,C) mas alto:")
for r in sorted(rows, key=lambda r: -r[3])[:5]:
    print(f"  A=tx{r[0]} C=tx{r[1]}: F_AB={r[2]:.3f} F_AC={r[3]:.3f}")