"""Matriz de fidelidad media entre transmisores (para elegir el par A/B vs C).

Diagonal: F(A,B) del mismo Tx. Fuera de la diagonal: F(A,C) entre Tx distintos.
Uso: python scripts/tx_matrix.py --kind iq --alpha 0.8 --n-avg 10 --eq 1 --plot
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import build_abc  # noqa: E402
from src import data as D, distances as M, encoding as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="data/ManySig.pkl")
ap.add_argument("--kind", default="iq")
ap.add_argument("--alpha", type=float, default=0.8)
ap.add_argument("--n-avg", type=int, default=10)
ap.add_argument("--rx", type=int, default=0)
ap.add_argument("--day", type=int, default=0)
ap.add_argument("--eq", type=int, default=1)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--plot", action="store_true")
a = ap.parse_args()

index = D.index_signals(D.load_pkl(a.pkl))
txs = D.available(index)[0]
n = len(txs)
mat = np.zeros((n, n))
for i, ta in enumerate(txs):
    for j, tc in enumerate(txs):
        vals = []
        for seed in range(a.seeds):
            d, _ = build_abc(index, a.kind, a.alpha, a.n_avg, ta, tc,
                             a.rx, a.day, seed, eq=a.eq)
            s = [E.amplitude_encode(v) for v in d]
            vals.append(M.fidelity(s[0], s[1]) if i == j
                        else (M.fidelity(s[0], s[2]) + M.fidelity(s[1], s[2])) / 2)
        mat[i, j] = np.mean(vals)

print(f"{a.kind} alpha={a.alpha} n_avg={a.n_avg} eq={a.eq} rx={a.rx} dia={a.day}")
print("filas = Tx de A/B, columnas = Tx de C (diagonal = F(A,B))\n")
print("      " + "".join(f"tx{t:<6}" for t in txs))
for i, t in enumerate(txs):
    print(f"tx{t:<4}" + "".join(f"{mat[i, j]:<8.3f}" for j in range(n)))

pairs = sorted(((abs((mat[i, j] + mat[j, i]) / 2 - 0.3), txs[i], txs[j],
                 (mat[i, j] + mat[j, i]) / 2)
                for i in range(n) for j in range(i + 1, n)))
print("\nPares de Tx con F(A,C) mas cercano a 0.3:")
for _, ta, tc, f in pairs[:5]:
    print(f"  tx{ta} vs tx{tc}: F={f:.3f}")

Path("results").mkdir(exist_ok=True)
np.savetxt(f"results/tx_matrix_{a.kind}_a{a.alpha}_n{a.n_avg}_eq{a.eq}.csv", mat, delimiter=",", fmt="%.4f")
if a.plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n), [f"tx{t}" for t in txs]); ax.set_yticks(range(n), [f"tx{t}" for t in txs])
    for (r, c), v in np.ndenumerate(mat):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=8)
    fig.colorbar(im); ax.set_title("Fidelidad media entre transmisores")
    fig.tight_layout()
    png = f"results/tx_matrix_{a.kind}_a{a.alpha}_n{a.n_avg}_eq{a.eq}.png"
    fig.savefig(png, dpi=200); print("guardado:", png)
