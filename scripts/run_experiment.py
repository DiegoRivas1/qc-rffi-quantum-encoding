"""Experimento A/B/C: A,B = mismo Tx (capturas disjuntas), C = otro Tx.

Uso:
  python scripts/run_experiment.py --pkl data/ManySig.pkl --tx-a 0 --tx-c 1 --rx 0 --day 0 \
         --kind fft_mag --alpha 0.9 --n-avg 50
  python scripts/run_experiment.py --pkl data/ManySig.pkl --sweep   # barrido de configuraciones
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D, distances as M, encoding as E, preprocessing as P

TARGET_AB, TARGET_AC = 0.9, 0.3


def build_abc(index, kind, alpha, n_avg, tx_a, tx_c, rx, day, seed, ref_per_tx=200, eq=0):
    rng = np.random.default_rng(seed)
    txs = D.available(index)[0]

    feats_all = []
    for tx in txs:
        try:
            x = D.get_complex(index, tx, rx, day, eq)
        except KeyError:
            continue
        sel = rng.permutation(len(x))[:ref_per_tx]
        feats_all.append(P.features(x[sel], kind))
    ref = P.reference_mean(feats_all)

    xa = D.get_complex(index, tx_a, rx, day, eq)
    xa = xa[rng.permutation(len(xa))]
    xc = D.get_complex(index, tx_c, rx, day, eq)
    xc = xc[rng.permutation(len(xc))]
    n = min(n_avg, len(xa) // 2, len(xc))
    fa = P.features(xa[:n], kind)
    fb = P.features(xa[n:2 * n], kind)
    fc = P.features(xc[:n], kind)
    return [P.make_datum(f, ref, alpha) for f in (fa, fb, fc)], n


def single(index, args):
    datums, n = build_abc(index, args.kind, args.alpha, args.n_avg,
                          args.tx_a, args.tx_c, args.rx, args.day, args.seed, eq=args.eq)
    states = [E.amplitude_encode(v) for v in datums]
    names = ["A", "B", "C"]
    rng = np.random.default_rng(args.seed)
    nq = E.n_qubits(len(datums[0]))

    print(f"\nkind={args.kind} alpha={args.alpha} n_avg={n}  ->  {nq} qubits")
    vmax = max(np.abs(v).max() for v in datums)
    bits = [E.to_bitstring(v, vmax) for v in datums]
    print(f"clasico: {len(bits[0])} bits por dato | cuantico: {nq} qubits\n")

    print(f"{'par':<5}{'<A|B>':>9}{'F=|<.>|^2':>11}{'<P_B>':>9}{'swap':>8}"
          f"{'1-JS':>8}{'KL(P||Q)':>10}{'hamming':>9}")
    rows = []
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        a, b = states[i], states[j]
        ip = M.inner_product(a, b).real
        f = M.fidelity(a, b)
        ex = M.expectation_fidelity(a, b)
        sw, _ = M.swap_test(a, b, shots=args.shots, rng=rng)
        js = M.js_similarity(a, b)
        kl = M.kl(a, b)
        ham = E.hamming_similarity(bits[i], bits[j])
        print(f"{names[i]}-{names[j]:<3}{ip:>9.4f}{f:>11.4f}{ex:>9.4f}{sw:>8.4f}"
              f"{js:>8.4f}{kl:>10.4f}{ham:>9.4f}")
        rows.append([f"{names[i]}-{names[j]}", ip, f, ex, sw, js, kl, ham])

    out = Path("results") / f"abc_{args.kind}_a{args.alpha}_n{n}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["par", "inner", "fidelity", "expectation", "swap", "js_sim", "kl", "hamming"])
        w.writerows(rows)
    print(f"\nguardado: {out}")
    if args.plot:
        import matplotlib.pyplot as plt
        mat = np.array([[M.fidelity(s, t) for t in states] for s in states])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="w")
        fig.colorbar(im); ax.set_title("Fidelidad")
        fig.tight_layout()
        png = out.with_suffix(".png"); fig.savefig(png, dpi=200)
        print(f"guardado: {png}")


def sweep(index, args):
    rows = []
    for kind in P.KINDS:
        for alpha in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0):
            for n_avg in (1, 10, 50):
                datums, n = build_abc(index, kind, alpha, n_avg, args.tx_a, args.tx_c,
                                      args.rx, args.day, args.seed, eq=args.eq)
                s = [E.amplitude_encode(v) for v in datums]
                fab, fac, fbc = M.fidelity(s[0], s[1]), M.fidelity(s[0], s[2]), M.fidelity(s[1], s[2])
                score = abs(fab - TARGET_AB) + abs((fac + fbc) / 2 - TARGET_AC)
                rows.append((score, kind, alpha, n, fab, fac, fbc))
    rows.sort()
    print(f"objetivo: F(A,B)~{TARGET_AB}, F(A,C)~F(B,C)~{TARGET_AC}\n")
    print(f"{'score':>6} {'kind':<8}{'alpha':>6}{'n_avg':>6}{'F_AB':>8}{'F_AC':>8}{'F_BC':>8}")
    for r in rows[:args.top]:
        print(f"{r[0]:>6.3f} {r[1]:<8}{r[2]:>6}{r[3]:>6}{r[4]:>8.3f}{r[5]:>8.3f}{r[6]:>8.3f}")
    out = Path("results") / f"sweep_eq{args.eq}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "kind", "alpha", "n_avg", "F_AB", "F_AC", "F_BC"])
        w.writerows(rows)
    print(f"\nbarrido completo en {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/ManySig.pkl")
    ap.add_argument("--tx-a", type=int, default=0)
    ap.add_argument("--tx-c", type=int, default=1)
    ap.add_argument("--rx", type=int, default=0)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--eq", type=int, default=0, choices=(0, 1))
    ap.add_argument("--kind", choices=P.KINDS, default="fft_mag")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--n-avg", type=int, default=50)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    index = D.index_signals(D.load_pkl(args.pkl))
    sweep(index, args) if args.sweep else single(index, args)