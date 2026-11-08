"""Imprime la estructura del pkl: claves, rutas (tx, rx, dia) y shapes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D

path = sys.argv[1] if len(sys.argv) > 1 else "data/ManySig.pkl"
ds = D.load_pkl(path)
print("tipo:", type(ds).__name__)
if isinstance(ds, dict):
    print("claves:", list(ds.keys()))
idx = D.index_signals(ds)
print("arrays (N,256,2) encontrados:", len(idx))
for p, a in list(idx.items())[:10]:
    print(" ", p, a.shape, a.dtype)
txs, rxs, days = D.available(idx)
print("tx:", txs, "\nrx:", rxs, "\ndias:", days)
