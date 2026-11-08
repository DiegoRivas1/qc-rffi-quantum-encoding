"""Carga de WiSig (ManySig.pkl) y acceso a capturas por (tx, rx, dia).

Se asume dataset["data"][tx][rx][dia] -> array (N, 256, 2) con I/Q.
El recorrido es tolerante: busca arrays con shape (N, 256, 2) y usa su
ruta de indices como (tx, rx, dia). Si tu pkl difiere, usa scripts/inspect_pkl.py.
"""
import pickle
import numpy as np

SIG_SHAPE = (256, 2)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _walk(obj, path=()):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[1:] == SIG_SHAPE:
            yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (i,))


def index_signals(dataset):
    root = dataset["data"] if isinstance(dataset, dict) and "data" in dataset else dataset
    return {p: a for p, a in _walk(root)}


def available(index):
    """Devuelve (txs, rxs, dias) como listas ordenadas segun las rutas halladas."""
    cols = list(zip(*[p[:3] for p in index if len(p) >= 3]))
    return [sorted(set(c), key=str) for c in cols]


def get_complex(index, tx, rx, day):
    """Capturas (N, 256) complejas para un (tx, rx, dia)."""
    parts = [a for p, a in index.items() if p[:3] == (tx, rx, day)]
    if not parts:
        raise KeyError(f"No hay capturas para tx={tx}, rx={rx}, dia={day}")
    arr = np.concatenate(parts, axis=0).astype(np.float64)
    return arr[..., 0] + 1j * arr[..., 1]
