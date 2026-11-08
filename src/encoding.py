"""Representacion clasica (bitstrings) y cuantica (amplitude encoding)."""
import numpy as np


def n_qubits(dim):
    return int(np.ceil(np.log2(dim)))


def amplitude_encode(v):
    """Vector real (D,) -> amplitudes de un estado de n qubits (relleno a 2^n)."""
    n = n_qubits(len(v))
    psi = np.zeros(2 ** n)
    psi[: len(v)] = v
    return psi / np.linalg.norm(psi)


def to_bitstring(v, vmax, bits=8):
    """Cuantiza cada muestra a `bits` bits en [-vmax, vmax] y concatena."""
    levels = 2 ** bits - 1
    q = np.round((np.clip(v, -vmax, vmax) + vmax) / (2 * vmax) * levels).astype(int)
    return "".join(format(int(x), f"0{bits}b") for x in q)


def hamming_similarity(s1, s2):
    d = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return 1.0 - d / len(s1)
