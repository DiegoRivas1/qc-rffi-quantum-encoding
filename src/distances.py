"""Medidas de similitud entre estados puros (NumPy puro, sin frameworks)."""
import numpy as np


def inner_product(a, b):
    """<A|B>."""
    return np.vdot(a, b)


def expectation_fidelity(a, b):
    """<A| (|B><B|) |A> = |<A|B>|^2 : esperanza del proyector sobre B."""
    proj = np.outer(b, np.conj(b))
    return float(np.real(np.vdot(a, proj @ a)))


def fidelity(a, b):
    """Equivalente a qiskit.quantum_info.state_fidelity para estados puros."""
    return float(abs(np.vdot(a, b)) ** 2)


def swap_test(a, b, shots=8192, rng=None):
    """Swap test: P(ancilla=0) = (1 + |<A|B>|^2)/2. Devuelve (F_estimada, P0_est)."""
    rng = rng or np.random.default_rng()
    p0 = (1.0 + fidelity(a, b)) / 2.0
    p0_est = rng.binomial(shots, p0) / shots
    return max(0.0, 2.0 * p0_est - 1.0), p0_est


def born(a):
    p = np.abs(a) ** 2
    return p / p.sum()


def kl(a, b, eps=1e-12):
    """KL(P||Q) entre distribuciones de Born (divergencia, no similitud)."""
    p, q = born(a) + eps, born(b) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def js_similarity(a, b):
    """1 - Jensen-Shannon (base 2) en [0,1]: version simetrica y acotada de KL."""
    p, q = born(a), born(b)
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return np.sum(x[mask] * np.log2(x[mask] / y[mask]))

    return float(1.0 - 0.5 * (_kl(p, m) + _kl(q, m)))
