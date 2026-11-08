"""Preprocesamiento: de captura I/Q (256 complejas) a vector real para codificar.

Idea clave: las capturas del mismo dispositivo difieren en fase de portadora,
asi que usamos magnitudes (invariantes a fase). Ademas se resta una fraccion
alpha del "modo comun" (espectro medio de todos los Tx), que domina por ser el
mismo preambulo WiFi, para dejar la huella propia de cada dispositivo.
"""
import numpy as np

KINDS = ("iq", "amp", "fft_mag")


def normalize(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def features(x_c, kind):
    """x_c: (N, 256) complejo -> (N, D) real, cada fila de norma 1."""
    x_c = normalize(x_c)
    if kind == "iq":          # baseline sensible a fase: D = 512 (9 qubits)
        f = np.concatenate([x_c.real, x_c.imag], axis=-1)
    elif kind == "amp":       # envolvente temporal: D = 256 (8 qubits)
        f = np.abs(x_c)
    elif kind == "fft_mag":   # magnitud del espectro: D = 256 (8 qubits)
        f = np.abs(np.fft.fftshift(np.fft.fft(x_c, axis=-1), axes=-1))
    else:
        raise ValueError(f"kind desconocido: {kind}")
    return normalize(f)


def reference_mean(feats_per_tx):
    """Modo comun: media de las features normalizadas de todos los Tx."""
    return np.concatenate(feats_per_tx, axis=0).mean(axis=0)


def make_datum(feats, ref, alpha):
    """Promedia capturas, resta alpha*modo comun y normaliza a norma 1."""
    v = feats.mean(axis=0) - alpha * ref
    return normalize(v)
