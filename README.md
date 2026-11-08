# qc-rffi-quantum-encoding

Examen parcial (Trabajo Interdisciplinario): representar señales de mi tesis de
**RF Fingerprinting con Vision Transformers** (WiSig) como estados cuánticos y
medir similitud entre ellos.

- **Dato A y B:** el mismo transmisor (capturas disjuntas).
- **Dato C:** otro transmisor.
- **Meta:** F(A,B) ≈ 0.9 y F(A,C) ≈ F(B,C) ≈ 0.3.

## Idea

| | Clásico | Cuántico |
|---|---|---|
| Representación | 256 muestras I/Q cuantizadas a 8 bits (bitstring) | *Amplitude encoding*: 256 amplitudes en **8 qubits** |
| Similitud | distancia de Hamming | `|<A\|B>|²`, Swap Test, KL/JS |

El preprocesamiento (`src/preprocessing.py`) es lo que logra la dispersión:
magnitud del espectro (invariante a la fase de portadora), promedio de varias
capturas y resta de una fracción `alpha` del espectro medio de todos los Tx
(el preámbulo común).

## Estructura

```
src/            data.py · preprocessing.py · encoding.py · distances.py   (NumPy puro)
scripts/        inspect_pkl.py · run_experiment.py · qiskit_swap_test.py
data/           ManySig.pkl (no se versiona)
results/        CSV/PNG generados (no se versiona)
report/         informe LaTeX (3 páginas)
```

## Uso

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

python scripts/inspect_pkl.py data/ManySig.pkl      # ver estructura (tx, rx, dia)
python scripts/run_experiment.py --pkl data/ManySig.pkl --sweep
python scripts/run_experiment.py --pkl data/ManySig.pkl --tx-a 0 --tx-c 1 --rx 0 --day 0 \
       --kind fft_mag --alpha 0.95 --n-avg 10 --plot
python scripts/qiskit_swap_test.py --pkl data/ManySig.pkl --kind fft_mag --alpha 0.95 --n-avg 10
```

`--sweep` prueba `kind × alpha × n_avg` y ordena por cercanía al objetivo.
Se asume `dataset["data"][tx][rx][dia] -> (N, 256, 2)`; si tu pkl difiere,
mira la salida de `inspect_pkl.py`.
