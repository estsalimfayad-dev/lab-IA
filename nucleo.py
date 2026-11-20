"""Núcleo sin interfaz: LSTM en NumPy puro (sin TensorFlow) + ventanas deslizantes.
Equivale a mejor_lstm_v4.keras (LSTM(64) + Dense sigmoide), verificado contra Keras."""
import os, csv
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
FS, WIN_S, STRIDE_S, THR = 25, 3.0, 0.5, 0.5
T = int(FS * WIN_S)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class ModeloLSTM:
    def __init__(self, ruta=None):
        p = np.load(ruta or os.path.join(D, "modelo_lstm.npz"))
        self.Wk, self.Wr, self.b, self.Wd, self.bd = (p[k].astype("float32") for k in ("Wk", "Wr", "b", "Wd", "bd"))
        self.h = self.Wr.shape[0]

    def predecir(self, X):
        """X: (n, T, 3) ya preprocesado -> (n,) probabilidad de caída. Orden de puertas Keras: i, f, c, o."""
        n, h = X.shape[0], self.h
        ht = np.zeros((n, h), "float32"); ct = np.zeros((n, h), "float32")
        for t in range(X.shape[1]):
            z = X[:, t] @ self.Wk + ht @ self.Wr + self.b
            i, f, g, o = z[:, :h], z[:, h:2 * h], z[:, 2 * h:3 * h], z[:, 3 * h:]
            ct = _sig(f) * ct + _sig(i) * np.tanh(g)
            ht = _sig(o) * np.tanh(ct)
        return _sig(ht @ self.Wd + self.bd).ravel()


def preprocesar(X):
    """Misma normalización del entrenamiento v4: signo*log1p(|x|/0.05), cada ventana / su máximo (piso 1.0)."""
    Z = np.sign(X) * np.log1p(np.abs(X) / 0.05)
    return (Z / np.maximum(np.abs(Z).max(axis=(1, 2), keepdims=True), 1.0)).astype("float32")


def ventana(t, xyz, t0):
    """Remuestrea [t0, t0+3 s) a 25 Hz (75 pasos), igual que el entrenamiento. None si hay pocas muestras."""
    m = (t >= t0) & (t < t0 + WIN_S)
    if m.sum() <= 10:
        return None
    grid = np.arange(T) / FS
    return np.stack([np.interp(grid, t[m] - t0, xyz[m, k], left=0, right=0) for k in range(3)], 1)


def clasificar(t, xyz, modelo):
    """Ventanas deslizantes sobre un registro completo -> lista de (inicio, fin, prob)."""
    t = np.asarray(t, float); xyz = np.asarray(xyz, float)
    t0, filas, X = t.min(), [], []
    while t0 + WIN_S <= t.max() + 1e-9:
        w = ventana(t, xyz, t0)
        if w is not None:
            X.append(w); filas.append((t0, t0 + WIN_S))
        t0 += STRIDE_S
    if not X:
        return []
    p = modelo.predecir(preprocesar(np.array(X, "float32")))
    return [(a, b, float(q)) for (a, b), q in zip(filas, p)]


def eventos(t, xyz, res, sep=3.0):
    """Igual que la interfaz original: de las ventanas con prob>=THR toma el pico de |a| y deja un
    evento por cada `sep` s (el de mayor pico). Devuelve [(inicio, fin, prob)]."""
    t = np.asarray(t, float); a = np.linalg.norm(np.asarray(xyz, float), axis=1)
    pk = []
    for ini, fin, p in res:
        if p >= THR:
            m = (t >= ini) & (t < fin)
            if m.any():
                i = np.flatnonzero(m)[a[m].argmax()]; pk.append((t[i], a[i], p))
    pk.sort(key=lambda x: -x[1]); keep = []
    for tt, aa, p in pk:
        if all(abs(tt - k[0]) >= sep for k in keep):
            keep.append((tt, aa, p))
    return [(tt - 1.0, tt + 1.0, p) for tt, aa, p in sorted(keep)]


def leer_csv(ruta):
    """CSV exportado por Phyphox (Acceleration without g): columnas t, x, y, z, |a|; delimitador , o ;"""
    filas = []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        txt = f.read()
    delim = ";" if txt.count(";") > txt.count(",") else ","
    for r in csv.reader(txt.splitlines(), delimiter=delim):
        try:
            filas.append([float(v.replace(",", ".")) for v in r[:4]])
        except (ValueError, IndexError):
            continue  # encabezados
    a = np.array(filas)
    if a.shape[0] < 20:
        raise ValueError("El CSV no tiene datos numéricos suficientes (t, x, y, z).")
    return a[:, 0], a[:, 1:4]
