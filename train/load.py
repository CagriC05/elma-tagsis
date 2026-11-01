"""Spektrumları ortak dalga sayısı ızgarasına alır ve önbelleğe yazar."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inventory import build
from spa import read_spa

GRID = np.round(np.arange(800.0, 4000.0 + 1e-9, 2.0), 1)  # cm-1, 2 cm-1 adım


def read_any(path):
    if path.lower().endswith(".spa"):
        x, y = read_spa(path)
    else:
        a = pd.read_csv(path, sep=None, engine="python").to_numpy(float)
        x, y = a[:, 0], a[:, 1]
    o = np.argsort(x)
    return x[o], y[o]


def load(root, cache="cache.npz"):
    if Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return pd.DataFrame(z["meta"].tolist()), z["X"]
    df, _ = build(root)
    X, info = [], []
    for p in df.path:
        x, y = read_any(p)
        info.append((x[0], x[-1], len(x)))
        X.append(np.interp(GRID, x, y))
    df[["xmin", "xmax", "npts"]] = info
    X = np.array(X)
    np.savez(cache, X=X, meta=np.array(df.to_dict("records"), dtype=object))
    return df, X


if __name__ == "__main__":
    df, X = load(sys.argv[1])
    print(df[["xmin", "xmax", "npts"]].round(1).value_counts())
    print(X.shape)
