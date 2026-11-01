"""Görev x ön işleme x teknik taraması (GPU), nested leave-one-family-out.

Dış döngü: LOFO (her aile bir kez test). İç döngü: kalan aileler üzerinde 5 katlı aile-gruplu CV.
Her ön işleme için sonuçlar results/<görev>/<ön işleme>.npz olarak yazılır (kesilirse kaldığı yerden devam eder):
  lofo  [C, n]  her örneğin kendi ailesi dışarıdayken aldığı tahmin/skor
  inner [C, O]  her dış kat için iç CV skoru (yüksek = iyi)
  meta  konfigürasyon listesi (json)

Kullanım: python search.py [görev ...] [--cnn-only]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import gpu_models as gm
from gpu_prep import Pipeline, all_pipelines

DEV = "cuda"
DT = torch.float64
OUT = Path(__file__).resolve().parent.parent / "results"
SPEC_MIN = 0.95
CNN_PIPES = [(r, s, d) for r in ("900-1800", "850-1800") for s in ("none", "snv") for d in ("none", "d1w11")]


def load_data():
    z = np.load(Path(__file__).resolve().parent.parent / "data" / "dataset.npz")
    t = pd.DataFrame(json.loads(str(z["meta"])))
    keep = (t.dahil == "Evet").to_numpy()
    return t[keep].reset_index(drop=True), z["X"][keep]


def tasks(t):
    c = t.sinif.to_numpy()
    adult = c != "otantik"
    pm = lambda m: np.where(m, 1.0, -1.0)
    T = {
        "T1_otantik": dict(kind="oc", train=c == "otantik", eval=np.ones(len(t), bool), y=pm(c == "otantik")),
        "T2_surup_meyve": dict(kind="clf", train=adult, eval=adult, y=pm(np.isin(c, ["glikoz", "fruktoz"]))),
        "T3_glikoz_fruktoz": dict(kind="clf", train=np.isin(c, ["glikoz", "fruktoz"]), y=pm(c == "glikoz")),
        "T4_hurma_uzum": dict(kind="clf", train=np.isin(c, ["hurma", "uzum"]), y=pm(c == "hurma")),
    }
    # regresyon: tüm otantikler %0. Yalnız otantik içeren aileler (PBN tank, Mart) LOFO süresini 4 katına çıkardığı için
    # 5 gruba toplanır; bunların tağşişli örneklerle ebeveyn ilişkisi yoktur.
    fam = t.aile.to_numpy().astype(object)
    solo = sorted(set(fam[t.kaynak.isin(["PBN tank", "Mart"]).to_numpy()]))
    grp = {f: f"OTANTIK-G{i % 5 + 1}" for i, f in enumerate(solo)}
    fam_r = np.array([grp.get(f, f) for f in fam])
    for a in ["glikoz", "fruktoz", "hurma", "uzum"]:
        T[f"R_{a}"] = dict(kind="reg", train=(c == a) | (c == "otantik"), y=t.oran_yuzde.to_numpy(float), fam=fam_r)
    for v in T.values():
        v.setdefault("eval", v["train"])
    return T


def make_folds(fam, train, ev, n_inner=5):
    fams = sorted(set(fam[ev]))
    tr, te, outer, inner = [], [], [], []
    for k in fams:
        out_te = fam == k
        outer.append(len(tr))
        tr.append(train & ~out_te), te.append(out_te & ev)
        rest = [f for f in fams if f != k]
        idx = []
        for j in range(n_inner):
            in_te = np.isin(fam, rest[j::n_inner])
            idx.append(len(tr))
            tr.append(train & ~out_te & ~in_te), te.append(in_te & ev)
        inner.append(idx)
    return fams, np.array(tr), np.array(te), np.array(outer), np.array(inner)


def scores(pred, mask, y, kind, ev):
    """pred [O, H, n], mask [O, n] -> skor [O, H] (yüksek = iyi)."""
    m = (mask & ev)[:, None, :]
    if kind == "reg":
        return -np.sqrt((((pred - y) ** 2) * m).sum(-1) / m.sum(-1))
    if kind == "clf":
        ok = (pred > 0) == (y > 0)
        rec = [((ok & m & cl).sum(-1) / np.maximum((m & cl).sum(-1), 1)) for cl in (y > 0, y < 0)]
        return (rec[0] + rec[1]) / 2
    acc = pred <= 1
    spec = (acc & m & (y > 0)).sum(-1) / np.maximum((m & (y > 0)).sum(-1), 1)
    sens = (~acc & m & (y < 0)).sum(-1) / np.maximum((m & (y < 0)).sum(-1), 1)
    return np.where(spec >= SPEC_MIN, sens, spec - 1)


def run_models(pipe, Xt, ytt, Mt, kind, cnn_only):
    """Bir ön işleme + bir kat yığını için tüm teknikler -> [(model, hyp, pred[F,H,n])]."""
    ref = (Mt @ pipe.crop(Xt)) / Mt.sum(1, keepdim=True) if pipe.needs_ref else None
    Z = pipe.apply(Xt, ref)
    res = []
    if cnn_only:
        p, h = gm.cnn(Z, Mt, ytt, kind)
        return [("cnn", h, p)]
    Zc, _ = gm.center(Z, Mt)
    T, lam, _ = gm.pca(Zc, Mt)
    if kind == "oc":
        p, h = gm.dd_simca(T, lam, Zc, Mt)
        return [("ddsimca", h, p)]
    p, h = gm.pls(Zc, Mt, ytt)
    res.append(("pls", h, p))
    p, h = gm.ridge_linear(Zc, Mt, ytt)
    res.append(("ridge", h, p))
    del Zc
    D2 = gm.sqdist(Z)
    p, h = gm.krr_rbf(D2, Mt, ytt)
    res.append(("krr_rbf", h, p))
    p, h = gm.knn(D2, Mt, ytt)
    res.append(("knn", h, p))
    del D2
    if kind == "clf":
        p, h = gm.pca_lda(T, Mt, ytt)
        res.append(("pca_lda", h, p))
    p, h = gm.mlp(T, Mt, ytt, kind)
    res.append(("mlp", h, p))
    return res


def chunk_size(pipe, n, cnn):
    p = pipe.i1 - pipe.i0
    if cnn:
        return max(4, int(2.0e9 / (n * p * 4 * 2 * 40)))
    return max(8, int(1.2e9 / (n * max(p, n) * 8)))


def search_task(name, task, t, X, cnn_only=False):
    fam = task.get("fam", t.aile.to_numpy())
    fams, TR, TE, outer, inner = make_folds(fam, task["train"], task["eval"])
    y, kind, ev = task["y"], task["kind"], task["eval"]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "folds.json").write_text(json.dumps(dict(fams=fams, kind=kind, n=len(t)), ensure_ascii=False))
    Xt = torch.tensor(X, device=DEV, dtype=DT)
    ytt = torch.tensor(y, device=DEV, dtype=DT)
    pipes = CNN_PIPES if cnn_only else all_pipelines()
    if kind == "oc" and cnn_only:
        return
    for r, s, dv in pipes:
        pipe = Pipeline(r, s, dv)
        f = d / (pipe.name.replace("|", "__") + ("__cnn" if cnn_only else "") + ".npz")
        if f.exists():
            continue
        t0 = time.time()
        cs = chunk_size(pipe, len(t), cnn_only)
        parts = []
        for i in range(0, len(TR), cs):
            Mt = torch.tensor(TR[i:i + cs], device=DEV, dtype=DT)
            out = run_models(pipe, Xt, ytt, Mt, kind, cnn_only)
            parts.append([(m, h, p.float().cpu().numpy()) for m, h, p in out])
            del out, Mt
        meta, preds = [], []
        for j, (m, h, _) in enumerate(parts[0]):
            P = np.concatenate([pt[j][2] for pt in parts], 0)          # [F, H, n]
            preds.append(P)
            meta += [dict(task=name, region=r, scatter=s, deriv=dv, model=m, **hh) for hh in h]
        P = np.concatenate(preds, 1)                                    # [F, C, n]
        lofo = (P[outer] * TE[outer][:, None, :]).sum(0)                # [C, n]
        pin = (P[inner] * TE[inner][..., None, :]).sum(1)              # [O, C, n]
        min_ = TE[inner].sum(1).astype(bool)                            # [O, n]
        inner_sc = scores(pin, min_, y, kind, ev).T                     # [C, O]
        lofo_sc = scores(lofo[None], TE[outer].sum(0, keepdims=True).astype(bool), y, kind, ev)[0]
        np.savez_compressed(f, lofo=lofo.astype(np.float32), inner=inner_sc.astype(np.float32),
                            lofo_score=lofo_sc.astype(np.float32), meta=json.dumps(meta))
        best = int(np.argmax(lofo_sc))
        print(f"{name:18s} {pipe.name:24s} C={len(meta):4d} {time.time() - t0:6.1f}s  "
              f"en iyi LOFO={lofo_sc[best]:.3f} ({meta[best]['model']})", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    t, X = load_data()
    T = tasks(t)
    for name in (args or list(T)):
        search_task(name, T[name], t, X, cnn_only="--cnn-only" in sys.argv)
