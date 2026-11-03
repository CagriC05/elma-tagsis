"""Seçilen konfigürasyonları tüm eğitim verisiyle yeniden kurar ve docs/model.json'a yazar.

Her model doğrusal cebirle (numpy) tahmin edilebilecek parametrelere indirgenir; JS tarafı (docs/engine.js)
bunların birebir karşılığıdır. Doğrulama için results/test_ref.json'a örnek spektrumlar ve beklenen çıktılar yazılır.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_coeffs
from scipy.stats import chi2

import gpu_models as gm
from gpu_prep import GRID, Pipeline
from search import DEV, DT, OUT, load_data, tasks

DOCS = Path(__file__).resolve().parent.parent / "docs"
# Arayüzde gösterilen gıda adı ve etiketler (gıda seçimi bu alanlarla yapılır)
PRODUCT = "Elma konsantresi (70 Brix)"
LABELS = {"glikoz": "Glikoz şurubu", "fruktoz": "Fruktoz şurubu", "hurma": "Hurma konsantresi", "uzum": "Üzüm konsantresi"}
NODE_LABELS = {"T1_otantik": "Otantiklik (OC-SIMCA)", "T2_surup_meyve": "Şurup / meyve konsantresi", "T3_glikoz_fruktoz": "Glikoz / fruktoz",
               "T4_hurma_uzum": "Hurma / üzüm", "R_glikoz": "Oran: glikoz", "R_fruktoz": "Oran: fruktoz", "R_hurma": "Oran: hurma",
               "R_uzum": "Oran: üzüm"}
TREE = ["T1_otantik", "T2_surup_meyve", "T3_glikoz_fruktoz", "T4_hurma_uzum",
        "R_glikoz", "R_fruktoz", "R_hurma", "R_uzum"]


def r7(a):
    """JSON boyutu için 10 anlamlı basamak (birbirini götüren büyük katsayılar için yeterli)."""
    a = np.asarray(a, dtype=float)
    return np.vectorize(lambda v: float(f"{v:.10g}"))(a).tolist() if a.size else []


def prep(cfg, Xt, M):
    pipe = Pipeline(cfg["region"], cfg["scatter"], cfg["deriv"])
    ref = (M @ pipe.crop(Xt)) / M.sum(1, keepdim=True) if pipe.needs_ref else None
    Z = pipe.apply(Xt, ref)
    Z = Z[0] if Z.dim() == 3 else Z
    pj = pipe.export()
    pj["ref"] = r7(ref[0].cpu().numpy()) if ref is not None else None
    return Z, pj


def fit(cfg, kind, Xt, yt, tr):
    """Tek model kurar -> (pipeline json, model json, numpy tahmin fonksiyonu girdisi Z, tüm örnek tahmini)."""
    M = torch.tensor(tr[None].astype(float), device=DEV, dtype=DT)
    Z, pj = prep(cfg, Xt, M)
    Zn = Z.cpu().numpy()
    y = yt.cpu().numpy()
    mu = Zn[tr].mean(0)
    Zc = Zn - mu
    m = cfg["model"]
    if m == "ddsimca":
        k, a = int(cfg["k"]), float(cfg["alpha"])
        U, s, Vt = np.linalg.svd(Zc[tr], full_matrices=False)
        P, lam = Vt[:k].T, (s[:k] ** 2) / (tr.sum() - 1)
        Tt = Zc @ P
        h = (Tt ** 2 / lam).sum(1)
        q = ((Zc - Tt @ P.T) ** 2).sum(1)
        h0, q0 = h[tr].mean(), q[tr].mean()
        Nh = float(np.clip(round(2 * h0 ** 2 / h[tr].var(ddof=1)), 1, 500))
        Nq = float(np.clip(round(2 * q0 ** 2 / q[tr].var(ddof=1)), 1, 500))
        fc = chi2.ppf(1 - a, Nh + Nq)
        mj = dict(type="ddsimca", mean=r7(mu), P=r7(P.T), lam=r7(lam), h0=h0, q0=q0, Nh=Nh, Nq=Nq, fcrit=fc, k=k, alpha=a)
        pred = (Nh * h / h0 + Nq * q / q0) / fc
    elif m in ("pls", "ridge", "pca_lda"):
        Mt = M
        Zt = torch.tensor(Zn, device=DEV, dtype=DT)
        Zct, _ = gm.center(Zt, Mt)
        if m == "pls":
            from sklearn.cross_decomposition import PLSRegression
            pl = PLSRegression(int(cfg["k"]), scale=False).fit(Zn[tr], y[tr])
            coef = pl.coef_.ravel()
            b = float(np.ravel(pl.intercept_)[0] - mu @ coef)   # sklearn intercept_ = y ortalaması (X merkezlenir)
        elif m == "ridge":
            from sklearn.linear_model import Ridge
            lam = cfg["r"] * np.mean((Zc[tr] ** 2).sum(1))
            rg = Ridge(alpha=lam).fit(Zn[tr], y[tr])
            coef, b = rg.coef_, float(rg.intercept_)
        else:
            k = int(cfg["k"])
            U, s, Vt = np.linalg.svd(Zc[tr], full_matrices=False)
            P = Vt[:k].T
            Tt = Zc @ P
            pos, neg = tr & (y > 0), tr & (y < 0)
            mp, mn = Tt[pos].mean(0), Tt[neg].mean(0)
            D = Tt[tr] - np.where((y[tr] > 0)[:, None], mp, mn)
            Sw = D.T @ D / (tr.sum() - 2)
            Sw += 1e-8 * np.eye(k) * np.trace(Sw) / k
            w = np.linalg.solve(Sw, mp - mn)
            coef = P @ w
            b = float(-(mu @ coef) - 0.5 * (mp + mn) @ w)
        mj = dict(type="linear", coef=r7(coef), intercept=b)
        pred = Zn @ coef + b
        # GPU arama sürümüyle tutarlılık kontrolü
        ref = {"pls": lambda: gm.pls(Zct, Mt, yt, K=int(cfg["k"]))[0][0, -1],
               "ridge": lambda: gm.ridge_linear(Zct, Mt, yt, ratios=[cfg["r"]])[0][0, 0],
               "pca_lda": lambda: gm.pca_lda(gm.pca(Zct, Mt)[0], Mt, yt, ks=[int(cfg["k"])])[0][0, 0]}[m]()
        dev = np.abs(ref.cpu().numpy() - pred).max() / (np.abs(pred).max() + 1e-12)
        assert dev < 1e-5 or m == "pca_lda" and np.mean(np.sign(ref.cpu().numpy()) == np.sign(pred)) > 0.99, (m, dev)
    elif m == "krr_rbf":
        S = Zn[tr]
        D2 = ((S[:, None] - S[None]) ** 2).sum(-1)
        gamma = cfg["g"] / D2.mean()
        K = np.exp(-gamma * D2)
        ybar = y[tr].mean()
        alpha = np.linalg.solve(K + cfg["r"] * np.eye(len(S)), y[tr] - ybar)
        mj = dict(type="krr_rbf", support=r7(S), alpha=r7(alpha), gamma=gamma, ybar=ybar)
        D2a = ((Zn[:, None] - S[None]) ** 2).sum(-1)
        pred = np.exp(-gamma * D2a) @ alpha + ybar
    elif m == "knn":
        S = Zn[tr]
        mj = dict(type="knn", support=r7(S), y=r7(y[tr]), k=int(cfg["k"]))
        D2a = ((Zn[:, None] - S[None]) ** 2).sum(-1)
        idx = np.argsort(D2a, 1)[:, :int(cfg["k"])]
        pred = y[tr][idx].mean(1)
    elif m == "mlp":
        Zt = torch.tensor(Zn, device=DEV, dtype=DT)
        Zct, _ = gm.center(Zt, M)
        T, lam, P = gm.pca(Zct, M)
        prm = {}
        out, hyp = gm.mlp(T, M, yt, kind, params=prm)
        j = [i for i, h in enumerate(hyp) if h["h"] == cfg["h"] and h["wd"] == cfg["wd"]][0]
        h = int(cfg["h"])
        mj = dict(type="mlp", mean=r7(mu), P=r7(P[0, :, :gm.MLP_PCS].T.cpu().numpy()),
                  tm=r7(prm["tm"][0].cpu().numpy()), tsd=r7(prm["tsd"][0].cpu().numpy()),
                  W1=r7(prm["W1"][j, :, :h].cpu().numpy()), b1=r7(prm["b1"][j, 0, :h].cpu().numpy()),
                  W2=r7(prm["W2"][j, :h, 0].cpu().numpy()), b2=float(prm["b2"][j, 0, 0]),
                  ym=float(prm["ym"][0]) if kind == "reg" else None, ysd=float(prm["ysd"][0]) if kind == "reg" else None)
        pred = out[0, j].cpu().numpy()
    elif m == "cnn":
        prm = {}
        out, hyp = gm.cnn(Z, M, yt, kind, params=prm)
        j = [i for i, h in enumerate(hyp) if h["wd"] == cfg["wd"]][0]
        g = lambda v: r7(v[j].cpu().numpy())
        mj = dict(type="cnn", mu=r7(prm["mu"][0].cpu().numpy()), sd=float(prm["sd"][0]),
                  conv1_w=g(prm["conv1_w"]), conv1_b=g(prm["conv1_b"]), conv2_w=g(prm["conv2_w"]),
                  conv2_b=g(prm["conv2_b"]), W=g(prm["W"]), b=float(prm["b"][j]), pool=prm["pool"], stride=2,
                  ym=float(prm["ym"][0]) if kind == "reg" else None, ysd=float(prm["ysd"][0]) if kind == "reg" else None)
        pred = out[0, j].cpu().numpy()
    else:
        raise ValueError(m)
    return pj, mj, pred


QC_KERNEL = savgol_coeffs(11, 3, use="conv")


def spectral_metrics(X):
    """docs/engine.js spectralMetrics ile aynı tanımlar (tam ızgara, 2 cm-1)."""
    X = np.atleast_2d(X)
    ix = lambda w: int(np.argmin(np.abs(GRID - w)))
    i0, i1 = int(np.searchsorted(GRID, 900)), int(np.searchsorted(GRID, 1200, side="right"))
    sm = np.array([np.convolve(r[i0 - 5:i1 + 5], QC_KERNEL, mode="valid") for r in X])
    base = X[:, ix(3900)]
    return dict(noise=(X[:, i0:i1] - sm).std(1) * 1e3, base=base, oh=X[:, ix(3300)] - base,
                fp=X[:, i0:i1].max(1) - base, minA=(X[:, ix(850):ix(3700)] - base[:, None]).min(1))


def qc_profile(t, X):
    """Kabul kriterleri: SPA ölçüm parametreleri eğitim profilinden, spektral metrikler eğitim dağılımından."""
    m = spectral_metrics(X)
    med = lambda v: float(np.median(v))
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    spec = dict(
        noise=dict(warn=float(np.quantile(m["noise"], 0.9)), reject=float(m["noise"].max() * 1.25),
                   label="Gürültü (900–1200 cm⁻¹, ×1000)"),
        oh=dict(warn_lo=med(m["oh"]) - 6 * mad(m["oh"]), warn_hi=med(m["oh"]) + 6 * mad(m["oh"]), lo=0.5 * med(m["oh"]),
                label="O–H bandı yüksekliği (3300 − 3900 cm⁻¹); Brix / kristal teması"),
        fp=dict(lo=med(m["fp"]) - 6 * mad(m["fp"]), warn_hi=med(m["fp"]) + 6 * mad(m["fp"]),
                label="Parmak izi tepe yüksekliği (900–1200 cm⁻¹); kristal teması"),
        base=dict(abs_max=float(np.abs(m["base"]).max() * 1.25), label="Baz çizgisi kayması (3900 cm⁻¹)"),
        minA=dict(lo=float(m["minA"].min() - 0.05), label="En düşük absorbans (baz düzeltilmiş); arka plan hatası"),
    )
    sp = t[t.spa_scans.notna()]
    acq = dict(units=sorted(set(sp.spa_units)), resolution=sorted(set(sp.spa_resolution.astype(float))),
               scans_min=int(sp.spa_scans.min()), scans_typ=int(sp.spa_scans.mode()[0]),
               bg_scans_min=int(sp.spa_bg_scans.min()), accessory=sorted(set(sp.spa_accessory.dropna())),
               serial=sorted(set(sp.spa_serial.dropna())), bg_age_max=float(np.ceil(sp.spa_bg_age_min.max() / 5) * 5),
               xmin_max=850.0, xmax_min=3700.0, period=[sp.spa_collected.min()[:10], sp.spa_collected.max()[:10]])
    # eğitim verisinde kaç spektrum reddedilirdi (bilgi)
    rej = (m["noise"] > spec["noise"]["reject"]) | (m["oh"] < spec["oh"]["lo"]) | \
          (m["fp"] < spec["fp"]["lo"]) | (np.abs(m["base"]) > spec["base"]["abs_max"]) | (m["minA"] < spec["minA"]["lo"])
    return dict(acquisition=acq, spectral=spec, train_rejected=int(rej.sum()), train_n=len(t))


def find_config(meta, cfg):
    ok = np.ones(len(meta), bool)
    for k, v in cfg.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            ok &= np.isclose(meta[k].astype(float).to_numpy(), v, equal_nan=False)
        elif isinstance(v, str):
            ok &= (meta[k] == v).to_numpy()
    return int(np.where(ok)[0][0])


def confidence_threshold(score, y, mask):
    """|skor| eşiği: altında LOFO doğruluğu %80'in altında kalır (|skor| üzerinde artan izotonik kalibrasyon)."""
    from sklearn.isotonic import IsotonicRegression
    s, ok = np.abs(score[mask]), ((score[mask] > 0) == (y[mask] > 0)).astype(float)
    o = np.argsort(s)
    fit = IsotonicRegression(increasing=True).fit_transform(s[o], ok[o])
    if fit[0] >= 0.8:
        return 0.0
    above = np.where(fit >= 0.8)[0]
    return float(s[o][above[0]]) if len(above) else 1e9


def main():
    t, X = load_data()
    T = tasks(t)
    sel = json.loads((OUT / "secim.json").read_text())
    Xt = torch.tensor(X, device=DEV, dtype=DT)
    out = dict(product=PRODUCT, labels=LABELS, node_labels=NODE_LABELS, version=datetime.date.today().isoformat(), grid=dict(start=float(GRID[0]), step=float(GRID[1] - GRID[0]), n=len(GRID)),
               n_train=dict(otantik=int((t.sinif == "otantik").sum()), tagsisli=int((t.sinif != "otantik").sum()),
                            aile=int(t.aile.nunique())), tasks={})
    preds = {}
    for name in TREE:
        task, s = T[name], sel[name]
        cfg = s["config"]
        yt = torch.tensor(task["y"], device=DEV, dtype=DT)
        pj, mj, pred = fit(cfg, task["kind"], Xt, yt, task["train"])
        if task["kind"] == "oc" and "threshold" in s:   # ampirik eşik: f/f_kritik ≤ 1 kuralı korunur
            mj["fcrit"] *= s["threshold"]
            mj["threshold_scale"] = s["threshold"]
            mj["suspect"] = s.get("suspect")   # bu değerin üstü (≤ 1) "şüpheli"
            mj["spec"], mj["suspect_spec"] = s.get("spec"), s.get("suspect_spec")
            pred = pred / s["threshold"]
        preds[name] = pred
        entry = dict(kind=task["kind"], config={k: v for k, v in cfg.items() if k != "task"}, pipeline=pj, model=mj,
                     perf=dict(lofo=s["lofo"], nested=s["nested"], **({"nested_tagsisli": s["nested_tagsisli"]} if "nested_tagsisli" in s else {})))
        if task["kind"] == "clf":
            from select_models import load_task
            meta, lofo, _, _, _ = load_task(name)
            i = find_config(meta, cfg)
            entry["conf_threshold"] = confidence_threshold(lofo[i], task["y"], task["eval"])
        if task["kind"] == "reg":
            entry["interval"] = 1.96 * s["nested"]["RMSEP"]
        out["tasks"][name] = entry
        print(f"{name:18s} {cfg['model']:8s} {pj['region']}|{pj['scatter']}|{pj['deriv']}  "
              f"{ {k: round(v, 3) for k, v in s['nested'].items()} }  "
              + (f"güven eşiği={entry.get('conf_threshold', 0):.3f}" if 'conf_threshold' in entry else ""))

    # kapsam kontrolü: T1 ön işlemesiyle tüm eğitim örneklerinde DD-SIMCA (alpha=0.001)
    c1 = dict(sel["T1_otantik"]["config"], alpha=0.001)
    allrows = np.ones(len(t), bool)
    pj, mj, pd_ = fit(c1, "oc", Xt, torch.zeros(len(t), device=DEV, dtype=DT), allrows)
    out["domain"] = dict(pipeline=pj, model=mj)
    preds["domain"] = pd_
    print("kapsam modeli: eğitim örneklerinin", round(float((pd_ <= 1).mean()), 3), "kadarı içeride")

    out["qc"] = qc_profile(t, X)
    print("kabul kriterleri: eğitim spektrumlarından", out["qc"]["train_rejected"], "/", len(t), "spektral kontrolde reddedilirdi")
    out["ref_spectrum"] = dict(otantik=r7(X[t.sinif == "otantik"].mean(0)[::2]), step=2 * float(GRID[1] - GRID[0]))
    out["validation"] = sel["_tablolar"]

    DOCS.mkdir(exist_ok=True)
    (DOCS / "model.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=float, allow_nan=False))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(t), 12, replace=False)
    sm_ = spectral_metrics(X[idx])
    ref = dict(ids=t.id.iloc[idx].tolist(), X=r7(X[idx]), expected={k: r7(v[idx]) for k, v in preds.items()},
               metrics={k: r7(v) for k, v in sm_.items()})
    (OUT / "test_ref.json").write_text(json.dumps(ref, separators=(",", ":")))   # yayınlanmaz (results/ depoda değil)
    print("model.json:", round((DOCS / "model.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
