"""Tarama sonuçlarından model seçimi, nested performans, uçtan uca değerlendirme ve rapor.

  - Final seçim: tüm LOFO skorları üzerinde 1-SE kuralı (en iyiye 1 SE yakın olanlar içinde en basit).
  - Nested tahmin: her dış katta seçim yalnızca iç CV skoruyla yapılır, dışarıdaki ailede test edilir.
  - Uçtan uca: her aile dışarıdayken seçilen modellerle tüm karar ağacı çalıştırılır.
Çıktı: results/secim.json, results/rapor.xlsx
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search import OUT, load_data, tasks

MODEL_RANK = {"ddsimca": 0, "pls": 0, "pca_lda": 0, "ridge": 1, "knn": 2, "krr_rbf": 2, "mlp": 3, "cnn": 4}
SCATTER_RANK = {"none": 0, "snv": 1, "msc": 2, "emsc": 3}
DERIV_RANK = {"none": 0, "sg11": 0.5, "d1w11": 1, "d1w21": 1, "d2w15": 2, "d2w25": 2, "d3w21": 3, "d3w31": 3}
CLASSES = ["otantik", "glikoz", "fruktoz", "hurma", "uzum"]
SIMCA_KMAX = 4   # OC-SIMCA'da fazla PC görülmemiş ailelerin saf elmalarını da reddettiriyor (overfitting)


def load_task(name):
    d = OUT / name
    meta, lofo, inner, sc = [], [], [], []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        meta += json.loads(str(z["meta"]))
        lofo.append(z["lofo"]), inner.append(z["inner"]), sc.append(z["lofo_score"])
    folds = json.loads((d / "folds.json").read_text())
    return pd.DataFrame(meta), np.concatenate(lofo), np.concatenate(inner), np.concatenate(sc), folds


def complexity(m):
    hyp = {"ddsimca": m.get("k", 0), "pls": m.get("k", 0), "pca_lda": m.get("k", 0),
           "ridge": -np.log10(m.get("r", 1) or 1), "knn": 1 / (m.get("k", 1) or 1),
           "krr_rbf": m.get("g", 0) - np.log10(m.get("r", 1) or 1),
           "mlp": m.get("h", 0) - np.log10(m.get("wd", 1) or 1), "cnn": -np.log10(m.get("wd", 1) or 1)}[m["model"]]
    return (MODEL_RANK[m["model"]], round(float(hyp), 6), DERIV_RANK[m["deriv"]], SCATTER_RANK[m["scatter"]])


def se(s, kind, n):
    if kind == "reg":
        return abs(s) / np.sqrt(2 * n)
    p = min(max(s, 0.0), 1.0)
    return np.sqrt(max(p * (1 - p), 0.25 / n) / n)


def pick(scores, meta, kind, n):
    """1-SE: en iyiye 1 SE yakın adaylar içinde en basit (eşitlikte yüksek skor)."""
    s = np.nan_to_num(scores, nan=-np.inf)
    b = int(np.argmax(s))
    if kind == "oc" and s[b] < 0:
        return b
    cand = np.where(s >= s[b] - se(s[b], kind, n))[0]
    return min(cand, key=lambda i: (complexity(meta[i]), -s[i]))


def oc_stats(f, y, mask, orgidx=None):
    acc = f <= 1
    a, d = mask & (y > 0), mask & (y < 0)
    out = dict(ozgulluk=acc[a].mean(), duyarlilik=(~acc[d]).mean())
    if orgidx is not None:
        has = d & (orgidx >= 0)
        org_rows = np.unique(orgidx[orgidx >= 0])
        out["ORG_kabul"] = acc[org_rows].mean()
        out["kosullu_duyarlilik"] = (~acc[has] & acc[orgidx[has]]).mean()
    return out


def sc_free_val(f, y, ev, task):
    return oc_stats(f, y, ev, task["orgidx"])["kosullu_duyarlilik"]


def reg_stats(p, y):
    e = p - y
    return dict(RMSEP=np.sqrt((e ** 2).mean()), bias=e.mean(), R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ba(p, y):
    ok = (p > 0) == (y > 0)
    return np.mean([ok[y > 0].mean(), ok[y < 0].mean()])


def main():
    t, _ = load_data()
    T = tasks(t)
    fam = t.aile.to_numpy()
    lvl = t.oran_yuzde.to_numpy()
    sel, nested_pred, rows_top, rows_grid, perf = {}, {}, [], [], []
    for name, task in T.items():
        meta_df, lofo, inner, sc, folds = load_task(name)
        meta = [{k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))} for r in meta_df.to_dict("records")]
        kind, y, ev = task["kind"], task["y"], task["eval"]
        n_eff = (ev & (y < 0)).sum() if kind == "oc" else ev.sum()
        if kind == "oc":
            n_eff = (ev & (y < 0) & (task["orgidx"] >= 0)).sum()
            free = pick(sc, meta, kind, n_eff)
            kmask = np.array([m_.get("k", 0) <= SIMCA_KMAX for m_ in meta])
            sc = np.where(kmask, sc, -np.inf)
            inner = np.where(kmask[:, None], inner, -np.inf)
            fs = oc_stats(lofo[free], y, ev)
            print(f"{name}: serbest seçim (k sınırı yok): {meta[free]} LOFO {fs} koşullu skor {sc_free_val(lofo[free], y, ev, task):.3f}")
        sel_extra = {}
        # final seçim
        b = pick(sc, meta, kind, n_eff)
        # nested: her dış katta iç skorla seçim
        npred = np.full(len(t), np.nan)
        chosen = []
        tfam = task.get("fam", fam)
        for i, k in enumerate(folds["fams"]):
            j = pick(inner[:, i], meta, kind, n_eff)
            rows = (tfam == k) & ev
            npred[rows] = lofo[j, rows]
            chosen.append(j)
        nested_pred[name] = (npred, [meta[j] for j in chosen], dict(zip(folds["fams"], chosen)))
        m = ev
        if kind == "oc":
            fin, nes = oc_stats(lofo[b], y, m, task["orgidx"]), oc_stats(npred, y, m, task["orgidx"])
        elif kind == "clf":
            fin, nes = dict(dengeli_dogruluk=ba(lofo[b][m], y[m])), dict(dengeli_dogruluk=ba(npred[m], y[m]))
        else:
            fin, nes = reg_stats(lofo[b][m], y[m]), reg_stats(npred[m], y[m])
            ad = m & (y > 0)
            sel_extra = dict(lofo_tagsisli=reg_stats(lofo[b][ad], y[ad]), nested_tagsisli=reg_stats(npred[ad], y[ad]),
                             nested_otantik_ort_tahmin=float(npred[m & (y == 0)].mean()))
        sel[name] = dict(config=meta[b], lofo_score=float(sc[b]), lofo=fin, nested=nes, **sel_extra,
                         nested_secim_cesitliligi=pd.Series([json.dumps(meta[j], sort_keys=True) for j in chosen]).nunique())
        clean = m & ~t["not"].str.contains("Düşük yoğunluk").to_numpy()
        if kind == "oc":
            nes_c = oc_stats(npred, y, clean, task["orgidx"])
        elif kind == "clf":
            nes_c = dict(dengeli_dogruluk=ba(npred[clean], y[clean]))
        else:
            nes_c = reg_stats(npred[clean], y[clean])
        sel[name]["nested_dusuk_yogunluk_haric"] = nes_c
        perf.append(dict(gorev=name, **{f"secilen_{k}": v for k, v in meta[b].items() if k != "task"},
                         **{f"LOFO_{k}": v for k, v in fin.items()}, **{f"nested_{k}": v for k, v in nes.items()},
                         **{f"nested_temiz_{k}": v for k, v in nes_c.items()},
                         **{f"nested_tagsisli_{k}": v for k, v in sel_extra.get("nested_tagsisli", {}).items()},
                         nested_farkli_secim=sel[name]["nested_secim_cesitliligi"], aday_sayisi=len(meta)))
        top = meta_df.assign(lofo_skor=sc).sort_values("lofo_skor", ascending=False).head(25)
        rows_top.append(top)
        g = meta_df.assign(lofo_skor=sc).groupby(["model", "region", "scatter", "deriv"]).lofo_skor.max().reset_index()
        g.insert(0, "task", name)
        rows_grid.append(g)
        print(f"{name:18s} seçilen: {meta[b]['model']:8s} {meta[b]['region']}|{meta[b]['scatter']}|{meta[b]['deriv']}  "
              f"LOFO {fin}  nested {nes}")

    # ---- T1 seviye bazında duyarlılık (nested)
    f1 = nested_pred["T1_otantik"][0]
    lev = pd.DataFrame({"sinif": t.sinif, "oran": lvl, "red": f1 > 1})
    t1_level = lev.pivot_table(index="sinif", columns="oran", values="red", aggfunc="mean").round(3)

    # ---- uçtan uca (nested)
    pred_cls, pred_lvl = [], []
    for i in range(len(t)):
        if f1[i] <= 1:
            pred_cls.append("otantik"), pred_lvl.append(0.0)
            continue
        if t.sinif[i] == "otantik":                     # yanlış alarm: alt düğümler bu aile için değerlendirilmez
            pred_cls.append("tağşişli(?)"), pred_lvl.append(np.nan)
            continue
        s2 = nested_pred["T2_surup_meyve"][0][i]
        if s2 > 0:
            s3 = nested_pred["T3_glikoz_fruktoz"][0][i]
            c = "glikoz" if s3 > 0 else "fruktoz"
            ok = not np.isnan(s3)
        else:
            s4 = nested_pred["T4_hurma_uzum"][0][i]
            c = "hurma" if s4 > 0 else "uzum"
            ok = not np.isnan(s4)
        if not ok:   # yanlış dala gitti; o dalın LOFO tahmini yok (sınıf zaten yanlış)
            c = ("surup" if s2 > 0 else "meyve") + "(yanlış dal)"
        pred_cls.append(c)
        r = nested_pred.get(f"R_{c}", (None,))[0]
        pred_lvl.append(np.clip(r[i], 0, 100) if r is not None and c == t.sinif[i] else np.nan)
    e2e = t[["id", "dosya", "kaynak", "aile", "sinif", "oran_yuzde"]].copy()
    e2e["T1_f_fkrit"] = f1
    e2e["tahmin_sinif"] = pred_cls
    e2e["tahmin_oran"] = np.round(pred_lvl, 2)
    e2e["sinif_dogru"] = e2e.tahmin_sinif == e2e.sinif
    conf = pd.crosstab(e2e.sinif, e2e.tahmin_sinif)
    acc_lvl = e2e.pivot_table(index="sinif", columns="oran_yuzde", values="sinif_dogru", aggfunc="mean").round(3)
    ok = e2e.sinif_dogru & (e2e.sinif != "otantik")
    reg_e2e = e2e[ok].groupby("sinif").apply(
        lambda g: pd.Series(reg_stats(g.tahmin_oran.to_numpy(), g.oran_yuzde.to_numpy())), include_groups=False).round(3)
    lvl_mean = e2e[ok].pivot_table(index="sinif", columns="oran_yuzde", values="tahmin_oran", aggfunc="mean").round(2)
    print("\nUçtan uca doğruluk (seviye):\n", acc_lvl)
    print("\nKarışıklık:\n", conf)

    tbl = lambda d: {str(r): {str(int(c)): (None if pd.isna(v) else float(v)) for c, v in row.items()} for r, row in d.iterrows()}
    sel["_tablolar"] = dict(t1_seviye_red=tbl(t1_level), uctan_uca_dogruluk=tbl(acc_lvl), uctan_uca_ort_oran=tbl(lvl_mean),
                            uctan_uca_regresyon={k: {kk: float(vv) for kk, vv in v.items()} for k, v in reg_e2e.T.to_dict().items()})
    (OUT / "secim.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False, default=float))
    with pd.ExcelWriter(OUT / "rapor.xlsx", engine="openpyxl") as w:
        pd.DataFrame(perf).round(4).to_excel(w, sheet_name="Secilen_modeller", index=False)
        t1_level.to_excel(w, sheet_name="T1_seviye_duyarlilik")
        acc_lvl.to_excel(w, sheet_name="UcUca_dogruluk_seviye")
        conf.to_excel(w, sheet_name="UcUca_karisiklik")
        reg_e2e.to_excel(w, sheet_name="UcUca_regresyon")
        lvl_mean.to_excel(w, sheet_name="UcUca_ort_tahmin_oran")
        e2e.to_excel(w, sheet_name="UcUca_ornekler", index=False)
        pd.concat(rows_top).round(4).to_excel(w, sheet_name="Ilk25_konfig", index=False)
        pd.concat(rows_grid).round(4).to_excel(w, sheet_name="Teknik_x_donusum", index=False)
        for name in T:
            ch = pd.DataFrame(nested_pred[name][1])
            ch.insert(0, "dis_kat_aile", list(nested_pred[name][2]))
            ch.to_excel(w, sheet_name=f"nested_{name}"[:31], index=False)


if __name__ == "__main__":
    main()
