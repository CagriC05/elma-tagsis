"""2025 veri setini temizleyip etiketler; dataset.xlsx + dataset.npz yazar.

Kullanım: python build_dataset.py "<Elma Otantikasyonu klasörü>" <çıktı klasörü>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from inventory import build
from load import GRID, read_any

GROUP = {"otantik": "otantik", "glikoz": "surup", "fruktoz": "surup", "hurma": "meyve", "uzum": "meyve"}
CODE = {"otantik": "A", "glikoz": "G", "fruktoz": "F", "hurma": "H", "uzum": "U"}


def source(d):
    if d.startswith("PBN"):
        return "PBN tank"
    if d.endswith("/Mart"):
        return "Mart"
    return "Parti serisi"


def family(r):
    """Leave-one-family-out grubu: aynı ebeveyn elmadan gelen tüm örnekler aynı ailede."""
    if r.kaynak == "PBN tank":
        return re.match(r"(?i)st\d+", r.dosya).group(0).upper()
    if r.kaynak == "Mart":
        return "M-" + r.dosya.split("-")[0]
    return r.parti


def tarih(name):
    m = re.search(r"(\d{2})[- ]?(\d{2})[- ]?(2025|25)(?!\d)", name)
    return f"2025-{m[2]}-{m[1]}" if m else ""


def main(root, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    df, _ = build(root)
    df = df[df.origin.str.contains("2025")].reset_index(drop=True)
    X = np.array([np.interp(GRID, *read_any(p)) for p in df.path])

    t = pd.DataFrame({
        "dosya": df.name.str.strip(),
        "kaynak": df.dir.map(source),
        "parti": np.where(df.dir.str.match(r".*/\d{7}$"), df.dir.str.split("/").str[-1], df.batch),
        "sinif": df.cls,
        "oran_yuzde": df.level,
    })
    t["aile"] = t.apply(family, axis=1)
    t["tarih"] = t.dosya.map(tarih)
    t["ust_grup"] = t.sinif.map(GROUP).fillna("")
    t["tagsisli"] = np.where(t.sinif == "otantik", 0, 1)
    t["olcum_araligi"] = [f"{read_any(p)[0][0]:.0f}-4000" for p in df.path]

    # QC metrikleri (bilgi amaçlı; dışlama sadece açık hatalarda)
    fp = (GRID >= 900) & (GRID <= 1200)
    med = np.median(X, 0)
    t["qc_gurultu"] = ((X[:, fp] - savgol_filter(X[:, fp], 11, 3, axis=1)).std(1) * 1e3).round(2)
    t["qc_tepe_900_1200"] = X[:, fp].max(1).round(3)
    t["qc_A3300"] = X[:, np.argmin(abs(GRID - 3300))].round(3)
    t["qc_baz_3900"] = X[:, np.argmin(abs(GRID - 3900))].round(3)
    t["qc_medyan_korelasyon"] = [round(np.corrcoef(x, med)[0, 1], 4) for x in X]

    notes = [[] for _ in range(len(t))]
    keep = np.ones(len(t), bool)
    # birebir aynı spektrumlar
    for i in range(len(X)):
        for j in range(i):
            if keep[j] and np.allclose(X[i], X[j], rtol=0, atol=1e-5):
                keep[i] = False
                notes[i].append(f"KOPYA: '{t.dosya[j]}' ile birebir aynı")
    for i, r in t.iterrows():
        if r.sinif == "bilinmiyor":
            keep[i] = False
            notes[i].append("Dosya kodu çözülemedi (F-904; F--4 olabilir)")
        if r.qc_medyan_korelasyon < 0:
            keep[i] = False
            notes[i].append("Başarısız ölçüm (ters/negatif spektrum)")
        if r.qc_gurultu > 2.5:
            notes[i].append("Yüksek gürültü")
        if r.qc_A3300 < 0.2 and r.qc_medyan_korelasyon > 0:
            notes[i].append("Düşük yoğunluk / zayıf kristal teması")
        if abs(r.qc_baz_3900) > 0.05:
            notes[i].append("Baz kayması")
    t["dahil"] = np.where(keep, "Evet", "Hayır")
    t["not"] = ["; ".join(n) for n in notes]
    t.insert(0, "id", [f"S{i + 1:03d}" for i in range(len(t))])
    t["dosya_yolu"] = [str(Path(p).relative_to(root)) for p in df.path]

    order = t.assign(_k=t.kaynak.map({"Parti serisi": 0, "PBN tank": 1, "Mart": 2}),
                     _c=t.sinif.map({"otantik": 0, "glikoz": 1, "fruktoz": 2, "hurma": 3, "uzum": 4}).fillna(9))
    idx = order.sort_values(["_k", "parti", "_c", "oran_yuzde"]).index
    t, X = t.loc[idx].reset_index(drop=True), X[idx]
    t["id"] = [f"S{i + 1:03d}" for i in range(len(t))]

    np.savez_compressed(out / "dataset.npz", X=X, wn=GRID, meta=t.to_json(orient="records", force_ascii=False))

    ok = t[t.dahil == "Evet"]
    ozet = pd.concat([
        ok.pivot_table(index="sinif", columns="oran_yuzde", values="id", aggfunc="count", fill_value=0, margins=True, margins_name="Toplam"),
    ])
    aile = ok.groupby("kaynak").agg(ornek=("id", "count"), aile=("aile", "nunique"))
    spek = pd.concat([t[["id", "dosya", "sinif", "oran_yuzde", "aile", "dahil"]],
                      pd.DataFrame(X.round(5), columns=[f"{w:g}" for w in GRID])], axis=1)

    aciklama = pd.DataFrame([
        ("id", "Örnek kimliği"),
        ("dosya", "Ham dosya adı (uzantısız)"),
        ("kaynak", "Parti serisi (laboratuvarda hazırlanan karışımlar + ORG), PBN tank, Mart (otantik referanslar)"),
        ("parti", "Parti/numune kodu (parti serilerinde klasör adı)"),
        ("sinif", "otantik / glikoz / fruktoz / hurma / uzum"),
        ("oran_yuzde", "Tağşiş oranı (%); otantik = 0"),
        ("aile", "Leave-one-family-out grubu: aynı ebeveyn elmadan türeyen örnekler (parti no; PBN için tank no; Mart için örnek kodu)"),
        ("tarih", "Dosya adındaki ölçüm/örnekleme tarihi (varsa)"),
        ("ust_grup", "Karar ağacı 2. düğüm: surup / meyve"),
        ("tagsisli", "0 = otantik, 1 = tağşişli"),
        ("olcum_araligi", "Ham dosyanın spektral aralığı (cm-1); tümü 800-4000'e indirildi"),
        ("qc_gurultu", "900-1200 cm-1 yüksek frekanslı kalıntı (x1000); normal ~1.6"),
        ("qc_tepe_900_1200", "Parmak izi bölgesi maksimum absorbans; normal ~0.50-0.53"),
        ("qc_A3300", "O-H bandı absorbansı (su/Brix vekili); normal ~0.35-0.40"),
        ("qc_baz_3900", "3900 cm-1 baz çizgisi; normal ~0"),
        ("qc_medyan_korelasyon", "Tüm 2025 medyan spektrumuyla korelasyon"),
        ("dahil", "Modellemeye girecek mi (yalnızca kopya, kodu çözülemeyen ve başarısız ölçümler hariç)"),
        ("not", "QC uyarıları (bilgi amaçlı; 'Hayır' olmayanlar modele girer)"),
        ("Spektrumlar sayfası", "Absorbans, 800-4000 cm-1, 2 cm-1 adım (ham 0.48 cm-1 ızgaradan doğrusal interpolasyon)"),
    ], columns=["sütun", "açıklama"])

    with pd.ExcelWriter(out / "dataset_2025.xlsx", engine="openpyxl") as w:
        t.to_excel(w, sheet_name="Ornekler", index=False)
        spek.to_excel(w, sheet_name="Spektrumlar", index=False)
        ozet.to_excel(w, sheet_name="Ozet", startrow=1)
        aile.to_excel(w, sheet_name="Ozet", startrow=len(ozet) + 5)
        aciklama.to_excel(w, sheet_name="Aciklama", index=False)
        for ws in (w.sheets["Ornekler"],):
            ws.freeze_panes = "C2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                n = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, n + 2), 60)
        w.sheets["Spektrumlar"].freeze_panes = "G2"
        w.sheets["Aciklama"].column_dimensions["A"].width = 22
        w.sheets["Aciklama"].column_dimensions["B"].width = 110
        w.sheets["Ozet"]["A1"] = "Modele giren örnekler: sınıf x tağşiş oranı"
        w.sheets["Ozet"].cell(len(ozet) + 5, 1, "Kaynak başına örnek ve aile (LOFO grubu) sayısı")
    print(t.dahil.value_counts().to_dict())
    print(ozet)
    print(aile)
    print(t[t.dahil == "Hayır"][["id", "dosya", "not"]].to_string())
    print(t["not"].str.split("; ").explode().value_counts())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
