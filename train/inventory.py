"""Ham veri klasöründeki spektrumları etiketler: parti, sınıf, tağşiş oranı."""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ADULT = {"G": "glikoz", "F": "fruktoz", "H": "hurma", "U": "uzum", "Ü": "uzum"}
# "8287236 -10 F", "8287236-10-F", "9096050-G--6", "8287236-ORG"
PAT_LEVEL_FIRST = re.compile(r"^(?P<batch>.+?)\s*-\s*(?P<lvl>\d+)\s*[- ]\s*(?P<a>[GFHUÜ])$")
PAT_ADULT_FIRST = re.compile(r"^(?P<batch>.+?)-(?P<a>[GFHUÜ])-+(?P<lvl>\d+)$")


def label(path: Path, root: Path):
    stem = unicodedata.normalize("NFC", path.stem).strip()
    rel = path.relative_to(root)
    for pat in (PAT_LEVEL_FIRST, PAT_ADULT_FIRST):
        m = pat.match(stem)
        if m:
            lvl = int(m["lvl"])
            if lvl not in (2, 4, 6, 10, 20, 30):
                break  # ör. "F-904": bilinmeyen kodlama
            return dict(batch=m["batch"].strip(), cls=ADULT[m["a"]], level=lvl)
    if re.search(r"-ORG$", stem):
        return dict(batch=stem[:-4].strip(), cls="otantik", level=0)
    top = rel.parts[0]
    if top.startswith("PBN TANKLAR") or rel.parts[-2] == "Mart":
        return dict(batch=stem, cls="otantik", level=0)
    if re.fullmatch(r"\d{7}", rel.parts[-2]):
        return dict(batch=rel.parts[-2], cls="bilinmiyor", level=None)
    return None


def build(root):
    root = Path(root)
    rows, skipped = {}, []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".tsv", ".spa"):
            continue
        lab = label(p, root)
        if lab is None:
            skipped.append(str(p.relative_to(root)))
            continue
        key = (str(p.parent.relative_to(root)), unicodedata.normalize("NFC", p.stem).lower())
        r = rows.setdefault(key, dict(lab, dir=key[0], name=p.stem))
        r[p.suffix.lower()[1:]] = str(p)
    df = pd.DataFrame(rows.values())
    df["path"] = df["tsv"].fillna(df["spa"]) if "tsv" in df else df["spa"]
    df["origin"] = df["dir"].str.split("/").str[0]
    return df, skipped


if __name__ == "__main__":
    df, skipped = build(sys.argv[1])
    print(df.groupby(["origin", "cls"]).size().unstack(fill_value=0))
    print("\nseviye x sınıf:\n", df.pivot_table(index="level", columns="cls", values="path", aggfunc="count", fill_value=0))
    print("\nparti sayısı:", df[df.cls != "otantik"].batch.nunique())
    print("atlanan:", len(skipped)); print("\n".join(skipped[:20]))
