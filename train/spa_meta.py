"""OMNIC .spa başlığından ölçüm parametreleri (docs/engine.js readSPAMeta ile aynı alanlar)."""
import re
import struct
from datetime import datetime

UNITS = {17: "Absorbance", 16: "%Transmittance", 11: "%Reflectance", 12: "Log(1/R)", 3: "Single beam", 15: "Interferogram"}


def _text(b, off, size):
    return re.sub(r"[\x00-\x08\x0b-\x1f]+", "\n", b[off:off + size].decode("latin1", "replace"))


def _when(s):
    m = re.search(r"(\w+ \d{1,2}, \d{4} \d{1,2}:\d{2}:\d{2} [AP]M)", s or "")
    return datetime.strptime(m.group(1), "%B %d, %Y %I:%M:%S %p") if m else None


def read_spa_meta(path):
    b = open(path, "rb").read()
    n = struct.unpack_from("<H", b, 294)[0]
    meta, hist, acc = {}, "", ""
    for i in range(n):
        pos = 304 + 16 * i
        key = b[pos]
        off, size = struct.unpack_from("<II", b, pos + 2)
        if key == 2:
            meta["npts"] = struct.unpack_from("<I", b, off + 4)[0]
            meta["units"] = UNITS.get(struct.unpack_from("<I", b, off + 12)[0], str(struct.unpack_from("<I", b, off + 12)[0]))
            meta["xmax"], meta["xmin"] = struct.unpack_from("<ff", b, off + 16)
            meta["scans"] = struct.unpack_from("<I", b, off + 36)[0]
            meta["bg_scans"] = struct.unpack_from("<I", b, off + 52)[0]
        elif key == 27:
            hist = _text(b, off, size)
        elif key == 151:
            acc = _text(b, off, size)
    m = re.search(r"Resolution:\s*([\d.]+)", hist)
    meta["resolution"] = float(m.group(1)) if m else None
    m = re.search(r"serial number:\s*(\S+)", hist)
    meta["serial"] = m.group(1) if m else None
    m = re.search(r"Final format:\s*(.+)", hist)
    meta["final_format"] = m.group(1).strip() if m else None
    ts, tb = _when(re.search(r"Collect Sample on (.+)", hist).group(1)) if "Collect Sample on" in hist else None, \
        _when(re.search(r"Background collected on (.+)", hist).group(1)) if "Background collected on" in hist else None
    meta["collected"] = ts.isoformat() if ts else None
    meta["bg_age_min"] = round((ts - tb).total_seconds() / 60, 1) if ts and tb else None
    meta["accessory"] = next((l.strip() for l in acc.split("\n") if "ATR" in l or "Smart" in l), None)
    return meta
