"""Thermo OMNIC .spa okuyucu (tek spektrum)."""
import struct
import numpy as np


def read_spa(path):
    with open(path, "rb") as f:
        b = f.read()
    n_keys = struct.unpack_from("<H", b, 294)[0]
    npts = wmax = wmin = None
    data_off = data_len = None
    for i in range(n_keys):
        pos = 304 + 16 * i
        key = b[pos]
        off = struct.unpack_from("<I", b, pos + 2)[0]
        size = struct.unpack_from("<I", b, pos + 6)[0]
        if key == 2:
            npts = struct.unpack_from("<I", b, off + 4)[0]
            wmax, wmin = struct.unpack_from("<ff", b, off + 16)
        elif key == 3:
            data_off, data_len = off, size
    y = np.frombuffer(b, "<f4", count=npts, offset=data_off).astype(float)
    x = np.linspace(wmax, wmin, npts)
    return x, y
