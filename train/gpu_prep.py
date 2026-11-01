"""GPU ön işleme: bölge kırpma -> saçılma düzeltmesi -> Savitzky-Golay.

Sıra sabittir ve uygulamadaki JS ile birebir aynıdır:
  1) [lo - h, hi + h] aralığını kırp (h = SG yarı pencere; SG 'valid' konvolüsyonu kenarı yer)
  2) saçılma düzeltmesi: none | snv | msc | emsc (MSC/EMSC referansı = eğitim katının ortalaması)
  3) SG düzleştirme/türev ('valid')
"""
import numpy as np
import torch
from scipy.signal import savgol_coeffs

GRID = np.round(np.arange(800.0, 4000.0 + 1e-9, 2.0), 1)

REGIONS = {"900-1500": (900, 1500), "900-1800": (900, 1800), "850-1800": (850, 1800), "850-3700": (850, 3700)}
SCATTERS = ["none", "snv", "msc", "emsc"]
# ad: (pencere, polinom, türev)
DERIVS = {"none": (0, 0, 0), "sg11": (11, 2, 0), "d1w11": (11, 2, 1), "d1w21": (21, 2, 1),
          "d2w15": (15, 2, 2), "d2w25": (25, 2, 2), "d3w21": (21, 3, 3), "d3w31": (31, 3, 3)}


def all_pipelines():
    return [(r, s, d) for r in REGIONS for s in SCATTERS for d in DERIVS]


class Pipeline:
    def __init__(self, region, scatter, deriv):
        self.region, self.scatter, self.deriv = region, scatter, deriv
        lo, hi = REGIONS[region]
        w, poly, d = DERIVS[deriv]
        self.half = w // 2
        self.i0 = int(np.searchsorted(GRID, lo)) - self.half
        self.i1 = int(np.searchsorted(GRID, hi, side="right")) + self.half
        self.kernel = savgol_coeffs(w, poly, deriv=d, use="conv") if w else None
        self.x = GRID[self.i0 + self.half:self.i1 - self.half]
        self.needs_ref = scatter in ("msc", "emsc")

    @property
    def name(self):
        return f"{self.region}|{self.scatter}|{self.deriv}"

    def crop(self, X):
        return X[..., self.i0:self.i1]

    def apply(self, X, ref=None):
        """X: [n, P] tam ızgara. ref: [F, p] kırpılmış referans (msc/emsc). Dönüş [n, p'] ya da [F, n, p']."""
        Z = self.crop(X)
        if self.scatter == "snv":
            Z = (Z - Z.mean(-1, keepdim=True)) / Z.std(-1, keepdim=True)
        elif self.scatter == "msc":
            Z = msc(Z, ref)
        elif self.scatter == "emsc":
            Z = emsc(Z, ref)
        if self.kernel is not None:
            Z = sg(Z, self.kernel)
        return Z

    def export(self):
        return dict(region=self.region, scatter=self.scatter, deriv=self.deriv,
                    grid_start=float(GRID[0]), grid_step=float(GRID[1] - GRID[0]), grid_n=len(GRID),
                    i0=self.i0, i1=self.i1, half=self.half,
                    kernel=None if self.kernel is None else self.kernel.tolist())


def msc(Z, ref):
    """x = a + b*ref  ->  (x - a) / b"""
    xm = Z.mean(-1)                                   # [n]
    rc = ref - ref.mean(-1, keepdim=True)             # [F, p]
    b = ((Z - xm[:, None]) @ rc.T).T / (rc * rc).sum(-1, keepdim=True)   # [F, n]
    a = xm[None] - b * ref.mean(-1, keepdim=True)
    return (Z[None] - a[..., None]) / b[..., None]


def emsc_design(ref):
    nu = torch.linspace(-1, 1, ref.shape[-1], dtype=ref.dtype, device=ref.device)
    one = torch.ones_like(nu)
    return torch.stack([one.expand_as(ref), ref, nu.expand_as(ref), (nu * nu).expand_as(ref)], -1)  # [F, p, 4]


def emsc(Z, ref):
    """x = c0 + c1*ref + c2*nu + c3*nu^2  ->  (x - c0 - c2*nu - c3*nu^2) / c1"""
    D = emsc_design(ref)
    DtD = D.transpose(1, 2) @ D                                   # [F, 4, 4]
    pinv = torch.cholesky_inverse(torch.linalg.cholesky(DtD)) @ D.transpose(1, 2)   # [F, 4, p]
    C = torch.einsum("fkp,np->fnk", pinv, Z)                      # [F, n, 4]
    base = torch.einsum("fnk,fpk->fnp", C[..., [0, 2, 3]], D[..., [0, 2, 3]])
    return (Z[None] - base) / C[..., 1:2]


def sg(Z, kernel):
    shape = Z.shape
    k = torch.as_tensor(kernel[::-1].copy(), dtype=Z.dtype, device=Z.device).view(1, 1, -1)
    out = torch.nn.functional.conv1d(Z.reshape(-1, 1, shape[-1]), k)
    return out.reshape(*shape[:-1], out.shape[-1])
