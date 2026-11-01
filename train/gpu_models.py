"""Kat (fold) boyutunda batched modeller.

Ortak girdi: Z [n, p] ya da [F, n, p] (ön işlenmiş), M [F, n] eğitim maskesi (0/1), y [n].
Her fonksiyon tüm n örnek için tahmin döndürür: [F, H, n]  (H = hiperparametre sayısı).
Sınıflandırmada y = +1 / -1 ve skor > 0 -> +1.
"""
import numpy as np
import torch
from scipy.stats import chi2

PLS_K = 20
PCA_K = 25
LDA_K = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
SIMCA_ALPHA = [0.01, 0.05]
RIDGE_R = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RBF_G = [0.03, 0.1, 0.3, 1.0, 3.0]
KNN_K = [1, 3, 5, 9]
MLP_H = [16, 64]
MLP_WD = [1e-3, 1e-1]
MLP_PCS = 15
CNN_WD = [1e-4, 1e-2]


def center(Z, M):
    cnt = M.sum(1)
    if Z.dim() == 2:
        mu = (M @ Z) / cnt[:, None]
        return Z[None] - mu[:, None], mu
    mu = (M[..., None] * Z).sum(1) / cnt[:, None]
    return Z - mu[:, None], mu


def wmean(v, M):
    """v [F, ..., n] içinde maskeli ortalama ve varyans (son eksen)."""
    cnt = M.sum(-1)
    while M.dim() < v.dim():
        M, cnt = M.unsqueeze(1), cnt.unsqueeze(1)
    m = (v * M).sum(-1) / cnt
    var = (((v - m[..., None]) ** 2) * M).sum(-1) / (cnt - 1)
    return m, var


# ---------------------------------------------------------------- PCA / DD-SIMCA
def pca(Zc, M, K=PCA_K):
    Zm = Zc * M[..., None]
    L, U = torch.linalg.eigh(Zm @ Zm.transpose(1, 2))
    L, U = L.flip(-1)[:, :K].clamp_min(1e-12), U.flip(-1)[:, :, :K]
    P = (Zm.transpose(1, 2) @ U) / L.sqrt()[:, None]          # [F, p, K] birim yükler
    T = Zc @ P                                                 # [F, n, K]
    lam = L / (M.sum(1, keepdim=True) - 1)
    return T, lam, P


def dd_simca(T, lam, Zc, M, alphas=SIMCA_ALPHA):
    """f / f_kritik döndürür (<= 1 -> otantik). Hiper: (k, alpha) k = 1..K."""
    h = torch.cumsum(T * T / lam[:, None], -1)                 # [F, n, K]
    q = ((Zc * Zc).sum(-1, keepdim=True) - torch.cumsum(T * T, -1)).clamp_min(1e-12)
    h, q = h.transpose(1, 2), q.transpose(1, 2)                # [F, K, n]
    h0, hv = wmean(h, M)
    q0, qv = wmean(q, M)
    Nh = torch.round(2 * h0 * h0 / hv).clamp(1, 500)
    Nq = torch.round(2 * q0 * q0 / qv).clamp(1, 500)
    f = Nh[..., None] * h / h0[..., None] + Nq[..., None] * q / q0[..., None]
    dof = (Nh + Nq).cpu().numpy()
    out, hyp = [], []
    for a in alphas:
        fc = torch.as_tensor(chi2.ppf(1 - a, dof), dtype=f.dtype, device=f.device)
        out.append(f / fc[..., None])
        hyp += [dict(k=k + 1, alpha=a) for k in range(T.shape[-1])]
    return torch.cat(out, 1), hyp


# ---------------------------------------------------------------- PLS (PLS-R / PLS-DA)
def pls(Zc, M, y, K=PLS_K):
    ybar = (M @ y) / M.sum(1)
    yc = (y[None] - ybar[:, None]) * M                         # [F, n]
    Xk = Zc * M[..., None]
    R, Q, P = [], [], []
    for _ in range(K):
        w = torch.einsum("fnp,fn->fp", Xk, yc)
        w = w / w.norm(dim=-1, keepdim=True)
        t = torch.einsum("fnp,fp->fn", Xk, w)
        tt = (t * t).sum(-1, keepdim=True)
        p = torch.einsum("fnp,fn->fp", Xk, t) / tt
        q = (yc * t).sum(-1, keepdim=True) / tt
        Xk = Xk - t[..., None] * p[:, None]
        r = w.clone()
        for rj, pj in zip(R, P):                               # r_k = w_k - sum_j (p_j' w_k) r_j
            r = r - (pj * w).sum(-1, keepdim=True) * rj
        R.append(r), P.append(p), Q.append(q)
    B = torch.cumsum(torch.stack([r * q for r, q in zip(R, Q)], 1), 1)   # [F, K, p]
    pred = torch.einsum("fnp,fkp->fkn", Zc, B) + ybar[:, None, None]
    return pred, [dict(k=k + 1) for k in range(K)]


# ---------------------------------------------------------------- PCA-LDA
def pca_lda(T, M, y, ks=LDA_K):
    out, hyp = [], []
    pos, neg = M * (y > 0), M * (y < 0)
    for k in ks:
        if k > T.shape[-1]:
            break
        Tk = T[..., :k]
        mp = (pos[..., None] * Tk).sum(1) / pos.sum(1, keepdim=True)
        mn = (neg[..., None] * Tk).sum(1) / neg.sum(1, keepdim=True)
        D = Tk - torch.where((y > 0)[None, :, None], mp[:, None], mn[:, None])
        D = D * M[..., None]
        Sw = D.transpose(1, 2) @ D / (M.sum(1)[:, None, None] - 2)
        Sw = Sw + 1e-8 * torch.eye(k, dtype=T.dtype, device=T.device) * Sw.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        w = torch.cholesky_solve((mp - mn)[..., None], torch.linalg.cholesky(Sw))[..., 0]
        out.append(torch.einsum("fnk,fk->fn", Tk - 0.5 * (mp + mn)[:, None], w))
        hyp.append(dict(k=k))
    return torch.stack(out, 1), hyp


# ---------------------------------------------------------------- kernel ridge (doğrusal / RBF = LS-SVM, KRR biçimi)
def _kridge(Kmat, M, y, ratios):
    """Kmat [F, n, n] (tüm örnekler); eğitim dışı satır/sütunlar maskeyle ayrılır."""
    MM = M[:, :, None] * M[:, None, :]
    Km = Kmat * MM + torch.diag_embed(1 - M)
    L, U = torch.linalg.eigh(Km)
    ybar = (M @ y) / M.sum(1)
    Uy = torch.einsum("fnm,fn->fm", U, (y[None] - ybar[:, None]) * M)
    scale = (Kmat.diagonal(dim1=1, dim2=2) * M).sum(1) / M.sum(1)
    out = []
    for r in ratios:
        alpha = torch.einsum("fnm,fm->fn", U, Uy / (L + r * scale[:, None]))
        alpha = alpha * M
        out.append(torch.einsum("fnm,fm->fn", Kmat, alpha) + ybar[:, None])
    return torch.stack(out, 1)


def ridge_linear(Zc, M, y, ratios=RIDGE_R):
    return _kridge(Zc @ Zc.transpose(1, 2), M, y, ratios), [dict(r=r) for r in ratios]


def sqdist(Z):
    sq = (Z * Z).sum(-1)
    return (sq[..., :, None] + sq[..., None, :] - 2 * Z @ Z.transpose(-1, -2)).clamp_min(0)


def krr_rbf(D2, M, y, gs=RBF_G, ratios=RIDGE_R):
    """D2 [n, n] ya da [F, n, n]; gamma = g / ortalama(eğitim çiftleri D2)."""
    if D2.dim() == 2:
        D2 = D2.expand(M.shape[0], -1, -1)
    MM = M[:, :, None] * M[:, None, :]
    dm = (D2 * MM).sum((1, 2)) / MM.sum((1, 2))
    out, hyp = [], []
    for g in gs:
        out.append(_kridge(torch.exp(-g * D2 / dm[:, None, None]), M, y, ratios))
        hyp += [dict(g=g, r=r) for r in ratios]
    return torch.cat(out, 1), hyp


# ---------------------------------------------------------------- kNN
def knn(D2, M, y, ks=KNN_K):
    if D2.dim() == 2:
        D2 = D2.expand(M.shape[0], -1, -1)
    D = D2 + (1 - M)[:, None, :] * 1e30
    idx = D.topk(max(ks), dim=-1, largest=False).indices      # [F, n, kmax]
    yy = torch.cumsum(y[idx], -1)
    out = torch.stack([yy[..., k - 1] / k for k in ks], 1)
    return out, [dict(k=k) for k in ks]


# ---------------------------------------------------------------- MLP (PCA skorları üzerinde, batched)
def mlp(T, M, y, kind, hidden=MLP_H, wds=MLP_WD, epochs=400, lr=1e-2, seed=0, params=None):
    T = T[..., :MLP_PCS].float()
    M = M.float()
    tm, tv = wmean(T.transpose(1, 2), M)
    X = (T - tm[:, None]) / tv.sqrt()[:, None]                 # [F, n, k]
    y = y.float()
    if kind == "reg":
        ym, yv = wmean(y[None].expand_as(M), M)
        target = (y[None] - ym[:, None]) / yv.sqrt()[:, None]
    else:
        target = (y[None] > 0).float().expand_as(M)
    F, n, k = X.shape
    combos = [(h, w) for h in hidden for w in wds]             # tüm gizli boyutlar tek döngüde: birim maskesi
    Hn, hm = len(combos), max(hidden)
    Xb = X.repeat_interleave(Hn, 0)                            # [F*Hn, n, k]
    tb, Mb = target.repeat_interleave(Hn, 0), M.repeat_interleave(Hn, 0)
    wd = torch.tensor([w for _, w in combos], device=X.device).repeat(F)
    um = torch.stack([(torch.arange(hm, device=X.device) < h).float() for h, _ in combos]).repeat(F, 1)[:, None]
    g = torch.Generator(device=X.device).manual_seed(seed)
    hs = torch.tensor([h for h, _ in combos], device=X.device, dtype=torch.float).repeat(F)
    W1 = (torch.randn(F * Hn, k, hm, device=X.device, generator=g) * (2 / k) ** 0.5).requires_grad_()
    b1 = torch.zeros(F * Hn, 1, hm, device=X.device, requires_grad=True)
    W2 = (torch.randn(F * Hn, hm, 1, device=X.device, generator=g) / hs.sqrt()[:, None, None]).requires_grad_()
    b2 = torch.zeros(F * Hn, 1, 1, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    cnt = Mb.sum(1)

    def forward():
        return ((torch.relu(Xb @ W1 + b1) * um) @ W2 + b2)[..., 0]

    for _ in range(epochs):
        o = forward()
        if kind == "reg":
            l = ((o - tb) ** 2 * Mb).sum(1) / cnt
        else:
            l = (torch.nn.functional.binary_cross_entropy_with_logits(o, tb, reduction="none") * Mb).sum(1) / cnt
        l = l + wd * (((W1 * um) ** 2).sum((1, 2)) + ((W2 * um.transpose(1, 2)) ** 2).sum((1, 2)))
        opt.zero_grad()
        l.sum().backward()
        opt.step()
    with torch.no_grad():
        o = forward().view(F, Hn, n)
        if kind == "reg":
            o = o * yv.sqrt()[:, None, None] + ym[:, None, None]
    if params is not None:   # dışa aktarma için ağırlıklar (F*Hn sırası: kat-major)
        params.update(tm=tm, tsd=tv.sqrt(), W1=W1.detach(), b1=b1.detach(), W2=W2.detach(), b2=b2.detach(),
                      um=um, combos=combos, ym=ym if kind == "reg" else None, ysd=yv.sqrt() if kind == "reg" else None)
    return o.double(), [dict(h=h, wd=w) for h, w in combos]


# ---------------------------------------------------------------- 1D-CNN (unfold + einsum ile batched; ROCm'da grouped conv'dan 2-3x hızlı)
def cnn(Z, M, y, kind, wds=CNN_WD, epochs=300, lr=3e-3, seed=0, c1=8, c2=16, ks=7, pool=8, params=None):
    Z, M, y = Z.float(), M.float(), y.float()
    F = M.shape[0]
    if Z.dim() == 2:
        Z = Z.expand(F, -1, -1)
    mu = (M[..., None] * Z).sum(1) / M.sum(1)[:, None]
    sd = ((((Z - mu[:, None]) ** 2) * M[..., None]).sum(1) / M.sum(1)[:, None]).sqrt().mean(-1)
    Z = (Z - mu[:, None]) / sd[:, None, None]
    if kind == "reg":
        ym, yv = wmean(y[None].expand_as(M), M)
        target = (y[None] - ym[:, None]) / yv.sqrt()[:, None]
    else:
        target = (y[None] > 0).float().expand_as(M)
    Hn = len(wds)
    B = F * Hn
    n = Z.shape[1]
    X = Z.repeat_interleave(Hn, 0).transpose(0, 1).contiguous()   # [n, B, p]
    tb, Mb = target.repeat_interleave(Hn, 0), M.repeat_interleave(Hn, 0)
    wd = torch.tensor(wds, device=Z.device).repeat(F)
    g = torch.Generator(device=Z.device).manual_seed(seed)
    uni = lambda shape, fan: ((torch.rand(*shape, device=Z.device, generator=g) * 2 - 1) / fan ** 0.5).requires_grad_()
    W1, b1 = uni((B, c1, ks), ks), uni((B, c1), ks)              # nn.Conv1d varsayılan başlatmasıyla aynı dağılım
    W2, b2 = uni((B, c2, c1, ks), c1 * ks), uni((B, c2), c1 * ks)
    W = (torch.randn(B, c2 * pool, device=Z.device, generator=g) / (c2 * pool) ** 0.5).requires_grad_()
    b = torch.zeros(B, device=Z.device, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2, W, b], lr=lr)
    cnt = Mb.sum(1)

    def forward():
        a = torch.relu(torch.einsum("nblk,bck->nbcl", X.unfold(-1, ks, 2), W1) + b1[None, :, :, None])
        a = torch.relu(torch.einsum("nbclk,bdck->nbdl", a.unfold(-1, ks, 2), W2) + b2[None, :, :, None])
        a = torch.nn.functional.adaptive_avg_pool1d(a.reshape(n, B * c2, -1), pool).view(n, B, c2 * pool)
        return torch.einsum("nbj,bj->bn", a, W) + b[:, None]       # [B, n]

    for _ in range(epochs):
        o = forward()
        if kind == "reg":
            l = ((o - tb) ** 2 * Mb).sum(1) / cnt
        else:
            l = (torch.nn.functional.binary_cross_entropy_with_logits(o, tb, reduction="none") * Mb).sum(1) / cnt
        l = l + wd * ((W1 ** 2).sum((1, 2)) + (W2 ** 2).sum((1, 2, 3)) + (W ** 2).sum(1))
        opt.zero_grad()
        l.sum().backward()
        opt.step()
    with torch.no_grad():
        o = forward().view(F, Hn, n)
        if kind == "reg":
            o = o * yv.sqrt()[:, None, None] + ym[:, None, None]
    if params is not None:
        params.update(mu=mu, sd=sd, conv1_w=W1.detach()[:, :, None], conv1_b=b1.detach(), conv2_w=W2.detach(),
                      conv2_b=b2.detach(), W=W.detach(), b=b.detach(), pool=pool, ym=ym if kind == "reg" else None,
                      ysd=yv.sqrt() if kind == "reg" else None)
    return o.double(), [dict(wd=w) for w in wds]
