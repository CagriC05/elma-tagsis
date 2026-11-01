// Tağşiş tespit motoru: dosya okuma, ön işleme, modeller, karar ağacı.
// train/gpu_prep.py, train/export.py ile birebir aynı hesaplar. Tarayıcıda ve Node'da çalışır.
(function (root) {
  "use strict";

  // ------------------------------------------------------------ dosya okuma
  function readSPA(buf) {
    const dv = new DataView(buf);
    const nKeys = dv.getUint16(294, true);
    let npts = null, wmax, wmin, off = null;
    for (let i = 0; i < nKeys; i++) {
      const pos = 304 + 16 * i;
      const key = dv.getUint8(pos);
      const o = dv.getUint32(pos + 2, true);
      if (key === 2) {
        npts = dv.getUint32(o + 4, true);
        wmax = dv.getFloat32(o + 16, true);
        wmin = dv.getFloat32(o + 20, true);
      } else if (key === 3) off = o;
    }
    if (npts === null || off === null) throw new Error("SPA dosyası çözülemedi");
    const x = new Float64Array(npts), y = new Float64Array(npts);
    for (let i = 0; i < npts; i++) {
      x[i] = wmax + (wmin - wmax) * i / (npts - 1);
      y[i] = dv.getFloat32(off + 4 * i, true);
    }
    return [{ name: null, x, y }];
  }

  function parseNum(s) {
    s = s.trim();
    if (s.indexOf(",") >= 0 && s.indexOf(".") < 0) s = s.replace(",", ".");
    return s === "" ? NaN : Number(s);
  }

  // İki biçim: (a) iki sütun x,y  (b) matris: ilk satır dalga sayıları, sonraki satırlar [ad,] absorbanslar
  function readText(text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim() !== "");
    const delim = ["\t", ";", ","].find(d => lines.slice(0, 20).every(l => l.split(d).length > 1 || !/\d/.test(l))) || /\s+/;
    const rows = lines.map(l => l.split(delim));
    const numeric = r => r.filter(v => !isNaN(parseNum(v))).length;
    const wide = rows.find(r => numeric(r) > 10);
    if (wide) {
      const hi = rows.indexOf(wide);
      const head = wide.map(parseNum);
      const first = isNaN(head[0]) ? 1 : 0;
      const xs = head.slice(first);
      const out = [];
      for (const r of rows.slice(hi + 1)) {
        const vals = r.slice(first).map(parseNum);
        if (vals.filter(v => !isNaN(v)).length < 10) continue;
        out.push({ name: first ? r[0].trim() : null, x: Float64Array.from(xs), y: Float64Array.from(vals) });
      }
      return out;
    }
    const x = [], y = [];
    for (const r of rows) {
      if (r.length < 2) continue;
      const a = parseNum(r[0]), b = parseNum(r[1]);
      if (!isNaN(a) && !isNaN(b)) { x.push(a); y.push(b); }
    }
    if (x.length < 100) throw new Error("Spektrum verisi bulunamadı (iki sütun: dalga sayısı, absorbans)");
    return [{ name: null, x: Float64Array.from(x), y: Float64Array.from(y) }];
  }

  function toGrid(x, y, grid) {
    const idx = Array.from(x.keys()).sort((a, b) => x[a] - x[b]);
    const xs = idx.map(i => x[i]), ys = idx.map(i => y[i]);
    const g = new Float64Array(grid.n);
    let j = 0;
    for (let i = 0; i < grid.n; i++) {
      const w = grid.start + grid.step * i;
      if (w < xs[0] || w > xs[xs.length - 1]) { g[i] = NaN; continue; }
      while (j < xs.length - 2 && xs[j + 1] < w) j++;
      const t = (w - xs[j]) / (xs[j + 1] - xs[j]);
      g[i] = ys[j] + t * (ys[j + 1] - ys[j]);
    }
    return g;
  }

  // ------------------------------------------------------------ ön işleme
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  function prep(p, full) {
    let z = Array.from(full.slice(p.i0, p.i1));
    if (z.some(v => isNaN(v))) throw new Error(`Spektrum ${p.region} cm⁻¹ aralığını kapsamıyor`);
    if (p.scatter === "snv") {
      const m = mean(z);
      const sd = Math.sqrt(z.reduce((s, v) => s + (v - m) ** 2, 0) / (z.length - 1));
      z = z.map(v => (v - m) / sd);
    } else if (p.scatter === "msc") {
      const r = p.ref, xm = mean(z), rm = mean(r);
      let num = 0, den = 0;
      for (let i = 0; i < z.length; i++) { num += (z[i] - xm) * (r[i] - rm); den += (r[i] - rm) ** 2; }
      const b = num / den, a = xm - b * rm;
      z = z.map(v => (v - a) / b);
    } else if (p.scatter === "emsc") {
      const r = p.ref, n = z.length;
      const D = r.map((rv, i) => { const nu = -1 + 2 * i / (n - 1); return [1, rv, nu, nu * nu]; });
      const A = [0, 1, 2, 3].map(a => [0, 1, 2, 3].map(b => D.reduce((s, d) => s + d[a] * d[b], 0)));
      const v = [0, 1, 2, 3].map(a => D.reduce((s, d, i) => s + d[a] * z[i], 0));
      const c = solve(A, v);
      z = z.map((zv, i) => (zv - c[0] - c[2] * D[i][2] - c[3] * D[i][3]) / c[1]);
    }
    if (p.kernel) {
      const k = p.kernel, L = k.length, out = new Array(z.length - L + 1);
      for (let j = 0; j < out.length; j++) {
        let s = 0;
        for (let m = 0; m < L; m++) s += k[L - 1 - m] * z[j + m];
        out[j] = s;
      }
      z = out;
    }
    return z;
  }

  function solve(A, b) {   // küçük sistem: Gauss eleme (kısmi pivot)
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  // ------------------------------------------------------------ modeller
  function predict(m, z) {
    switch (m.type) {
      case "ddsimca": {
        const zc = z.map((v, i) => v - m.mean[i]);
        const ss = dot(zc, zc);
        let h = 0, ex = 0;
        for (let k = 0; k < m.P.length; k++) {
          const t = dot(zc, m.P[k]);
          h += t * t / m.lam[k];
          ex += t * t;
        }
        const q = Math.max(ss - ex, 1e-12);
        return (m.Nh * h / m.h0 + m.Nq * q / m.q0) / m.fcrit;
      }
      case "linear":
        return dot(z, m.coef) + m.intercept;
      case "krr_rbf": {
        let s = 0;
        for (let i = 0; i < m.support.length; i++) {
          const sv = m.support[i];
          let d = 0;
          for (let j = 0; j < z.length; j++) d += (z[j] - sv[j]) ** 2;
          s += m.alpha[i] * Math.exp(-m.gamma * d);
        }
        return s + m.ybar;
      }
      case "knn": {
        const d = m.support.map((sv, i) => {
          let s = 0;
          for (let j = 0; j < z.length; j++) s += (z[j] - sv[j]) ** 2;
          return [s, m.y[i]];
        }).sort((a, b) => a[0] - b[0]);
        return mean(d.slice(0, m.k).map(v => v[1]));
      }
      case "mlp": {
        const zc = z.map((v, i) => v - m.mean[i]);
        const t = m.P.map((pk, k) => (dot(zc, pk) - m.tm[k]) / m.tsd[k]);
        const H = m.b1.length;
        let o = m.b2;
        for (let j = 0; j < H; j++) {
          let a = m.b1[j];
          for (let k = 0; k < t.length; k++) a += t[k] * m.W1[k][j];
          o += Math.max(a, 0) * m.W2[j];
        }
        return m.ym === null ? o : o * m.ysd + m.ym;
      }
      case "cnn": {
        const x = z.map((v, i) => (v - m.mu[i]) / m.sd);
        const conv = (inp, W, B) => {   // inp [cin][L], W [cout][cin][ks]
          const L = inp[0].length, ks = W[0][0].length, Lo = Math.floor((L - ks) / m.stride) + 1;
          return W.map((wc, c) => {
            const out = new Array(Lo);
            for (let t = 0; t < Lo; t++) {
              let s = B[c];
              for (let ci = 0; ci < wc.length; ci++) {
                const w = wc[ci], row = inp[ci], o = t * m.stride;
                for (let k = 0; k < ks; k++) s += w[k] * row[o + k];
              }
              out[t] = Math.max(s, 0);
            }
            return out;
          });
        };
        const a2 = conv(conv([x], m.conv1_w, m.conv1_b), m.conv2_w, m.conv2_b);
        let o = m.b, j = 0;
        for (const ch of a2) {
          const L = ch.length;
          for (let i = 0; i < m.pool; i++) {
            const s = Math.floor(i * L / m.pool), e = Math.ceil((i + 1) * L / m.pool);
            o += mean(ch.slice(s, e)) * m.W[j++];
          }
        }
        return m.ym === null ? o : o * m.ysd + m.ym;
      }
    }
    throw new Error("Bilinmeyen model: " + m.type);
  }

  function runTask(model, name, full) {
    const t = model.tasks[name];
    return predict(t.model, prep(t.pipeline, full));
  }

  // ------------------------------------------------------------ karar ağacı
  const LABEL = { glikoz: "Glikoz şurubu", fruktoz: "Fruktoz şurubu", hurma: "Hurma konsantresi", uzum: "Üzüm konsantresi" };

  function analyze(model, full) {
    const r = { warnings: [], steps: [] };
    const dom = predict(model.domain.model, prep(model.domain.pipeline, full));
    r.domain = dom;
    if (dom > 1) r.warnings.push("Spektrum eğitim verisindeki hiçbir örneğe benzemiyor (ölçüm hatası, farklı ürün ya da tanımsız adulterant olabilir). Sonuç güvenilir değil.");
    const f = runTask(model, "T1_otantik", full);
    r.steps.push({ node: "T1_otantik", value: f });
    if (f <= 1) {
      r.verdict = "otantik";
      r.label = "Otantik";
      if (f > 0.8) r.warnings.push("OC-SIMCA mesafesi sınıra yakın (f/f_kritik = " + f.toFixed(2) + ").");
      return r;
    }
    r.verdict = "tagsisli";
    if (f < 1.25) r.warnings.push("OC-SIMCA mesafesi sınıra yakın (f/f_kritik = " + f.toFixed(2) + ").");
    const node = (name, pos, neg) => {
      const s = runTask(model, name, full), thr = model.tasks[name].conf_threshold || 0;
      const low = Math.abs(s) < thr;
      r.steps.push({ node: name, value: s, low });
      return [s > 0 ? pos : neg, low];
    };
    const [grp, low2] = node("T2_surup_meyve", "surup", "meyve");
    const [cls, low3] = grp === "surup" ? node("T3_glikoz_fruktoz", "glikoz", "fruktoz") : node("T4_hurma_uzum", "hurma", "uzum");
    r.group = grp;
    r.cls = cls;
    r.label = LABEL[cls];
    r.lowConfidence = low2 || low3;
    if (low2) r.warnings.push("Şurup / meyve konsantresi ayrımı düşük güvenli.");
    if (low3) r.warnings.push((grp === "surup" ? "Glikoz / fruktoz" : "Hurma / üzüm") + " ayrımı düşük güvenli.");
    const rt = "R_" + cls, lvl = runTask(model, rt, full);
    r.steps.push({ node: rt, value: lvl });
    r.level = Math.max(0, lvl);
    r.interval = model.tasks[rt].interval;
    if (lvl > 35) r.warnings.push("Tahmini oran eğitim aralığının (%2–30) dışında; değer ekstrapolasyondur.");
    return r;
  }

  const api = { readSPA, readText, toGrid, prep, predict, runTask, analyze, LABEL };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(this);
