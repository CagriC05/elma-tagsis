// Tağşiş tespit motoru: dosya okuma, ön işleme, modeller, karar ağacı.
// train/gpu_prep.py, train/export.py ile birebir aynı hesaplar. Tarayıcıda ve Node'da çalışır.
(function (root) {
  "use strict";

  // ------------------------------------------------------------ dosya okuma
  const UNITS = { 17: "Absorbance", 16: "%Transmittance", 11: "%Reflectance", 12: "Log(1/R)", 3: "Single beam", 15: "Interferogram" };

  function latin1(dv, off, size) {
    let s = "";
    for (let i = 0; i < size; i++) s += String.fromCharCode(dv.getUint8(off + i));
    return s.replace(/[\x00-\x08\x0b-\x1f]+/g, "\n");
  }

  function parseWhen(s) {   // "Monday, September 1, 2025 9:14:31 AM (GMT03:00)"
    const m = /(\w+) (\d{1,2}), (\d{4}) (\d{1,2}):(\d{2}):(\d{2}) ([AP]M)/.exec(s || "");
    if (!m) return null;
    const mon = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].indexOf(m[1]);
    let h = +m[4] % 12 + (m[7] === "PM" ? 12 : 0);
    return new Date(Date.UTC(+m[3], mon, +m[2], h, +m[5], +m[6]));
  }

  // train/spa_meta.py ile aynı alanlar
  function readSPA(buf) {
    const dv = new DataView(buf);
    const nKeys = dv.getUint16(294, true);
    let npts = null, wmax, wmin, off = null, hist = "", acc = "";
    const meta = {};
    for (let i = 0; i < nKeys; i++) {
      const pos = 304 + 16 * i;
      const key = dv.getUint8(pos);
      const o = dv.getUint32(pos + 2, true), size = dv.getUint32(pos + 6, true);
      if (key === 2) {
        npts = dv.getUint32(o + 4, true);
        const u = dv.getUint32(o + 12, true);
        meta.units = UNITS[u] || String(u);
        wmax = dv.getFloat32(o + 16, true);
        wmin = dv.getFloat32(o + 20, true);
        meta.scans = dv.getUint32(o + 36, true);
        meta.bg_scans = dv.getUint32(o + 52, true);
      } else if (key === 3) off = o;
      else if (key === 27) hist = latin1(dv, o, size);
      else if (key === 151) acc = latin1(dv, o, size);
    }
    if (npts === null || off === null) throw new Error("SPA dosyası çözülemedi");
    const x = new Float64Array(npts), y = new Float64Array(npts);
    for (let i = 0; i < npts; i++) {
      x[i] = wmax + (wmin - wmax) * i / (npts - 1);
      y[i] = dv.getFloat32(off + 4 * i, true);
    }
    meta.xmin = Math.min(wmin, wmax);
    meta.xmax = Math.max(wmin, wmax);
    let m = /Resolution:\s*([\d.]+)/.exec(hist);
    meta.resolution = m ? parseFloat(m[1]) : null;
    m = /serial number:\s*(\S+)/.exec(hist);
    meta.serial = m ? m[1] : null;
    m = /Collect Sample on (.+)/.exec(hist);
    const ts = m ? parseWhen(m[1]) : null;
    m = /Background collected on (.+)/.exec(hist);
    const tb = m ? parseWhen(m[1]) : null;
    meta.collected = ts ? ts.toISOString().slice(0, 19).replace("T", " ") : null;
    meta.bg_age_min = ts && tb ? Math.round((ts - tb) / 6000) / 10 : null;
    meta.accessory = acc.split("\n").map(l => l.trim()).find(l => /ATR|Smart/.test(l)) || null;
    meta.title = latin1(dv, 30, 256).split("\n")[0].trim();
    return [{ name: null, x, y, meta }];
  }

  // PerkinElmer .sp ("PEPE"): blok tabanlı; blok kimliği (uint16) + boyut (int32), ardından değişken kimliği.
  // 35698: x başlangıç/bitiş (float64), 35701: nokta sayısı, 35708: veri (float64), 5104: ölçüm bilgileri.
  function readPESP(buf) {
    const dv = new DataView(buf), n = buf.byteLength;
    const find = (block, varId) => {
      for (let i = 44; i + 8 < n; i++)
        if (dv.getUint16(i, true) === block && dv.getUint16(i + 6, true) === varId) return i + 8;
      return -1;
    };
    const px = find(35698, 29981), pn = find(35701, 29995), pd = find(35708, 29974);
    if (px < 0 || pn < 0 || pd < 0) throw new Error("PerkinElmer .sp dosyası çözülemedi");
    const x0 = dv.getFloat64(px, true), x1 = dv.getFloat64(px + 8, true), npts = dv.getUint32(pn, true);
    const x = new Float64Array(npts), y = new Float64Array(npts);
    for (let i = 0; i < npts; i++) {
      x[i] = x0 + (x1 - x0) * i / (npts - 1);
      y[i] = dv.getFloat64(pd + 4 + 8 * i, true);
    }
    // Ölçüm bilgileri bloğu (kimlik 122): iç içe bloklar gezilerek bulunur (specio sp.py ile aynı yol).
    // İçerik: '#u' metin, '$u' / ',u' tamsayı alanları; alan sırası specio'daki gibi.
    const meta = { format: "PerkinElmer .sp", xmin: Math.min(x0, x1), xmax: Math.max(x0, x1), resolution: null, bg_age_min: null, collected: null,
      units: null, scans: null, bg_scans: null, serial: null, accessory: null, title: null };
    try {
      let p = 44, id = dv.getUint16(p, true), size = dv.getInt32(p + 2, true);
      p += 6;
      const nbp = [p + size];
      while (id !== 122 && p < n - 2) {
        if (dv.getUint8(p + 1) === 117) { p = nbp.pop(); }
        else { id = dv.getUint16(p, true); size = dv.getInt32(p + 2, true); p += 6; nbp.push(p + size); }
      }
      const end = Math.min(n, p + size), f = [];
      let j = p;
      while (j + 2 < end) {
        const a = dv.getUint8(j), b = dv.getUint8(j + 1);
        if (b === 117 && a === 35) { const L = dv.getInt16(j + 2, true); f.push(new TextDecoder().decode(new Uint8Array(buf, j + 4, Math.max(0, L)))); j += 4 + L + 6; }
        else if (b === 117 && a === 36) { f.push(dv.getInt16(j + 2, true)); j += 4 + 6; }
        else if (b === 117 && a === 44) { f.push(dv.getInt16(j + 2, true)); j += 4; }
        else j++;
      }
      if (f.length > 32) {
        const type = String(f[16]);
        meta.units = /%T/.test(type) ? "%Transmittance" : /%R/.test(type) ? "%Reflectance" : /\(A\)|Abs/i.test(type) ? "Absorbance" : type;
        meta.scans = f[9]; meta.bg_scans = f[32]; meta.serial = String(f[6]); meta.accessory = String(f[26]);
        meta.title = `${f[5]} · ${f[2]}`;
      }
    } catch (e) { /* başlık okunamazsa yalnızca spektral kontroller uygulanır */ }
    return [{ name: null, x, y, meta }];
  }

  // İçerikten biçim tanıma: uzantıdan bağımsız
  function readAny(buf) {
    const head = String.fromCharCode.apply(null, new Uint8Array(buf, 0, Math.min(64, buf.byteLength)));
    if (head.startsWith("Spectral Data File")) return readSPA(buf);
    if (head.startsWith("PEPE")) return readPESP(buf);
    const bytes = new Uint8Array(buf, 0, Math.min(4096, buf.byteLength));
    if (bytes.some(b => b === 0)) throw new Error("Desteklenmeyen dosya biçimi (desteklenenler: OMNIC .spa, PerkinElmer .sp, metin .tsv/.csv/.txt)");
    return readText(new TextDecoder("utf-8").decode(buf));
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

  // ------------------------------------------------------------ ölçüm uygunluk kontrolü
  // train/export.py spectral_metrics ile aynı tanımlar
  const QC_KERNEL = [-0.08391608391608392, 0.02097902097902098, 0.10256410256410256, 0.16083916083916083, 0.1958041958041958,
    0.2074592074592075, 0.1958041958041958, 0.16083916083916083, 0.10256410256410256, 0.02097902097902098, -0.08391608391608392];

  function spectralMetrics(full, grid) {
    const idx = w => Math.round((w - grid.start) / grid.step);
    const i0 = Math.ceil((900 - grid.start) / grid.step), i1 = Math.floor((1200 - grid.start) / grid.step) + 1;
    const base = full[idx(3900)];
    const res = [];
    let fpMax = -Infinity;
    for (let i = i0; i < i1; i++) {
      let s = 0;
      for (let k = 0; k < 11; k++) s += QC_KERNEL[k] * full[i - 5 + k];
      res.push(full[i] - s);
      fpMax = Math.max(fpMax, full[i]);
    }
    const mr = mean(res);
    let minA = Infinity;
    for (let i = idx(850); i < idx(3700); i++) minA = Math.min(minA, full[i] - base);
    return { noise: Math.sqrt(res.reduce((a, v) => a + (v - mr) ** 2, 0) / res.length) * 1e3,
      base, oh: full[idx(3300)] - base, fp: fpMax - base, minA };
  }

  // Dönüş: [{ level: "red" | "uyari" | "ok", item, value, rule }]
  function checkQuality(model, full, meta) {
    const q = model.qc, out = [];
    const add = (level, item, value, rule) => out.push({ level, item, value, rule });
    if (!q) return out;
    const a = q.acquisition, s = q.spectral;
    if (meta) {
      add(meta.units == null ? "uyari" : a.units.includes(meta.units) ? "ok" : "red", "Veri birimi", meta.units == null ? "okunamadı" : meta.units, "Absorbans olmalı");
      add(meta.resolution == null ? "uyari" : a.resolution.includes(meta.resolution) ? "ok" : "red", "Çözünürlük",
        meta.resolution == null ? "okunamadı" : meta.resolution + " cm⁻¹", a.resolution.join(", ") + " cm⁻¹");
      add(meta.scans == null ? "uyari" : meta.scans < a.scans_min ? "red" : meta.scans < a.scans_typ ? "uyari" : "ok", "Örnek tarama sayısı", meta.scans == null ? "okunamadı" : meta.scans,
        `≥ ${a.scans_typ} (en az ${a.scans_min})`);
      add(meta.bg_scans == null ? "uyari" : meta.bg_scans < a.bg_scans_min ? "red" : "ok", "Arka plan tarama sayısı", meta.bg_scans == null ? "okunamadı" : meta.bg_scans, `≥ ${a.bg_scans_min}`);
      add(meta.accessory == null ? "uyari" : a.accessory.includes(meta.accessory) ? "ok" : "red", "Örnekleme aksesuarı",
        meta.accessory || "okunamadı", a.accessory.join(", "));
      add(meta.serial == null || a.serial.includes(meta.serial) ? "ok" : "uyari", "Cihaz seri no", meta.serial || "okunamadı",
        a.serial.join(", ") + " (farklı cihazda model doğrulanmamıştır)");
      add(meta.bg_age_min == null ? "uyari" : meta.bg_age_min > a.bg_age_max || meta.bg_age_min < 0 ? "red" : "ok", "Arka plan yaşı",
        meta.bg_age_min == null ? "okunamadı" : meta.bg_age_min + " dk", `0–${a.bg_age_max} dk`);
      add(meta.xmin <= a.xmin_max && meta.xmax >= a.xmax_min ? "ok" : "red", "Spektral aralık",
        `${Math.round(meta.xmin)}–${Math.round(meta.xmax)} cm⁻¹`, `${a.xmin_max}–${a.xmax_min} cm⁻¹ kapsanmalı`);
    } else {
      add("uyari", "Ölçüm parametreleri", "doğrulanamadı", "Yalnızca .spa dosyalarında okunabilir");
    }
    if (full.slice(Math.round((850 - model.grid.start) / model.grid.step), Math.round((3700 - model.grid.start) / model.grid.step)).some(v => isNaN(v))) {
      add("red", "Spektral aralık", "eksik", "850–3700 cm⁻¹ kapsanmalı");
      return out;
    }
    const m = spectralMetrics(full, model.grid), f = (v, d = 3) => v.toFixed(d);
    add(m.noise > s.noise.reject ? "red" : m.noise > s.noise.warn ? "uyari" : "ok", "Gürültü", f(m.noise, 2),
      `uyarı > ${f(s.noise.warn, 1)}, red > ${f(s.noise.reject, 1)}`);
    add(m.fp < s.fp.lo ? "red" : m.fp > s.fp.warn_hi ? "uyari" : "ok", "Parmak izi tepe yüksekliği", f(m.fp),
      `≥ ${f(s.fp.lo)} (kristal teması); uyarı > ${f(s.fp.warn_hi)}`);
    add(m.oh < s.oh.lo ? "red" : m.oh < s.oh.warn_lo || m.oh > s.oh.warn_hi ? "uyari" : "ok", "O–H bandı yüksekliği", f(m.oh),
      `${f(s.oh.warn_lo)}–${f(s.oh.warn_hi)} (Brix); red < ${f(s.oh.lo)}`);
    add(Math.abs(m.base) > s.base.abs_max ? "red" : "ok", "Baz çizgisi (3900 cm⁻¹)", f(m.base), `|A| ≤ ${f(s.base.abs_max)}`);
    add(m.minA < s.minA.lo ? "red" : "ok", "En düşük absorbans", f(m.minA), `≥ ${f(s.minA.lo)} (arka plan hatası)`);
    return out;
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

  // meta: .spa başlığından ölçüm parametreleri (yoksa null). Herhangi bir "red" maddesi varsa sonuç verilmez.
  function analyze(model, full, meta) {
    const r = { warnings: [], steps: [] };
    r.qc = checkQuality(model, full, meta || null);
    if (!r.qc.some(c => c.level === "red" && c.item === "Spektral aralık")) {
      const dom = predict(model.domain.model, prep(model.domain.pipeline, full));
      r.domain = dom;
      r.qc.push({ level: dom > 1 ? "red" : "ok", item: "Eğitim verisine benzerlik", value: "f/f_kritik = " + dom.toFixed(2),
        rule: "≤ 1 (aksi halde ölçüm hatası, farklı ürün ya da tanımsız adulterant)" });
    }
    r.qc.filter(c => c.level === "uyari").forEach(c => r.warnings.push(`${c.item}: ${c.value} (${c.rule})`));
    if (r.qc.some(c => c.level === "red")) {
      r.verdict = "reddedildi";
      r.label = "Ölçüm reddedildi";
      r.rejectReasons = r.qc.filter(c => c.level === "red").map(c => `${c.item}: ${c.value} — beklenen ${c.rule}`);
      return r;
    }
    // f ≤ şüpheli eşiği: otantik · şüpheli eşiği < f ≤ 1: şüpheli (zincirden geçer) · f > 1: tağşişli
    const f = runTask(model, "T1_otantik", full);
    const sus = model.tasks.T1_otantik.model.suspect;
    r.steps.push({ node: "T1_otantik", value: f, suspect: sus });
    if (f <= 1 && !(sus && f > sus)) {
      r.verdict = "otantik";
      r.label = "Otantik";
      return r;
    }
    r.verdict = f > 1 ? "tagsisli" : "supheli";
    const t1m = model.tasks.T1_otantik.model, pc = v => "%" + Math.round(v * 100);
    if (r.verdict === "supheli") r.warnings.push(`OC-SIMCA ${pc(t1m.spec || 0.95)} eşiğinin altında ama ${pc(t1m.suspect_spec || 0.8)} şüphe eşiğinin üstünde (f/f_kritik = ${f.toFixed(2)}); tağşiş olasılığı nedeniyle tür ve oran tahmini yapıldı.`);
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
    r.label = (r.verdict === "supheli" ? "Şüpheli: olası " : "") + LABEL[cls];
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

  const api = { readSPA, readPESP, readAny, readText, toGrid, prep, predict, runTask, analyze, spectralMetrics, checkQuality, LABEL };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(this);
