// Arayüz: dosya yükleme, sonuç tablosu, detay paneli, doğrulama özeti.
(function () {
  "use strict";
  const E = window.Engine;
  const $ = s => document.querySelector(s);
  let MODEL = null, rows = [];

  const MODEL_NAME = { ddsimca: "DD-SIMCA", pls: "PLS", pca_lda: "PCA-LDA", ridge: "Ridge", krr_rbf: "RBF çekirdek (LS-SVM)",
    knn: "kNN", mlp: "MLP (ANN)", cnn: "1D-CNN" };
  const SCATTER = { none: "—", snv: "SNV", msc: "MSC", emsc: "EMSC" };
  const DERIV = { none: "—", sg11: "SG düzleştirme (11)", d1w11: "SG 1. türev (11)", d1w21: "SG 1. türev (21)",
    d2w15: "SG 2. türev (15)", d2w25: "SG 2. türev (25)", d3w21: "SG 3. türev (21)", d3w31: "SG 3. türev (31)" };
  const NODE = { T1_otantik: "Otantiklik (OC-SIMCA)", T2_surup_meyve: "Şurup / meyve konsantresi", T3_glikoz_fruktoz: "Glikoz / fruktoz",
    T4_hurma_uzum: "Hurma / üzüm", R_glikoz: "Oran: glikoz", R_fruktoz: "Oran: fruktoz", R_hurma: "Oran: hurma", R_uzum: "Oran: üzüm" };
  const CLS = { otantik: "Otantik", glikoz: "Glikoz", fruktoz: "Fruktoz", hurma: "Hurma", uzum: "Üzüm" };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v, d = 1) => v === null || v === undefined || isNaN(v) ? "—" : Number(v).toFixed(d);

  fetch("model.json").then(r => r.json()).then(m => { MODEL = m; showModels(); }).catch(e => {
    $("#status").textContent = "Model dosyası yüklenemedi: " + e.message;
  });

  // ---------------------------------------------------------------- yükleme
  const drop = $("#drop"), input = $("#file");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handle(e.dataTransfer.files));
  input.addEventListener("change", () => { handle(input.files); input.value = ""; });

  async function handle(files) {
    if (!MODEL) return;
    for (const f of files) {
      $("#status").textContent = f.name + " işleniyor…";
      await new Promise(r => setTimeout(r, 0));
      try {
        const specs = f.name.toLowerCase().endsWith(".spa") ? E.readSPA(await f.arrayBuffer()) : E.readText(await f.text());
        specs.forEach((s, i) => {
          const name = s.name || (specs.length > 1 ? `${f.name} #${i + 1}` : f.name);
          const full = E.toGrid(s.x, s.y, MODEL.grid);
          try { rows.push({ name, full, res: E.analyze(MODEL, full) }); }
          catch (err) { rows.push({ name, full, error: err.message }); }
        });
      } catch (err) {
        rows.push({ name: f.name, error: err.message });
      }
    }
    $("#status").textContent = rows.length + " spektrum analiz edildi.";
    render();
  }

  $("#clear").onclick = () => { rows = []; render(); $("#detail").style.display = "none"; $("#status").textContent = ""; };
  $("#csv").onclick = () => {
    const head = ["dosya", "karar", "tur", "tahmini_oran", "aralik_pm", "dusuk_guven", "OC_SIMCA_f_fkrit", "uyarilar"];
    const lines = [head.join(";")].concat(rows.map(r => {
      const x = r.res || {};
      const f = x.steps ? x.steps[0].value : "";
      return [r.name, r.error ? "HATA" : x.label === "Otantik" ? "otantik" : "tagsisli", x.cls || "",
        x.level !== undefined ? x.level.toFixed(1) : "", x.interval ? x.interval.toFixed(1) : "",
        x.lowConfidence ? "evet" : "", f === "" ? "" : f.toFixed(3),
        (r.error ? [r.error] : x.warnings).join(" | ")].map(v => `"${String(v).replace(/"/g, '""')}"`).join(";");
    }));
    const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tagsis_sonuclari.csv";
    a.click();
  };

  function render() {
    const tb = $("#results tbody");
    $("#csv").disabled = $("#clear").disabled = !rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="5" class="muted">Henüz dosya yüklenmedi.</td></tr>'; return; }
    tb.innerHTML = rows.map((r, i) => {
      if (r.error) return `<tr data-i="${i}"><td>${esc(r.name)}</td><td><span class="tag err">Hata</span></td><td></td><td></td><td class="warn">${esc(r.error)}</td></tr>`;
      const x = r.res, adult = x.verdict === "tagsisli";
      const lvl = adult ? `${fmt(x.level)} <span class="muted">± ${fmt(x.interval)}</span>` : "";
      const w = x.warnings.length ? `<span class="warn">${x.warnings.length} uyarı</span>` : "";
      return `<tr data-i="${i}"><td>${esc(r.name)}</td><td><span class="tag ${adult ? "bad" : "ok"}">${adult ? "Tağşişli" : "Otantik"}</span></td>` +
        `<td>${adult ? esc(x.label) + (x.lowConfidence ? ' <span class="warn">(düşük güven)</span>' : "") : ""}</td><td class="num">${lvl}</td><td>${w}</td></tr>`;
    }).join("");
    tb.querySelectorAll("tr").forEach(tr => tr.onclick = () => {
      tb.querySelectorAll("tr").forEach(t => t.classList.remove("sel"));
      tr.classList.add("sel");
      detail(rows[+tr.dataset.i]);
    });
  }

  // ---------------------------------------------------------------- detay
  function detail(r) {
    $("#detail").style.display = "block";
    $("#d-title").textContent = r.name;
    $("#d-plot").innerHTML = r.full ? plot(r.full) : "";
    if (r.error) {
      $("#d-verdict").innerHTML = `<span class="tag err">Hata</span>`;
      $("#d-path").innerHTML = "";
      $("#d-warn").innerHTML = `<p class="warn">${esc(r.error)}</p>`;
      return;
    }
    const x = r.res, adult = x.verdict === "tagsisli";
    $("#d-verdict").innerHTML = adult
      ? `<div class="big" style="color:var(--bad)">${esc(x.label)}</div><div>Tahmini oran: <b>%${fmt(x.level)}</b> <span class="muted">(± ${fmt(x.interval)}, %95 aralık)</span></div>`
      : `<div class="big" style="color:var(--good)">Otantik</div><div class="muted">Tağşiş, yöntemin tespit sınırının altında</div>`;
    $("#d-path").innerHTML = x.steps.map(s => {
      const t = MODEL.tasks[s.node], c = t.config;
      let val;
      if (s.node === "T1_otantik") val = `f/f<sub>kritik</sub> = ${s.value.toFixed(3)} → ${s.value <= 1 ? "kabul (otantik)" : "red (tağşişli)"}`;
      else if (t.kind === "clf") val = `skor = ${s.value.toFixed(3)}${s.low ? ' <span class="warn">düşük güven</span>' : ""}`;
      else val = `tahmin = %${s.value.toFixed(2)}`;
      return `<li><b>${NODE[s.node]}</b>${val}<br><small>${MODEL_NAME[c.model]} · ${c.region} cm⁻¹ · ${SCATTER[c.scatter]} · ${DERIV[c.deriv]}</small></li>`;
    }).join("");
    $("#d-warn").innerHTML = x.warnings.map(w => `<p class="warn">⚠ ${esc(w)}</p>`).join("");
  }

  function plot(full) {
    const W = 640, H = 240, L = 40, R = 10, T = 10, B = 28;
    const g = MODEL.grid, ref = MODEL.ref_spectrum.otantik, rs = MODEL.ref_spectrum.step;
    const lo = 800, hi = 4000;
    const pts = [];
    for (let i = 0; i < g.n; i += 2) if (!isNaN(full[i])) pts.push([g.start + g.step * i, full[i]]);
    const rp = ref.map((v, i) => [g.start + rs * i, v]);
    const ys = pts.concat(rp).map(p => p[1]);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = w => L + (hi - w) / (hi - lo) * (W - L - R);
    const sy = v => T + (y1 - v) / (y1 - y0 || 1) * (H - T - B);
    const path = p => p.map((q, i) => (i ? "L" : "M") + sx(q[0]).toFixed(1) + " " + sy(q[1]).toFixed(1)).join("");
    let ticks = "";
    for (let w = 4000; w >= 1000; w -= 500) ticks += `<text x="${sx(w)}" y="${H - 8}" text-anchor="middle">${w}</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Spektrum">
      <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="var(--line)"/>
      ${ticks}<text x="${L}" y="${T + 8}">A</text><text x="${W - R}" y="${H - 8}" text-anchor="end"> cm⁻¹</text>
      <path d="${path(rp)}" fill="none" stroke="var(--plot-b)" stroke-width="1.5"/>
      <path d="${path(pts)}" fill="none" stroke="var(--plot-a)" stroke-width="1.5"/></svg>`;
  }

  // ---------------------------------------------------------------- model özeti
  function showModels() {
    const m = MODEL;
    $("#m-info").textContent = `Model sürümü ${m.version} · eğitim: ${m.n_train.otantik} otantik, ${m.n_train.tagsisli} tağşişli spektrum, ${m.n_train.aile} aile.`;
    const perf = (t) => {
      const n = t.perf.nested;
      if (t.kind === "oc") return `özgüllük ${fmt(n.ozgulluk * 100, 0)}% · duyarlılık ${fmt(n.duyarlilik * 100, 0)}%`;
      if (t.kind === "clf") return `dengeli doğruluk ${fmt(n.dengeli_dogruluk * 100, 0)}%`;
      return `RMSEP ${fmt(n.RMSEP, 2)} · R² ${fmt(n.R2, 2)} · bias ${fmt(n.bias, 2)}`;
    };
    $("#m-table").innerHTML = `<thead><tr><th>Düğüm</th><th>Teknik</th><th>Ön işleme</th><th>Nested LOFO performansı</th></tr></thead><tbody>` +
      Object.entries(m.tasks).map(([k, t]) => {
        const c = t.config, hp = Object.entries(c).filter(([a]) => !["region", "scatter", "deriv", "model"].includes(a)).map(([a, v]) => `${a}=${v}`).join(", ");
        return `<tr><td>${NODE[k]}</td><td>${MODEL_NAME[c.model]} <span class="muted">${esc(hp)}</span></td><td>${c.region} · ${SCATTER[c.scatter]} · ${DERIV[c.deriv]}</td><td>${perf(t)}</td></tr>`;
      }).join("") + "</tbody>";
    const heat = (tbl, sel) => {
      const rowsK = Object.keys(tbl).filter(k => CLS[k]);
      const cols = Array.from(new Set(rowsK.flatMap(k => Object.keys(tbl[k])))).filter(c => rowsK.some(k => tbl[k][c] !== null)).sort((a, b) => a - b);
      $(sel).innerHTML = `<thead><tr><th></th>${cols.map(c => `<th class="num">%${c}</th>`).join("")}</tr></thead><tbody>` +
        rowsK.map(k => `<tr><td>${CLS[k]}</td>${cols.map(c => {
          const v = tbl[k][c];
          return `<td class="num">${v === null || v === undefined ? "" : Math.round(v * 100) + "%"}</td>`;
        }).join("")}</tr>`).join("") + "</tbody>";
    };
    heat(m.validation.t1_seviye_red, "#m-t1");
    heat(m.validation.uctan_uca_dogruluk, "#m-e2e");
    $("#foot").textContent = "Hesaplamaların tamamı tarayıcınızda yapılır; yüklenen dosyalar hiçbir sunucuya gönderilmez.";
  }
})();
