// Arayüz: model yönetimi, dosya yükleme, sonuç tablosu, detay, rapor (PDF/CSV), doğrulama özeti.
(function () {
  "use strict";
  const E = window.Engine;
  const $ = s => document.querySelector(s);
  let MODEL = null, MODEL_INFO = null, BUILTIN = null, rows = [];

  const MODEL_NAME = { ddsimca: "DD-SIMCA", pls: "PLS", pca_lda: "PCA-LDA", ridge: "Ridge", krr_rbf: "RBF çekirdek (LS-SVM)",
    knn: "kNN", mlp: "MLP (ANN)", cnn: "1D-CNN" };
  const SCATTER = { none: "—", snv: "SNV", msc: "MSC", emsc: "EMSC" };
  const DERIV = { none: "—", sg11: "SG düzleştirme (11)", d1w11: "SG 1. türev (11)", d1w21: "SG 1. türev (21)",
    d2w15: "SG 2. türev (15)", d2w25: "SG 2. türev (25)", d3w21: "SG 3. türev (21)", d3w31: "SG 3. türev (31)" };
  const NODE = { T1_otantik: "Otantiklik (OC-SIMCA)", T2_surup_meyve: "Şurup / meyve konsantresi", T3_glikoz_fruktoz: "Glikoz / fruktoz",
    T4_hurma_uzum: "Hurma / üzüm", R_glikoz: "Oran: glikoz", R_fruktoz: "Oran: fruktoz", R_hurma: "Oran: hurma", R_uzum: "Oran: üzüm" };
  const TREE = Object.keys(NODE);
  const CLS = { otantik: "Otantik", glikoz: "Glikoz", fruktoz: "Fruktoz", hurma: "Hurma", uzum: "Üzüm" };
  const LVL = { red: "Red", uyari: "Uyarı", ok: "Uygun" };
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v, d = 1) => v === null || v === undefined || isNaN(v) ? "—" : Number(v).toFixed(d);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* özel pencere */ } },
  };

  // ================================================================ model deposu (IndexedDB)
  const DB = {
    open() {
      return new Promise((res, rej) => {
        const r = indexedDB.open("elma-tagsis", 1);
        r.onupgradeneeded = () => r.result.createObjectStore("models", { keyPath: "id" });
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
    },
    async tx(mode, fn) {
      const db = await this.open();
      return new Promise((res, rej) => {
        const t = db.transaction("models", mode), s = t.objectStore("models");
        const out = fn(s);
        t.oncomplete = () => res(out && out.result !== undefined ? out.result : out);
        t.onerror = () => rej(t.error);
      });
    },
    all() { return this.tx("readonly", s => s.getAll()).catch(() => []); },
    put(rec) { return this.tx("readwrite", s => s.put(rec)); },
    del(id) { return this.tx("readwrite", s => s.delete(id)); },
  };

  function validate(m) {
    const miss = [];
    if (!m || typeof m !== "object") throw new Error("JSON nesnesi değil");
    if (!m.grid || !m.grid.n) miss.push("grid");
    if (!m.tasks) miss.push("tasks");
    else TREE.forEach(k => { if (!m.tasks[k] || !m.tasks[k].pipeline || !m.tasks[k].model) miss.push("tasks." + k); });
    if (!m.domain) miss.push("domain");
    if (miss.length) throw new Error("Eksik alanlar: " + miss.join(", "));
    // örnek bir spektrumla çalıştırılabilirlik testi
    const ref = m.ref_spectrum && m.ref_spectrum.otantik;
    if (ref) {
      const full = new Float64Array(m.grid.n);
      for (let i = 0; i < m.grid.n; i++) full[i] = ref[Math.min(ref.length - 1, Math.round(i * m.grid.step / m.ref_spectrum.step))];
      E.analyze(m, full, null);
    }
    return m;
  }

  async function boot() {
    try {
      BUILTIN = await fetch("model.json", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    } catch (e) { BUILTIN = null; }
    const active = store.get("activeModel");
    const saved = await DB.all();
    const rec = saved.find(r => r.id === active);
    if (rec) setModel(rec.model, { source: "yüklenen", id: rec.id, name: rec.name, added: rec.added });
    else if (BUILTIN) setModel(BUILTIN, { source: "yerleşik", id: "builtin" });
    else $("#model-active").innerHTML = '<span class="warn">Model bulunamadı. Bir model dosyası (.json) yükleyin.</span>';
    renderModelList();
  }

  function setModel(m, info) {
    MODEL = m;
    MODEL_INFO = info;
    store.set("activeModel", info.id);
    const n = m.n_train || {};
    $("#model-active").innerHTML = `Etkin model: <b>${esc(m.version || "?")}</b> <span class="tag small ${info.source === "yerleşik" ? "ok" : "err"}">${info.source}</span>` +
      (info.name ? ` <span class="muted">${esc(info.name)}</span>` : "") +
      `<div class="note">Eğitim: ${n.otantik ?? "?"} otantik, ${n.tagsisli ?? "?"} tağşişli spektrum, ${n.aile ?? "?"} aile` +
      (m.qc ? ` · ölçüm dönemi ${esc(m.qc.acquisition.period.join(" – "))}` : " · <span class='warn'>bu modelde ölçüm kabul kriterleri yok</span>") + "</div>";
    showModels();
    rows.forEach(r => { if (r.full) analyzeRow(r); });
    render();
    if ($("#detail").style.display === "block") $("#detail").style.display = "none";
  }

  async function renderModelList() {
    const saved = (await DB.all()).sort((a, b) => b.added.localeCompare(a.added));
    const line = (id, ver, name, src, added) => `<tr><td><b>${esc(ver)}</b> <span class="muted">${esc(name || "")}</span></td><td>${src}</td><td>${esc(added || "")}</td>` +
      `<td style="text-align:right;white-space:nowrap">${MODEL_INFO && MODEL_INFO.id === id ? '<span class="tag small ok">etkin</span>' : `<button data-use="${id}">Kullan</button>`}` +
      (id !== "builtin" ? ` <button data-del="${id}">Sil</button>` : "") + "</td></tr>";
    $("#model-list").innerHTML = "<thead><tr><th>Sürüm</th><th>Kaynak</th><th>Eklenme</th><th></th></tr></thead><tbody>" +
      (BUILTIN ? line("builtin", BUILTIN.version, "", "yerleşik (depodaki docs/model.json)", "") : "") +
      saved.map(r => line(r.id, r.model.version, r.name, "yüklenen", r.added)).join("") + "</tbody>";
    $("#model-list").querySelectorAll("[data-use]").forEach(b => b.onclick = async () => {
      const id = b.dataset.use;
      if (id === "builtin") setModel(BUILTIN, { source: "yerleşik", id });
      else { const r = (await DB.all()).find(x => x.id === id); setModel(r.model, { source: "yüklenen", id, name: r.name, added: r.added }); }
      renderModelList();
    });
    $("#model-list").querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
      if (!confirm("Bu model bu tarayıcıdan silinsin mi?")) return;
      await DB.del(b.dataset.del);
      if (MODEL_INFO.id === b.dataset.del && BUILTIN) setModel(BUILTIN, { source: "yerleşik", id: "builtin" });
      renderModelList();
    });
  }

  $("#model-file").addEventListener("change", async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const m = validate(JSON.parse(await f.text()));
      const rec = { id: "m" + Date.now(), name: f.name, added: new Date().toLocaleString("tr-TR"), model: m };
      await DB.put(rec).catch(() => { $("#model-msg").textContent = "Uyarı: model tarayıcıya kaydedilemedi; sayfa yenilenince kaybolur."; });
      setModel(m, { source: "yüklenen", id: rec.id, name: rec.name, added: rec.added });
      renderModelList();
      $("#model-msg").textContent = `"${f.name}" yüklendi ve etkinleştirildi (sürüm ${m.version || "?"}).`;
    } catch (err) {
      $("#model-msg").innerHTML = `<span class="warn">Model yüklenemedi: ${esc(err.message)}</span>`;
    }
  });
  $("#model-download").onclick = () => {
    if (!MODEL) return;
    download(new Blob([JSON.stringify(MODEL)], { type: "application/json" }), `model_${MODEL.version || "etkin"}.json`);
  };

  // ================================================================ dosya yükleme ve analiz
  const drop = $("#drop"), input = $("#file");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handle(e.dataTransfer.files));
  input.addEventListener("change", () => { handle(input.files); input.value = ""; });

  function analyzeRow(r) {
    try { r.res = E.analyze(MODEL, r.full, r.meta); r.error = null; }
    catch (err) { r.res = null; r.error = err.message; }
  }

  async function handle(files) {
    files = Array.from(files);
    $("#status").textContent = "Model hazırlanıyor…";
    await READY;
    if (!MODEL) { $("#status").textContent = "Önce bir model yükleyin."; return; }
    for (const f of files) {
      $("#status").textContent = f.name + " işleniyor…";
      await new Promise(r => setTimeout(r, 0));
      try {
        const isSpa = f.name.toLowerCase().endsWith(".spa");
        const specs = isSpa ? E.readSPA(await f.arrayBuffer()) : E.readText(await f.text());
        specs.forEach((s, i) => {
          const r = { name: s.name || (specs.length > 1 ? `${f.name} #${i + 1}` : f.name), meta: s.meta || null,
            full: E.toGrid(s.x, s.y, MODEL.grid), time: new Date() };
          analyzeRow(r);
          rows.push(r);
        });
      } catch (err) {
        rows.push({ name: f.name, error: err.message, time: new Date() });
      }
    }
    $("#status").textContent = rows.length + " spektrum analiz edildi.";
    render();
  }

  const verdictTag = x => x.verdict === "reddedildi" ? '<span class="tag rej">Reddedildi</span>'
    : x.verdict === "tagsisli" ? '<span class="tag bad">Tağşişli</span>' : '<span class="tag ok">Otantik</span>';

  function render() {
    const tb = $("#results tbody");
    $("#csv").disabled = $("#clear").disabled = $("#pdf").disabled = !rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="5" class="muted">Henüz dosya yüklenmedi.</td></tr>'; return; }
    tb.innerHTML = rows.map((r, i) => {
      if (r.error) return `<tr data-i="${i}"><td>${esc(r.name)}</td><td><span class="tag err">Hata</span></td><td></td><td></td><td class="warn">${esc(r.error)}</td></tr>`;
      const x = r.res, adult = x.verdict === "tagsisli";
      const lvl = adult ? `${fmt(x.level)} <span class="muted">± ${fmt(x.interval)}</span>` : "";
      const info = x.verdict === "reddedildi" ? `<span class="warn">${esc(x.rejectReasons[0])}${x.rejectReasons.length > 1 ? ` (+${x.rejectReasons.length - 1})` : ""}</span>`
        : x.warnings.length ? `<span class="warn">${x.warnings.length} uyarı</span>` : "";
      return `<tr data-i="${i}"><td>${esc(r.name)}</td><td>${verdictTag(x)}</td>` +
        `<td>${adult ? esc(x.label) + (x.lowConfidence ? ' <span class="warn">(düşük güven)</span>' : "") : ""}</td><td class="num">${lvl}</td><td>${info}</td></tr>`;
    }).join("");
    tb.querySelectorAll("tr").forEach(tr => tr.onclick = () => {
      tb.querySelectorAll("tr").forEach(t => t.classList.remove("sel"));
      tr.classList.add("sel");
      detail(rows[+tr.dataset.i]);
    });
  }

  $("#clear").onclick = () => { rows = []; render(); $("#detail").style.display = "none"; $("#status").textContent = ""; };

  // ================================================================ detay
  function pathHTML(x) {
    return x.steps.map(s => {
      const t = MODEL.tasks[s.node], c = t.config;
      let val;
      if (s.node === "T1_otantik") val = `f/f<sub>kritik</sub> = ${s.value.toFixed(3)} → ${s.value <= 1 ? "kabul (otantik)" : "red (tağşişli)"}`;
      else if (t.kind === "clf") val = `skor = ${s.value.toFixed(3)}${s.low ? ' <span class="warn">düşük güven</span>' : ""}`;
      else val = `tahmin = %${s.value.toFixed(2)}`;
      return `<li><b>${NODE[s.node]}</b>${val}<br><small>${MODEL_NAME[c.model]} · ${c.region} cm⁻¹ · ${SCATTER[c.scatter]} · ${DERIV[c.deriv]}</small></li>`;
    }).join("");
  }

  function qcHTML(qc) {
    return "<thead><tr><th>Kontrol</th><th>Değer</th><th>Kriter</th><th>Durum</th></tr></thead><tbody>" +
      qc.map(c => `<tr><td>${esc(c.item)}</td><td>${esc(c.value)}</td><td class="muted">${esc(c.rule)}</td><td class="lvl-${c.level}">${LVL[c.level]}</td></tr>`).join("") + "</tbody>";
  }

  function metaHTML(m) {
    if (!m) return "Ölçüm parametreleri: dosya .spa olmadığı için okunamadı.";
    return `Ölçüm: ${esc(m.collected || "?")} · ${esc(m.title || "")} · ${esc(m.units)} · ${m.resolution ?? "?"} cm⁻¹ · ${m.scans} tarama (arka plan ${m.bg_scans}) · ` +
      `arka plan yaşı ${m.bg_age_min ?? "?"} dk · ${esc(m.accessory || "?")} · cihaz ${esc(m.serial || "?")}`;
  }

  function verdictHTML(x) {
    if (x.verdict === "reddedildi") return `<div class="big" style="color:var(--rej)">Ölçüm reddedildi</div><div class="muted">Sonuç verilmedi. Numuneyi yeniden ölçün.</div>`;
    if (x.verdict === "tagsisli") return `<div class="big" style="color:var(--bad)">${esc(x.label)}</div><div>Tahmini oran: <b>%${fmt(x.level)}</b> <span class="muted">(± ${fmt(x.interval)}, %95 aralık)</span></div>`;
    return `<div class="big" style="color:var(--good)">Otantik</div><div class="muted">Tağşiş, yöntemin tespit sınırının altında</div>`;
  }

  function detail(r) {
    $("#detail").style.display = "block";
    $("#d-title").textContent = r.name;
    $("#d-plot").innerHTML = r.full ? plot(r.full) : "";
    if (r.error) {
      $("#d-verdict").innerHTML = '<span class="tag err">Hata</span>';
      $("#d-path").innerHTML = $("#d-qc").innerHTML = $("#d-meta").innerHTML = "";
      $("#d-warn").innerHTML = `<p class="warn">${esc(r.error)}</p>`;
      return;
    }
    const x = r.res;
    $("#d-verdict").innerHTML = verdictHTML(x);
    $("#d-path").innerHTML = x.verdict === "reddedildi" ? "" : pathHTML(x);
    $("#d-warn").innerHTML = (x.verdict === "reddedildi" ? x.rejectReasons.map(w => `<p class="warn">✕ ${esc(w)}</p>`) : x.warnings.map(w => `<p class="warn">⚠ ${esc(w)}</p>`)).join("");
    $("#d-qc").innerHTML = qcHTML(x.qc);
    $("#d-meta").innerHTML = metaHTML(r.meta);
  }

  function plot(full, W = 640, H = 240) {
    const L = 40, R = 10, T = 10, B = 28;
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

  // ================================================================ dışa aktarma
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  $("#csv").onclick = () => {
    const head = ["dosya", "karar", "tur", "tahmini_oran", "aralik_pm", "dusuk_guven", "OC_SIMCA_f_fkrit", "olcum_tarihi", "tarama",
      "cozunurluk", "arka_plan_yasi_dk", "aksesuar", "cihaz", "kontrol_red", "kontrol_uyari", "model_surumu", "analist", "not"];
    const lines = [head.join(";")].concat(rows.map(r => {
      const x = r.res || {}, m = r.meta || {};
      const t1 = x.steps && x.steps[0] ? x.steps[0].value.toFixed(3) : "";
      const qc = x.qc || [];
      return [r.name, r.error ? "HATA: " + r.error : x.verdict, x.cls || "", x.level !== undefined ? x.level.toFixed(1) : "",
        x.interval ? x.interval.toFixed(1) : "", x.lowConfidence ? "evet" : "", t1, m.collected || "", m.scans ?? "", m.resolution ?? "",
        m.bg_age_min ?? "", m.accessory || "", m.serial || "",
        qc.filter(c => c.level === "red").map(c => `${c.item}=${c.value}`).join(" | "),
        qc.filter(c => c.level === "uyari").map(c => `${c.item}=${c.value}`).join(" | "),
        MODEL.version, $("#analyst").value, $("#note").value].map(v => `"${String(v).replace(/"/g, '""')}"`).join(";");
    }));
    download(new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" }), `tagsis_sonuclari_${stamp()}.csv`);
  };

  const stamp = () => new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");

  $("#pdf").onclick = () => {
    const now = new Date().toLocaleString("tr-TR");
    const count = v => rows.filter(r => r.res && r.res.verdict === v).length;
    const summary = rows.map(r => {
      if (r.error) return `<tr><td>${esc(r.name)}</td><td>Hata</td><td colspan="3">${esc(r.error)}</td></tr>`;
      const x = r.res;
      return `<tr><td>${esc(r.name)}</td><td>${verdictTag(x)}</td><td>${x.verdict === "tagsisli" ? esc(x.label) + (x.lowConfidence ? " (düşük güven)" : "") : ""}</td>` +
        `<td class="num">${x.verdict === "tagsisli" ? `%${fmt(x.level)} ± ${fmt(x.interval)}` : ""}</td>` +
        `<td>${x.verdict === "reddedildi" ? esc(x.rejectReasons.join("; ")) : x.warnings.length ? x.warnings.length + " uyarı" : "—"}</td></tr>`;
    }).join("");
    const details = rows.filter(r => !r.error).map(r => {
      const x = r.res;
      return `<div class="sample"><h3>${esc(r.name)} — ${verdictTag(x)} ${x.verdict === "tagsisli" ? esc(x.label) + ` %${fmt(x.level)} ± ${fmt(x.interval)}` : ""}</h3>` +
        `<p class="small">${metaHTML(r.meta)}</p>` +
        (x.verdict !== "reddedildi" ? `<table><thead><tr><th>Karar düğümü</th><th>Sonuç</th><th>Model</th></tr></thead><tbody>` +
          x.steps.map(s => {
            const c = MODEL.tasks[s.node].config, k = MODEL.tasks[s.node].kind;
            const v = s.node === "T1_otantik" ? `f/f_kritik = ${s.value.toFixed(3)} (${s.value <= 1 ? "kabul" : "red"})` : k === "clf" ? `skor ${s.value.toFixed(3)}${s.low ? " · düşük güven" : ""}` : `%${s.value.toFixed(2)}`;
            return `<tr><td>${NODE[s.node]}</td><td>${v}</td><td class="small">${MODEL_NAME[c.model]} · ${c.region} · ${SCATTER[c.scatter]} · ${DERIV[c.deriv]}</td></tr>`;
          }).join("") + "</tbody></table>" : "") +
        `<p class="small"><b>Ölçüm kontrolü:</b> ${x.qc.filter(c => c.level === "ok").length}/${x.qc.length} uygun` +
        x.qc.filter(c => c.level !== "ok").map(c => ` · <b>${LVL[c.level]}</b> ${esc(c.item)} = ${esc(c.value)} (${esc(c.rule)})`).join("") + "</p>" +
        (x.warnings.length ? `<p class="small">Uyarılar: ${x.warnings.map(esc).join(" · ")}</p>` : "") + "</div>";
    }).join("");
    $("#report").innerHTML = `
      <h1>Elma Konsantresi Tağşiş Tespiti — Analiz Raporu</h1>
      <div class="meta"><div><b>Rapor tarihi:</b> ${esc(now)}</div><div><b>Analizi yapan:</b> ${esc($("#analyst").value || "—")}</div>
        <div><b>Model sürümü:</b> ${esc(MODEL.version)} (${esc(MODEL_INFO.source)})</div><div><b>Not:</b> ${esc($("#note").value || "—")}</div>
        <div><b>Spektrum sayısı:</b> ${rows.length} (otantik ${count("otantik")}, tağşişli ${count("tagsisli")}, reddedilen ${count("reddedildi")})</div></div>
      <h2>Özet</h2>
      <table><thead><tr><th>Dosya</th><th>Karar</th><th>Tür</th><th class="num">Oran</th><th>Uyarı / red nedeni</th></tr></thead><tbody>${summary}</tbody></table>
      <h2>Numune ayrıntıları</h2>${details}
      <h2>Yöntem ve sınırlamalar</h2><ul class="small">${$("#method").innerHTML}</ul>
      <p class="small">Model doğrulama performansları (nested leave-one-family-out): ${Object.entries(MODEL.tasks).map(([k, t]) => `${NODE[k]}: ${perf(t)}`).join("; ")}.</p>`;
    window.print();
  };

  // ================================================================ model özeti
  function perf(t) {
    const n = t.perf.nested;
    if (t.kind === "oc") return `özgüllük ${fmt(n.ozgulluk * 100, 0)}%, ORG kabul ${fmt((n.ORG_kabul ?? NaN) * 100, 0)}%`;
    if (t.kind === "clf") return `dengeli doğruluk ${fmt(n.dengeli_dogruluk * 100, 0)}%`;
    const a = t.perf.nested_tagsisli;
    return `RMSEP ${fmt(n.RMSEP, 2)}` + (a ? ` (tağşişlilerde ${fmt(a.RMSEP, 2)})` : "") + `, R² ${fmt(n.R2, 2)}`;
  }

  function showModels() {
    const m = MODEL;
    $("#m-info").textContent = `Model sürümü ${m.version} · eğitim: ${m.n_train.otantik} otantik, ${m.n_train.tagsisli} tağşişli spektrum, ${m.n_train.aile} aile.`;
    $("#m-table").innerHTML = `<thead><tr><th>Düğüm</th><th>Teknik</th><th>Ön işleme</th><th>Nested LOFO performansı</th></tr></thead><tbody>` +
      Object.entries(m.tasks).map(([k, t]) => {
        const c = t.config, hp = Object.entries(c).filter(([a]) => !["region", "scatter", "deriv", "model"].includes(a)).map(([a, v]) => `${a}=${v}`).join(", ");
        return `<tr><td>${NODE[k]}</td><td>${MODEL_NAME[c.model]} <span class="muted">${esc(hp)}</span></td><td>${c.region} · ${SCATTER[c.scatter]} · ${DERIV[c.deriv]}</td><td>${perf(t)}</td></tr>`;
      }).join("") + "</tbody>";
    const heat = (tbl, sel) => {
      if (!tbl) { $(sel).innerHTML = ""; return; }
      const rowsK = Object.keys(tbl).filter(k => CLS[k]);
      const cols = Array.from(new Set(rowsK.flatMap(k => Object.keys(tbl[k])))).filter(c => rowsK.some(k => tbl[k][c] !== null)).sort((a, b) => a - b);
      $(sel).innerHTML = `<thead><tr><th></th>${cols.map(c => `<th class="num">%${c}</th>`).join("")}</tr></thead><tbody>` +
        rowsK.map(k => `<tr><td>${CLS[k]}</td>${cols.map(c => {
          const v = tbl[k][c];
          return `<td class="num">${v === null || v === undefined ? "" : Math.round(v * 100) + "%"}</td>`;
        }).join("")}</tr>`).join("") + "</tbody>";
    };
    heat(m.validation && m.validation.t1_seviye_red, "#m-t1");
    heat(m.validation && m.validation.uctan_uca_dogruluk, "#m-e2e");
    if (m.qc) {
      const a = m.qc.acquisition, s = m.qc.spectral, f = v => Number(v).toFixed(3);
      const rr = [["Veri birimi", a.units.join(", "), "red"], ["Çözünürlük", a.resolution.join(", ") + " cm⁻¹", "red"],
        ["Örnek tarama sayısı", `≥ ${a.scans_typ} (altı uyarı, < ${a.scans_min} red)`, "red/uyarı"], ["Arka plan tarama sayısı", `≥ ${a.bg_scans_min}`, "red"],
        ["Örnekleme aksesuarı", a.accessory.join(", "), "red"], ["Cihaz", a.serial.join(", "), "uyarı"], ["Arka plan yaşı", `≤ ${a.bg_age_max} dk`, "red"],
        ["Spektral aralık", `${a.xmin_max}–${a.xmax_min} cm⁻¹ kapsanmalı`, "red"],
        [s.noise.label, `uyarı > ${s.noise.warn.toFixed(1)}, red > ${s.noise.reject.toFixed(1)}`, "red/uyarı"],
        [s.fp.label, `red < ${f(s.fp.lo)}, uyarı > ${f(s.fp.warn_hi)}`, "red/uyarı"],
        [s.oh.label, `uyarı ${f(s.oh.warn_lo)}–${f(s.oh.warn_hi)} dışı, red < ${f(s.oh.lo)}`, "red/uyarı"],
        [s.base.label, `|A| ≤ ${f(s.base.abs_max)}`, "red"], [s.minA.label, `≥ ${f(s.minA.lo)}`, "red"],
        ["Eğitim verisine benzerlik (DD-SIMCA, α = 0,001)", "f/f_kritik ≤ 1", "red"]];
      $("#m-qc").innerHTML = "<thead><tr><th>Kontrol</th><th>Kriter</th><th>Sonuç</th></tr></thead><tbody>" +
        rr.map(r => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td class="muted">${r[2]}</td></tr>`).join("") + "</tbody>" +
        `<caption class="note" style="caption-side:bottom;text-align:left;padding-top:6px">Kriterler eğitim verisinin ölçüm profilinden türetilmiştir (${esc(a.period.join(" – "))}). Spektral kontroller eğitim spektrumlarının ${m.qc.train_rejected}/${m.qc.train_n} tanesini reddederdi.</caption>`;
    } else $("#m-qc").innerHTML = '<tbody><tr><td class="warn">Bu modelde ölçüm kabul kriterleri tanımlı değil.</td></tr></tbody>';
    $("#method").innerHTML = [
      "Her spektrum önce ölçüm uygunluk kontrolünden geçer: .spa başlığındaki parametreler (veri birimi, çözünürlük, tarama sayısı, aksesuar, arka plan yaşı) ve spektral metrikler (gürültü, kristal teması, baz çizgisi) eğitim verisinin profiliyle karşılaştırılır; uygunsuz ölçümler reddedilir ve sonuç verilmez.",
      "Her karar düğümü için ön işleme (bölge × SNV/MSC/EMSC × Savitzky-Golay türevleri) ve teknik (DD-SIMCA, PLS, PLS-DA, PCA-LDA, ridge, RBF çekirdek, kNN, MLP, 1D-CNN) kombinasyonları tarandı; seçim aile bazlı (leave-one-family-out) çapraz doğrulama ve 1-SE kuralıyla yapıldı.",
      "Raporlanan performanslar nested LOFO'dur: test edilen partinin hiçbir örneği (ebeveyn elma dahil) model kurulumunda ve seçiminde kullanılmamıştır.",
      "OC-SIMCA kabul sınırı, görülmemiş otantik örneklerin en az %95'i kabul edilecek şekilde seçildi; seçimde bir partinin tağşişli örneğinin reddi yalnızca aynı partinin saf elması kabul edildiyse sayıldı.",
      "Düşük tağşiş oranları ve özellikle fruktoz şurubu FTIR ile zor ayırt edilir; \"otantik\" kararı tağşiş olmadığını kanıtlamaz, yalnızca bu yöntemin tespit sınırının altında olduğunu gösterir.",
      "Modeller tek laboratuvarın 2025 örnekleriyle ve her adulterantın sınırlı sayıda lotuyla kurulmuştur; farklı cihaz, bölge, çeşit ya da adulterant lotlarında performans düşebilir. İki adulterantın karışımı modellenmemiştir. Sonuçlar tarama amaçlıdır; resmi karar için doğrulayıcı analizlerle desteklenmelidir.",
    ].map(s => `<li>${esc(s)}</li>`).join("");
  }

  const READY = boot();
})();
