// docs/engine.js çıktılarını export.py'nin Python tahminleriyle karşılaştırır: node train/test_engine.js
const fs = require("fs"), path = require("path");
const E = require("../docs/engine.js");
const docs = path.join(__dirname, "..", "docs");
const M = JSON.parse(fs.readFileSync(path.join(docs, "model.json")));
const R = JSON.parse(fs.readFileSync(path.join(docs, "test_ref.json")));
let worst = 0, fail = 0;
for (const [task, exp] of Object.entries(R.expected)) {
  const t = task === "domain" ? M.domain : M.tasks[task];
  let maxd = 0, signFlip = 0;
  R.X.forEach((x, i) => {
    const v = E.predict(t.model, E.prep(t.pipeline, Float64Array.from(x)));
    const d = Math.abs(v - exp[i]) / (Math.abs(exp[i]) + 1);
    maxd = Math.max(maxd, d);
    const thr = task.startsWith("T1") || task === "domain" ? 1 : 0;
    if (!task.startsWith("R_") && (v > thr) !== (exp[i] > thr)) signFlip++;
  });
  const ok = maxd < 1e-4 && signFlip === 0;
  if (!ok) fail++;
  worst = Math.max(worst, maxd);
  console.log(`${task.padEnd(18)} ${t.model.type.padEnd(8)} en büyük göreli fark ${maxd.toExponential(2)}  karar farkı ${signFlip}  ${ok ? "OK" : "HATA"}`);
}
// SPA ve TSV okuyucu
const raw = process.argv[2];
if (raw) {
  const a = E.readText(fs.readFileSync(raw + ".tsv", "utf8"))[0];
  const b = E.readSPA(new Uint8Array(fs.readFileSync(raw + ".spa")).buffer)[0];
  const ga = E.toGrid(a.x, a.y, M.grid), gb = E.toGrid(b.x, b.y, M.grid);
  let d = 0; ga.forEach((v, i) => { d = Math.max(d, Math.abs(v - gb[i])); });
  console.log("SPA vs TSV ızgara farkı:", d.toExponential(2), "| karar:", E.analyze(M, ga).label);
}
process.exit(fail ? 1 : 0);
