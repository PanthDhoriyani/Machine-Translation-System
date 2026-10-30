/**
 * app.js — Frontend logic for the Machine Translation UI
 * NLPA Project 2026 | Module 4
 *
 * Connects to the FastAPI backend (same origin).
 * FR-4.4: Web UI with text box, language selector, output
 * TC-4.5: UI reflects language selector choice
 * TC-4.8: Loading spinner visible during request
 */

const API_BASE = "";  // same origin

// ── Element refs ──────────────────────────────────────────────────────────
const srcText       = document.getElementById("src-text");
const outputText    = document.getElementById("output-text");
const srcLang       = document.getElementById("src-lang");
const tgtLang       = document.getElementById("tgt-lang");
const beamSize      = document.getElementById("beam-size");
const translateBtn  = document.getElementById("translate-btn");
const btnText       = document.getElementById("btn-text");
const btnSpinner    = document.getElementById("btn-spinner");
const swapBtn       = document.getElementById("swap-btn");
const copyBtn       = document.getElementById("copy-btn");
const charCount     = document.getElementById("char-count");
const latencyBadge  = document.getElementById("latency-badge");
const latencyVal    = document.getElementById("latency-val");
const modelUsed     = document.getElementById("model-used");
const errorBox      = document.getElementById("error-box");

const evalFile      = document.getElementById("eval-file");
const evalBtn       = document.getElementById("eval-btn");
const evalBtnText   = document.getElementById("eval-btn-text");
const evalSpinner   = document.getElementById("eval-spinner");
const uploadZone    = document.getElementById("upload-zone");
const browseBtn     = document.getElementById("browse-btn");
const scoresGrid    = document.getElementById("scores-grid");
const bleuVal       = document.getElementById("bleu-val");
const meteorVal     = document.getElementById("meteor-val");
const chrfVal       = document.getElementById("chrf-val");
const exportRow     = document.getElementById("export-row");
const exportCsvBtn  = document.getElementById("export-csv-btn");
const exportPdfBtn  = document.getElementById("export-pdf-btn");
const evalError     = document.getElementById("eval-error");

const miId     = document.getElementById("mi-id");
const miType   = document.getElementById("mi-type");
const miParams = document.getElementById("mi-params");
const miCached = document.getElementById("mi-cached");
const refreshModelBtn = document.getElementById("refresh-model-btn");

// Persisted evaluation data for export
let lastEvalReport = null;

// ── Char counter ──────────────────────────────────────────────────────────
srcText.addEventListener("input", () => {
  const len = srcText.value.length;
  charCount.textContent = len;
  if (len > 2000) srcText.value = srcText.value.slice(0, 2000);
});

// ── Swap languages ────────────────────────────────────────────────────────
swapBtn.addEventListener("click", () => {
  const tmp = srcLang.value;
  srcLang.value = tgtLang.value;
  tgtLang.value = tmp;
  // Also swap text
  const tmpText = srcText.value;
  srcText.value = outputText.innerText.replace(/\n/g, "") === "Translation will appear here…" ? "" : outputText.innerText;
  setOutput(tmpText || "");
});

// ── Translate ─────────────────────────────────────────────────────────────
translateBtn.addEventListener("click", doTranslate);
srcText.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") doTranslate();
});

async function doTranslate() {
  const text = srcText.value.trim();
  if (!text) { showError(errorBox, "Please enter some text to translate."); return; }

  setLoading(true);
  clearError(errorBox);
  latencyBadge.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        src_lang: srcLang.value,
        tgt_lang: tgtLang.value,
        beam_size: parseInt(beamSize.value, 10),
        decoding: beamSize.value === "1" ? "greedy" : "beam",
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    setOutput(data.translation);
    latencyVal.textContent = Math.round(data.latency_ms);
    modelUsed.textContent  = data.model_used.split("/").pop();
    latencyBadge.classList.remove("hidden");
    loadModelInfo();  // refresh model panel

  } catch (err) {
    showError(errorBox, `Translation failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  translateBtn.disabled = on;
  btnText.textContent   = on ? "Translating…" : "Translate";
  btnSpinner.classList.toggle("hidden", !on);
}

function setOutput(text) {
  outputText.innerHTML = "";
  if (text) {
    outputText.appendChild(document.createTextNode(text));
  } else {
    outputText.innerHTML = '<span class="placeholder">Translation will appear here…</span>';
  }
}

// ── Copy ──────────────────────────────────────────────────────────────────
copyBtn.addEventListener("click", async () => {
  const text = outputText.innerText;
  if (!text || text === "Translation will appear here…") return;
  await navigator.clipboard.writeText(text).catch(() => {});
  copyBtn.textContent = "✅";
  setTimeout(() => { copyBtn.textContent = "📋"; }, 1500);
});

// ── Evaluation ────────────────────────────────────────────────────────────
browseBtn.addEventListener("click", () => evalFile.click());
uploadZone.addEventListener("click", () => evalFile.click());

// Drag & drop
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) {
    evalFile.files = e.dataTransfer.files;
    onFileSelected(e.dataTransfer.files[0]);
  }
});

evalFile.addEventListener("change", () => {
  if (evalFile.files[0]) onFileSelected(evalFile.files[0]);
});

function onFileSelected(file) {
  const label = uploadZone.querySelector("p");
  if (label) label.innerHTML = `📄 <strong>${file.name}</strong> selected`;
  evalBtn.disabled = false;
}

evalBtn.addEventListener("click", async () => {
  const file = evalFile.files[0];
  if (!file) return;

  setEvalLoading(true);
  clearError(evalError);
  scoresGrid.classList.add("hidden");
  exportRow.classList.add("hidden");

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/evaluate/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    lastEvalReport = data;
    bleuVal.textContent   = data.bleu?.toFixed(2) ?? "—";
    meteorVal.textContent = data.meteor?.toFixed(4) ?? "—";
    chrfVal.textContent   = data.chrf?.toFixed(2) ?? "—";
    scoresGrid.classList.remove("hidden");
    exportRow.classList.remove("hidden");

  } catch (err) {
    showError(evalError, `Evaluation failed: ${err.message}`);
  } finally {
    setEvalLoading(false);
  }
});

function setEvalLoading(on) {
  evalBtn.disabled = on;
  evalBtnText.textContent = on ? "Evaluating…" : "Run Evaluation";
  evalSpinner.classList.toggle("hidden", !on);
}

// ── Export CSV (TC-4.7) ───────────────────────────────────────────────────
exportCsvBtn.addEventListener("click", async () => {
  const res = await fetch(`${API_BASE}/evaluate/export/csv`);
  if (!res.ok) return;
  const blob = await res.blob();
  triggerDownload(blob, "evaluation_report.csv", "text/csv");
});

// ── Export PDF ────────────────────────────────────────────────────────────
exportPdfBtn.addEventListener("click", () => {
  // Trigger browser PDF download via /evaluate/export/pdf if implemented
  window.open(`${API_BASE}/evaluate/export/pdf`, "_blank");
});

function triggerDownload(blob, filename, type) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Model Info ────────────────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) return;
    const data = await res.json();
    miId.textContent     = data.model_id || "—";
    miType.textContent   = data.type || "—";
    miParams.textContent = data.parameters_M ? `${data.parameters_M} M` : "—";
    miCached.textContent = data.cached ? "✅ Yes" : "❌ No";
  } catch (_) { /* silent */ }
}

refreshModelBtn.addEventListener("click", loadModelInfo);

// ── Error helpers ─────────────────────────────────────────────────────────
function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearError(el) { el.textContent = ""; el.classList.add("hidden"); }

// ── Init ──────────────────────────────────────────────────────────────────
loadModelInfo();
