/* Table-Aware RAG — frontend logic (vanilla JS, fetch API) */

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const composer = $("composer");
const input = $("questionInput");
const sendBtn = $("sendBtn");

/* ---------------- composer state + empty state ---------------- */

const EMPTY_STATE_HTML = document.getElementById("emptyState").outerHTML;

function setComposerEnabled(on) {
  input.disabled = !on;
  sendBtn.disabled = !on;
  input.placeholder = on
    ? "Ask about the document (numbers welcome)…"
    : "Upload a PDF to start…";
}

function restoreEmptyState() {
  chat.innerHTML = EMPTY_STATE_HTML;
  $("clearChatBtn").hidden = true;
}

/* ---------------- health check ---------------- */

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    $("healthDot").className = "health-dot " + (d.openai_key_set ? "ok" : "warn");
    $("healthText").textContent = d.openai_key_set
      ? "backend ready"
      : "OPENAI_API_KEY not set";
    if (d.document) setComposerEnabled(true);
  } catch {
    $("healthDot").className = "health-dot";
    $("healthText").textContent = "backend unreachable";
  }
}
checkHealth();

/* ---------------- helpers ---------------- */

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function scrollDown() { chat.scrollTop = chat.scrollHeight; }

function hideEmptyState() {
  const es = $("emptyState");
  if (es) es.remove();
  $("clearChatBtn").hidden = false;
}

function badgeFor(route) {
  const kind = route.startsWith("sql") ? "sql"
             : route.startsWith("vector") ? "vector" : "hybrid";
  const b = el("span", `badge badge-${kind}`, route.toUpperCase());
  return b;
}

function renderMiniTable(columns, rows) {
  const wrap = el("div", "mini-table");
  const table = el("table");
  const thead = el("thead"), trh = el("tr");
  columns.forEach(c => trh.appendChild(el("th", null, c)));
  thead.appendChild(trh);
  const tbody = el("tbody");
  rows.forEach(r => {
    const tr = el("tr");
    (Array.isArray(r) ? r : columns.map(c => r[c]))
      .forEach(v => tr.appendChild(el("td", null, v === null ? "—" : String(v))));
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  wrap.appendChild(table);
  return wrap;
}

function renderCodeBlock(code) {
  const block = el("div", "codeblock");
  block.textContent = code;
  const copy = el("button", "copy", "copy");
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(code);
    copy.textContent = "copied";
    setTimeout(() => (copy.textContent = "copy"), 1200);
  });
  block.appendChild(copy);
  return block;
}

/* ---------------- upload ---------------- */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) ingest(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) ingest(fileInput.files[0]);
});

async function ingest(file) {
  const status = $("ingestStatus");
  status.innerHTML = "";
  status.append(el("span", "fname", file.name), "Extracting tables & text…");

  const form = new FormData();
  form.append("file", file);

  try {
    const r = await fetch("/api/ingest", { method: "POST", body: form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Ingestion failed");

    status.innerHTML = "";
    status.append(
      el("span", "fname", d.filename),
      Object.assign(el("span", "ok"),
        { textContent: `${d.tables.length} tables → SQLite · ${d.chunk_count} chunks → FAISS` })
    );
    const rm = el("button", "remove-btn", "✕ remove file");
    rm.addEventListener("click", removeDocument);
    status.appendChild(rm);
    renderTableList(d.tables);
    setComposerEnabled(true);
  } catch (err) {
    status.innerHTML = "";
    status.append(el("span", "err", err.message));
  }
}

function renderTableList(tables) {
  const section = $("tablesSection");
  const list = $("tableList");
  list.innerHTML = "";
  section.hidden = tables.length === 0;

  tables.forEach(t => {
    const chip = el("button", "table-chip");
    chip.append(el("span", null, t.name), el("span", "rows", `${t.total_rows} rows`));
    chip.addEventListener("click", () => showTablePreview(t.name));
    list.appendChild(chip);
  });
}

async function showTablePreview(name) {
  hideEmptyState();
  const r = await fetch(`/api/tables/${encodeURIComponent(name)}?limit=10`);
  const d = await r.json();
  const card = el("div", "sys-card");
  card.appendChild(el("h3", null, `${d.name} — first ${d.rows.length} of ${d.total_rows} rows`));
  card.appendChild(renderMiniTable(d.columns, d.rows));
  chat.appendChild(card);
  scrollDown();
}

/* ---------------- chat ---------------- */

composer.addEventListener

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  hideEmptyState();

  const userMsg = el("div", "msg user");
  userMsg.appendChild(el("div", "bubble", q));
  chat.appendChild(userMsg);

  const pending = el("div", "typing", "routing → retrieving → answering…");
  chat.appendChild(pending);
  scrollDown();
  sendBtn.disabled = true;

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Request failed");
    pending.replaceWith(renderAssistant(d));
  } catch (err) {
    pending.replaceWith(renderAssistantError(err.message));
  } finally {
    sendBtn.disabled = false;
    scrollDown();
    input.focus();
  }
});

function renderAssistant(d) {
  const msg = el("div", "msg assistant");

  const meta = el("div", "msg-meta");
  meta.appendChild(badgeFor(d.route || "hybrid"));
  msg.appendChild(meta);

  msg.appendChild(el("div", "bubble", d.answer));

  const evidence = el("details", "evidence");
  evidence.appendChild(el("summary", null, "evidence — how this was answered"));

  if (d.sql) {
    evidence.appendChild(renderCodeBlock(d.sql));
    if (d.rows && d.rows.length) {
      evidence.appendChild(renderMiniTable(Object.keys(d.rows[0]), d.rows));
    }
  }
  if (d.sql_error) {
    evidence.appendChild(el("div", "chunk", `SQL path failed: ${d.sql_error}`));
  }
  (d.chunks || []).forEach((c, i) => {
    const ch = el("div", "chunk");
    ch.appendChild(el("span", "score", `chunk ${i + 1} · L2 distance ${c.score}`));
    ch.append(c.text);
    evidence.appendChild(ch);
  });

  if (d.sql || d.chunks) msg.appendChild(evidence);
  return msg;
}

function renderAssistantError(message) {
  const msg = el("div", "msg assistant");
  msg.appendChild(el("div", "bubble", `⚠ ${message}`));
  return msg;
}

/* ---------------- remove document / clear chat ---------------- */

async function removeDocument() {
  await fetch("/api/document", { method: "DELETE" });
  $("ingestStatus").innerHTML = "";
  $("fileInput").value = "";
  $("tablesSection").hidden = true;
  $("tableList").innerHTML = "";
  restoreEmptyState();          // chat answers referenced this doc — clear them too
  setComposerEnabled(false);
}

$("clearChatBtn").addEventListener("click", restoreEmptyState);