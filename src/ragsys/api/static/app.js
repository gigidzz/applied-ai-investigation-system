"use strict";

// Plain DOM, no framework and no build step. The deployment image has no node
// toolchain, and a page that cannot fail to build cannot fail to deploy.

const $ = (id) => document.getElementById(id);

const CITATION = /\[(S\d+)\]/g;

/** Escape before inserting into innerHTML. Source text is document content,
 *  and a document in the corpus is free to contain angle brackets. */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/** Turn [S3] into a button that reveals that source. */
function linkCitations(text, validLabels) {
  return escapeHtml(text).replace(CITATION, (_, label) => {
    const known = validLabels.has(label) ? "" : " missing";
    return `<cite class="ref${known}" data-label="${label}" title="show ${label}">[${label}]</cite>`;
  });
}

async function call(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail =
      body?.detail ??
      body?.error ??
      (Array.isArray(body?.detail) ? body.detail[0]?.msg : null) ??
      `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

// ---------- health ----------

async function refreshHealth() {
  const el = $("health");
  try {
    const h = await call("/api/health");
    const chunks = h.chunks === null ? "unknown" : h.chunks.toLocaleString();
    const ready = h.model_loaded
      ? "search model ready"
      : "search model loads on the first question";
    el.textContent =
      `${chunks} searchable passages indexed · ${ready}`;
    el.title =
      `status ${h.status} · Qdrant collection "${h.collection}" · ` +
      `embedding model ${h.embedding_model}`;
    el.className = `health ${h.status === "ok" ? "ok" : "bad"}`;
    if (h.detail) el.textContent += ` · ${h.detail}`;
  } catch (err) {
    el.textContent = `the index is unavailable: ${err.message}`;
    el.className = "health bad";
  }
}

// ---------- ask ----------

function renderMeta(answer) {
  const rows = [
    ["grounded", String(answer.grounded)],
    ["confidence", answer.confidence],
    ["sources retrieved", String(answer.sources.length)],
    ["tool calls", String(answer.tool_calls)],
    ["tokens", `${answer.prompt_tokens} in / ${answer.output_tokens} out`],
    ["model", answer.model || "unknown"],
  ];
  if (answer.steps.length) rows.push(["steps", answer.steps.join(" → ")]);
  return rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("");
}

function renderSources(answer) {
  const cited = new Set(answer.citations.map((c) => c.label));
  return answer.sources
    .map((s) => {
      const facts = [s.kind, s.date, s.service].filter(Boolean).map(escapeHtml).join(" · ");
      return `<li id="src-${escapeHtml(s.label)}" class="${cited.has(s.label) ? "cited" : ""}">
        <div class="src-head">
          <span class="src-label">${escapeHtml(s.label)}</span>
          <span class="src-path">${escapeHtml(s.source_path)}</span>
          <span class="src-facts">${facts}${facts ? " · " : ""}score ${s.score.toFixed(5)}</span>
        </div>
        <div class="src-text">${escapeHtml(s.text)}</div>
      </li>`;
    })
    .join("");
}

function renderAnswer(answer) {
  const valid = new Set(answer.citations.map((c) => c.label));
  const flags = [];
  if (answer.abstained) flags.push('<span class="flag abstained">abstained</span>');
  flags.push(
    answer.grounded
      ? '<span class="flag grounded">grounded</span>'
      : '<span class="flag ungrounded">not grounded</span>',
  );
  if (answer.contradictions.length)
    flags.push('<span class="flag conflict">sources disagree</span>');
  if (answer.invalid_citations.length)
    flags.push(
      `<span class="flag ungrounded">invented citation: ${escapeHtml(
        answer.invalid_citations.join(", "),
      )}</span>`,
    );
  $("flags").innerHTML = flags.join("");

  const body = [];
  if (answer.abstained && answer.abstain_reason) {
    body.push(`<p><strong>No answer in the documents.</strong> ${escapeHtml(answer.abstain_reason)}</p>`);
  }
  if (answer.answer) body.push(linkCitations(answer.answer, valid));
  $("answer-text").innerHTML = body.join("");

  $("contradictions").innerHTML = answer.contradictions
    .map(
      (c) =>
        `<div class="conflict-box"><strong>Sources disagree.</strong> ${escapeHtml(
          c.description,
        )} <span class="hint">(${escapeHtml(c.labels.join(", "))})</span></div>`,
    )
    .join("");

  $("meta").innerHTML = renderMeta(answer);
  $("sources").innerHTML = renderSources(answer);

  // When the system abstains the sources are still worth showing — they are
  // what it looked at before deciding it could not answer — but presenting
  // fifteen unrelated passages as prominently as a real answer's evidence
  // reads like a malfunction. So they are collapsed and relabelled.
  const details = $("sources-wrap");
  const summary = $("sources-summary");
  details.open = !answer.abstained;
  summary.innerHTML = answer.abstained
    ? `Searched ${answer.sources.length} passages &mdash; <span class="hint">none of them answered the question</span>`
    : `Sources <span class="hint">the model saw all of these; cited ones are marked</span>`;
}

async function ask(question) {
  const button = $("ask-button");
  button.disabled = true;
  button.textContent = "Thinking…";
  $("flags").innerHTML = "";
  $("answer-text").innerHTML = '<span class="hint">retrieving and reasoning…</span>';
  $("contradictions").innerHTML = "";
  $("meta").innerHTML = "";
  $("sources").innerHTML = "";
  $("sources-summary").textContent = "Sources";
  $("answer").hidden = false;
  try {
    renderAnswer(
      await call("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      }),
    );
  } catch (err) {
    $("answer-text").innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

// ---------- documents ----------

async function refreshDocuments() {
  const list = $("documents");
  try {
    const data = await call("/api/documents?limit=300");
    $("documents-summary").textContent =
      `${data.shown} documents you can manage · ` +
      `${data.total_chunks.toLocaleString()} passages indexed in total ` +
      `(the bulk Wikipedia corpus is searchable but not listed here)`;
    // A list rather than a table: the panel is too narrow for five columns,
    // and a path has no spaces to wrap at, so columns either collide or
    // overflow. Two lines per document reads cleanly at any width.
    list.innerHTML = data.documents
      .map((d) => {
        const passages = `${d.chunk_count} passage${d.chunk_count === 1 ? "" : "s"}`;
        const facts = [d.format, d.date || "no date", passages].map(escapeHtml).join(" · ");
        return `<li>
          <div class="doc-path" title="${escapeHtml(d.source_path)}">${escapeHtml(d.source_path)}</div>
          <div class="doc-row">
            <span class="doc-facts">${facts}</span>
            <button class="delete" data-path="${escapeHtml(d.source_path)}"
              title="remove this document and all of its passages">delete</button>
          </div>
        </li>`;
      })
      .join("");
  } catch (err) {
    list.innerHTML = `<li class="error">${escapeHtml(err.message)}</li>`;
  }
}

async function upload(file) {
  const button = $("upload-button");
  const out = $("upload-result");
  button.disabled = true;
  out.textContent = `indexing ${file.name}…`;
  out.className = "hint";
  try {
    const form = new FormData();
    form.append("file", file);
    const result = await call("/api/documents", { method: "POST", body: form });
    out.textContent =
      `indexed ${result.filename}: ${result.documents} document(s), ` +
      `${result.chunks_written} searchable passage(s)`;
    out.className = "ok";
    $("file").value = "";
    await Promise.all([refreshDocuments(), refreshHealth()]);
  } catch (err) {
    out.textContent = err.message;
    out.className = "error";
  } finally {
    button.disabled = false;
  }
}

async function remove(path) {
  if (!window.confirm(`Remove the document ${path} from the index?`)) return;
  try {
    const result = await call(`/api/documents?source_path=${encodeURIComponent(path)}`, {
      method: "DELETE",
    });
    $("upload-result").textContent =
      `removed ${path} from the index (${result.chunks_removed} passage(s))`;
    $("upload-result").className = "ok";
    await Promise.all([refreshDocuments(), refreshHealth()]);
  } catch (err) {
    $("upload-result").textContent = err.message;
    $("upload-result").className = "error";
  }
}

// ---------- wiring ----------

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question.length >= 3) ask(question);
});

// A textarea swallows Enter, so the obvious way to ask a question does
// nothing. Enter submits; Shift+Enter still inserts a newline for anyone
// typing a multi-line question.
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});

$("upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (file) upload(file);
});

document.addEventListener("click", (event) => {
  const example = event.target.closest("button.example");
  if (example) {
    $("question").value = example.textContent.trim();
    ask(example.textContent.trim());
    return;
  }
  const ref = event.target.closest("cite.ref");
  if (ref) {
    const target = $(`src-${ref.dataset.label}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1200);
    }
    return;
  }
  const del = event.target.closest("button.delete");
  if (del) remove(del.dataset.path);
});

refreshHealth();
refreshDocuments();
