/*
 * assets/edit.js — in-browser document editing for the Knowledge Base
 * viewer. NOT part of the forked OKF viewer (see viz.js's header comment
 * for the license/provenance of the graph code this file sits on top of);
 * this is new code written for md-kb-rag's web UI.
 *
 * Talks to the write pipeline via the fixed HTTP API contract:
 *   GET    api/schema/<path>  -> schema hints panel + new-document template
 *   GET    api/doc/<path>     -> load content + content_hash for editing
 *   POST   api/doc/<path>     -> create/edit
 *                                {content, commit_message, create, expected_hash?}
 *   DELETE api/doc/<path>     -> delete {commit_message}
 *
 * Editor engine: a plain <textarea> plus a live marked.js preview pane —
 * "Plan B" from the implementation plan (no vendored CodeMirror bundle).
 * marked.js is already vendored for the detail panel's body renderer in
 * viz.js; this file reuses the same global `marked`, so no new vendor file
 * or asset route is needed to support editing itself.
 *
 * Coordinates with viz.js exclusively through `window.KBViz` (see that
 * file's header comment and the `window.KBViz = {...}` assignment at its
 * bottom) rather than reaching into its closed-over state directly.
 *
 * All fetch URLs are relative (`api/...`), matching viz.js, so the app
 * keeps working behind a path-prefixed reverse proxy.
 */
(function () {
  "use strict";

  const PREVIEW_DEBOUNCE_MS = 150;

  const els = {};

  // Editor session state. `mode` is null when the overlay is closed.
  let mode = null; // "create" | "edit"
  let originalPath = null;
  let originalHash = null; // content_hash of the last-loaded/last-saved content
  let previewTimer = null;
  let confirmCallback = null;

  function $(id) {
    return document.getElementById(id);
  }

  function init() {
    els.overlay = $("editor-overlay");
    els.pathInput = $("editor-path");
    els.modeBadge = $("editor-mode-badge");
    els.status = $("editor-status");
    els.saveBtn = $("editor-save");
    els.cancelBtn = $("editor-cancel");
    els.errors = $("editor-errors");
    els.schemaContent = $("editor-schema-content");
    els.textarea = $("editor-textarea");
    els.preview = $("editor-preview");

    els.newDocBtn = $("new-doc-btn");
    els.editBtn = $("edit-btn");
    els.deleteBtn = $("delete-btn");

    els.confirmOverlay = $("confirm-overlay");
    els.confirmMessage = $("confirm-message");
    els.confirmOk = $("confirm-ok");
    els.confirmCancel = $("confirm-cancel");

    if (!els.overlay || !window.KBViz) return; // markup/viz.js not present

    els.newDocBtn.addEventListener("click", openCreate);
    els.editBtn.addEventListener("click", openEditCurrent);
    els.deleteBtn.addEventListener("click", confirmDeleteCurrent);

    els.cancelBtn.addEventListener("click", closeEditor);
    els.saveBtn.addEventListener("click", save);
    els.textarea.addEventListener("input", schedulePreview);
    els.pathInput.addEventListener("blur", onPathBlur);

    els.overlay.addEventListener("click", (e) => {
      if (e.target === els.overlay) closeEditor();
    });

    els.confirmCancel.addEventListener("click", hideConfirm);
    els.confirmOk.addEventListener("click", () => {
      const cb = confirmCallback;
      hideConfirm();
      if (cb) cb();
    });
    els.confirmOverlay.addEventListener("click", (e) => {
      if (e.target === els.confirmOverlay) hideConfirm();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (!els.confirmOverlay.hidden) hideConfirm();
        else if (!els.overlay.hidden) closeEditor();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "s" && !els.overlay.hidden) {
        e.preventDefault();
        save();
      }
    });
  }

  // -------------------------------------------------------------------
  // Opening the editor
  // -------------------------------------------------------------------

  function openEditCurrent() {
    const id = window.KBViz.getCurrentId();
    if (!id) return;
    openEditor("edit", id);
  }

  async function openCreate() {
    const raw = window.prompt(
      "New document path (repo-relative, e.g. domain/subdir/file.md):",
      ""
    );
    if (raw === null) return;
    const path = raw.trim().replace(/^\/+/, "");
    if (!path) return;
    if (!path.toLowerCase().endsWith(".md")) {
      window.alert("Path must end in .md");
      return;
    }
    if (window.KBViz.getNodeIds().includes(path)) {
      window.alert(
        "A document already exists at that path. Use Edit instead."
      );
      return;
    }
    await openEditor("create", path);
  }

  async function openEditor(newMode, path) {
    mode = newMode;
    originalPath = path;
    originalHash = null;
    clearErrors();
    setStatus("", null);
    updateModeBadge();

    els.pathInput.value = path;
    if (mode === "edit") {
      els.pathInput.setAttribute("readonly", "readonly");
    } else {
      els.pathInput.removeAttribute("readonly");
    }

    els.textarea.value = "";
    els.preview.innerHTML = "";
    els.overlay.hidden = false;

    const schemaPromise = fetchSchema(path);

    if (mode === "edit") {
      els.textarea.disabled = true;
      els.textarea.value = "Loading…";
      // Disabled until a fresh load confirms we have a content_hash to key
      // the save on — otherwise a failed/aborted load would leave Save
      // clickable with a stale (or missing) originalHash. See save()'s own
      // belt-and-suspenders guard for non-click save paths (Cmd/Ctrl+S).
      disableSave("Waiting for the document to load before you can save.");
      try {
        const res = await fetch(`api/doc/${window.KBViz.encodePathForApi(path)}`);
        const data = await safeJson(res);
        if (!res.ok) {
          els.textarea.value = "";
          showError(
            `Could not load document (HTTP ${res.status}): ${
              errorMessageFrom(data) || "no further detail from the server."
            }`
          );
          disableSave("The document failed to load — reload before saving.");
        } else {
          originalHash = data.content_hash;
          els.textarea.value = data.content || "";
          enableSave();
        }
      } catch (err) {
        els.textarea.value = "";
        showError(`Could not load document: ${err.message}`);
        disableSave("The document failed to load — reload before saving.");
      } finally {
        els.textarea.disabled = false;
        renderPreview();
      }
    } else {
      enableSave();
      const schema = await schemaPromise;
      els.textarea.value = buildTemplate(schema, path);
      renderPreview();
    }

    await schemaPromise;
  }

  /** Disable the Save button and surface why via its tooltip. Used whenever
   * an edit-mode load hasn't (yet) produced a trustworthy `originalHash` —
   * saving without one would make write.rs's stale-read guard a no-op. */
  function disableSave(reason) {
    if (!els.saveBtn) return;
    els.saveBtn.disabled = true;
    els.saveBtn.title = reason || "";
  }

  function enableSave() {
    if (!els.saveBtn) return;
    els.saveBtn.disabled = false;
    els.saveBtn.title = "";
  }

  function closeEditor() {
    els.overlay.hidden = true;
    mode = null;
    originalPath = null;
    originalHash = null;
    clearErrors();
    setStatus("", null);
    enableSave();
  }

  function updateModeBadge() {
    if (!els.modeBadge) return;
    els.modeBadge.textContent = mode === "create" ? "NEW" : "EDIT";
    // Theme-aware via CSS (--accent-success / --accent tokens in viz.css)
    // rather than an inline hex background, so the badge follows
    // prefers-color-scheme instead of being pinned to the light palette.
    els.modeBadge.classList.toggle("mode-create", mode === "create");
    els.modeBadge.classList.toggle("mode-edit", mode !== "create");
  }

  async function onPathBlur() {
    if (mode !== "create") return;
    const path = els.pathInput.value.trim();
    if (!path) return;
    await fetchSchema(path);
  }

  // -------------------------------------------------------------------
  // Schema hints panel + new-document template
  // -------------------------------------------------------------------

  async function fetchSchema(path) {
    els.schemaContent.textContent = "Loading…";
    try {
      const res = await fetch(`api/schema/${window.KBViz.encodePathForApi(path)}`);
      if (!res.ok) {
        els.schemaContent.textContent = "Schema unavailable.";
        return null;
      }
      const schema = await res.json();
      renderSchema(schema);
      return schema;
    } catch (err) {
      els.schemaContent.textContent = "Schema unavailable.";
      return null;
    }
  }

  function renderSchema(schema) {
    els.schemaContent.innerHTML = "";

    if (schema && schema.frozen) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent =
        "This location's schema failed to parse — writes here will fail until it's fixed.";
      els.schemaContent.appendChild(p);
    }

    const fields = (schema && schema.fields) || [];
    if (!fields.length) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "No declared fields for this location.";
      els.schemaContent.appendChild(p);
      return;
    }

    for (const f of fields) {
      const field = document.createElement("div");
      field.className = "schema-field";

      const name = document.createElement("div");
      name.className = "schema-field-name";
      name.textContent = f.field;
      if (f.required) {
        const req = document.createElement("span");
        req.className = "req";
        req.textContent = "*";
        req.title = "required";
        name.appendChild(req);
      }
      const type = document.createElement("span");
      type.className = "schema-field-type";
      type.textContent = f.type || "any";
      name.appendChild(type);
      field.appendChild(name);

      if (f.values && f.values.length) {
        const values = document.createElement("div");
        values.className = "schema-field-values";
        for (const v of f.values) {
          const span = document.createElement("span");
          span.textContent = v;
          values.appendChild(span);
        }
        field.appendChild(values);
      }

      if (f.declared_in) {
        const origin = document.createElement("div");
        origin.className = "muted";
        origin.textContent = `via ${f.declared_in}`;
        field.appendChild(origin);
      }

      els.schemaContent.appendChild(field);
    }
  }

  /** Build a starting frontmatter skeleton for a new document: required
   * fields get a default value (schema `default`, else the first allowed
   * value, else a type-appropriate placeholder). Optional fields are left
   * out — the schema hints panel documents them for the user to add.
   *
   * Fields are declared by dot-path (e.g. "planning.prep_minutes") for
   * nested schema attributes. Server-side `schema::get_by_dotpath` splits
   * on '.' and descends into nested mappings, so the skeleton must emit
   * real nested YAML (`planning:\n  prep_minutes: 0`) rather than a
   * literal dotted key — siblings under the same parent are merged into
   * one parent mapping, and this nests to arbitrary depth. Fields whose
   * own type is "object" are containers only (their children, if any
   * required, supply the nesting) and never get a value line themselves. */
  function buildTemplate(schema, path) {
    const fields = (schema && schema.fields) || [];
    const tree = {};
    for (const f of fields) {
      if (!f.required || f.type === "object") continue;
      insertTemplateValue(tree, f.field.split("."), placeholderFor(f));
    }
    const lines = ["---", ...renderTemplateTree(tree, 0)];
    lines.push("---", "", `# ${titleFromPath(path)}`, "", "");
    return lines.join("\n");
  }

  /** Insert `value` into `tree` at the nested location named by
   * `segments`, creating intermediate mapping objects as needed. */
  function insertTemplateValue(tree, segments, value) {
    let node = tree;
    for (let i = 0; i < segments.length - 1; i++) {
      const seg = segments[i];
      if (typeof node[seg] !== "object" || node[seg] === null) {
        node[seg] = {};
      }
      node = node[seg];
    }
    node[segments[segments.length - 1]] = value;
  }

  /** Render a template tree (built by insertTemplateValue) to YAML lines,
   * indenting two spaces per level of nesting. */
  function renderTemplateTree(node, depth) {
    const indent = "  ".repeat(depth);
    const lines = [];
    for (const key of Object.keys(node)) {
      const val = node[key];
      if (val !== null && typeof val === "object") {
        lines.push(`${indent}${key}:`);
        lines.push(...renderTemplateTree(val, depth + 1));
      } else {
        lines.push(`${indent}${key}: ${val}`);
      }
    }
    return lines;
  }

  function placeholderFor(f) {
    if (f.default !== null && f.default !== undefined) {
      return typeof f.default === "string" ? f.default : JSON.stringify(f.default);
    }
    if (f.values && f.values.length) return f.values[0];
    switch (f.type) {
      case "integer":
      case "number":
        return "0";
      case "boolean":
        return "false";
      case "list":
        return "[]";
      case "date":
        return new Date().toISOString().slice(0, 10);
      case "timestamp":
        return new Date().toISOString();
      default:
        return "";
    }
  }

  function titleFromPath(path) {
    const base = path.split("/").pop().replace(/\.md$/i, "");
    return base.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // -------------------------------------------------------------------
  // Preview
  // -------------------------------------------------------------------

  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(renderPreview, PREVIEW_DEBOUNCE_MS);
  }

  function renderPreview() {
    const body = window.KBViz.stripFrontmatter(els.textarea.value || "");
    try {
      // Route through KBViz.renderMarkdown rather than calling
      // window.marked.parse directly: marked does not sanitize its own
      // output, and the preview pane assigns straight to innerHTML, so an
      // unsanitized render here would let a document body execute
      // arbitrary script in the editor the same way viz.js's detail panel
      // could (see viz.js's header comment and its renderMarkdown helper,
      // which wraps marked.parse in DOMPurify.sanitize).
      els.preview.innerHTML = window.KBViz.renderMarkdown(body);
    } catch (err) {
      els.preview.textContent = "(preview error)";
    }
  }

  // -------------------------------------------------------------------
  // Save (create or edit)
  // -------------------------------------------------------------------

  async function save() {
    clearErrors();

    // Belt-and-suspenders: the Save button is disabled while an edit-mode
    // load is pending/failed, but Cmd/Ctrl+S calls save() directly and
    // bypasses that. Without originalHash, a save would omit expected_hash
    // and clobber concurrent edits — refuse rather than proceed unguarded.
    if (mode === "edit" && !originalHash) {
      showError(
        "Cannot save: the document hasn't finished loading (or failed to load). Close and reopen the editor."
      );
      return;
    }

    const path = mode === "create" ? els.pathInput.value.trim() : originalPath;
    if (!path) {
      showError("Path is required.");
      return;
    }
    if (!path.toLowerCase().endsWith(".md")) {
      showError("Path must end in .md");
      return;
    }

    const verb = mode === "create" ? "add" : "update";
    const commitMessage = window.prompt("Commit message:", `docs: ${verb} ${path}`);
    if (commitMessage === null) return; // user cancelled
    if (!commitMessage.trim()) {
      showError("Commit message is required.");
      return;
    }

    const body = {
      content: els.textarea.value,
      commit_message: commitMessage,
      create: mode === "create",
    };
    if (mode === "edit") body.expected_hash = originalHash;

    setStatus("Saving…", null);
    els.saveBtn.disabled = true;
    try {
      const res = await fetch(`api/doc/${window.KBViz.encodePathForApi(path)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await safeJson(res);

      if (res.ok) {
        await onSaveSuccess(path, data);
      } else if (res.status === 422) {
        renderFieldErrors(data);
      } else if (res.status === 409) {
        renderConflict(data, path);
      } else {
        renderGenericError(res.status, data);
      }
    } catch (err) {
      setStatus(`Save failed: ${err.message}`, "err");
    } finally {
      els.saveBtn.disabled = false;
    }
  }

  async function onSaveSuccess(path, data) {
    const sha = data && data.sha ? String(data.sha).slice(0, 8) : null;
    const pendingSync = data && data.outcome === "committed_pending_sync";
    setStatus(
      pendingSync
        ? `Committed locally${sha ? ` (${sha})` : ""} — push pending`
        : `Saved${sha ? ` (${sha})` : ""}`,
      "ok"
    );

    // Re-fetch to pick up the fresh content_hash for any further save in
    // this session, and to reflect any server-side normalization.
    try {
      const res = await fetch(`api/doc/${window.KBViz.encodePathForApi(path)}`);
      if (res.ok) {
        const doc = await res.json();
        originalHash = doc.content_hash;
        els.textarea.value = doc.content || els.textarea.value;
      }
    } catch (err) {
      // Non-fatal: the save itself already succeeded.
    }

    mode = "edit";
    originalPath = path;
    updateModeBadge();
    els.pathInput.setAttribute("readonly", "readonly");
    renderPreview();

    await window.KBViz.refreshGraph();
    window.KBViz.showDetail(path);

    setTimeout(closeEditor, 700);
  }

  // -------------------------------------------------------------------
  // Delete
  // -------------------------------------------------------------------

  function confirmDeleteCurrent() {
    const id = window.KBViz.getCurrentId();
    if (!id) return;
    showConfirm(`Delete "${id}"? This cannot be undone from the UI.`, () =>
      doDelete(id)
    );
  }

  async function doDelete(path) {
    const commitMessage = window.prompt("Commit message:", `docs: delete ${path}`);
    if (commitMessage === null) return;
    if (!commitMessage.trim()) {
      window.alert("Commit message is required.");
      return;
    }
    try {
      const res = await fetch(`api/doc/${window.KBViz.encodePathForApi(path)}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ commit_message: commitMessage }),
      });
      const data = await safeJson(res);
      if (res.ok) {
        if (mode === "edit" && originalPath === path) closeEditor();
        window.KBViz.removeNodeById(path);
        await window.KBViz.refreshGraph();
      } else {
        window.alert(
          `Delete failed (HTTP ${res.status}): ${
            errorMessageFrom(data) || "no further detail from the server."
          }`
        );
      }
    } catch (err) {
      window.alert(`Delete failed: ${err.message}`);
    }
  }

  // -------------------------------------------------------------------
  // Confirm dialog
  // -------------------------------------------------------------------

  function showConfirm(message, onConfirm) {
    els.confirmMessage.textContent = message;
    confirmCallback = onConfirm;
    els.confirmOverlay.hidden = false;
  }

  function hideConfirm() {
    els.confirmOverlay.hidden = true;
    confirmCallback = null;
  }

  // -------------------------------------------------------------------
  // Error / status rendering
  // -------------------------------------------------------------------

  function clearErrors() {
    els.errors.innerHTML = "";
    els.errors.hidden = true;
  }

  function showError(msg) {
    els.errors.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = msg;
    els.errors.appendChild(p);
    els.errors.hidden = false;
  }

  /** 422: render the structured per-field validation errors the write
   * pipeline returns (see write.rs's `FieldError` / `ValidationResult`). */
  function renderFieldErrors(data) {
    els.errors.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = "Validation failed:";
    els.errors.appendChild(p);

    const list = (data && data.field_errors) || [];
    if (!list.length) {
      const msg = document.createElement("p");
      msg.textContent = errorMessageFrom(data) || "Unknown validation error.";
      els.errors.appendChild(msg);
    } else {
      const ul = document.createElement("ul");
      for (const fe of list) {
        const li = document.createElement("li");
        li.textContent = `${fe.field} (${fe.rule}): ${fe.message}`;
        if (fe.got !== null && fe.got !== undefined) {
          li.textContent += ` — got: ${fe.got}`;
        }
        if (fe.expected && fe.expected.length) {
          const small = document.createElement("div");
          small.className = "muted";
          small.textContent = `allowed: ${fe.expected.join(", ")}`;
          li.appendChild(small);
        }
        ul.appendChild(li);
      }
      els.errors.appendChild(ul);
    }
    els.errors.hidden = false;
    setStatus("Validation failed", "err");
  }

  /** 409: either a stale `expected_hash` (someone else edited this file
   * since it was loaded) or a create/edit mismatch (create-on-existing,
   * edit-on-missing). Either way, offer to reload the current server
   * state into the editor rather than guessing which case it was. */
  function renderConflict(data, path) {
    els.errors.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = `Conflict (HTTP 409): ${
      errorMessageFrom(data) ||
      "the document changed on the server, or already exists/doesn't exist."
    }`;
    els.errors.appendChild(p);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Reload latest version";
    btn.addEventListener("click", () => openEditor("edit", path));
    els.errors.appendChild(btn);

    els.errors.hidden = false;
    setStatus("Save conflict", "err");
  }

  function renderGenericError(status, data) {
    els.errors.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = `Save failed (HTTP ${status}): ${
      errorMessageFrom(data) || "no further detail from the server."
    }`;
    els.errors.appendChild(p);
    els.errors.hidden = false;
    setStatus("Save failed", "err");
  }

  function setStatus(text, kind) {
    if (!els.status) return;
    els.status.textContent = text;
    els.status.classList.remove("status-ok", "status-err");
    if (kind === "ok") els.status.classList.add("status-ok");
    if (kind === "err") els.status.classList.add("status-err");
  }

  function errorMessageFrom(data) {
    if (!data) return null;
    if (typeof data === "string") return data;
    return data.error || data.message || data.msg || data.reason || null;
  }

  async function safeJson(res) {
    try {
      return await res.json();
    } catch (err) {
      return null;
    }
  }

  init();
})();
