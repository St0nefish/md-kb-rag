/*
 * Adapted from Google's Open Knowledge Format (OKF) reference viewer
 * (reference_agent/bundle/static/viz.js), licensed under the Apache
 * License, Version 2.0. See LICENSE-okf.md in this directory for the
 * full license text and a summary of what changed.
 *
 * Changes from the original:
 *   - The graph bundle is no longer inlined at generation time
 *     (window.BUNDLE/window.BUNDLE_NAME). It is fetched live from
 *     `api/graph` on load, so the whole body runs inside an async
 *     init() instead of a plain IIFE.
 *   - Document bodies are no longer inlined either (there is no
 *     `bundle.bodies` map). `showDetail` now lazily fetches
 *     `api/doc/<path>`, strips the leading frontmatter block, and
 *     renders the remaining markdown body. Node metadata (title,
 *     type, tags, description, status, domain) still comes from the
 *     graph payload, matching this project's node schema rather than
 *     OKF's concept schema (no resource/generated/verified/sources/
 *     trust_tier/stale fields here).
 *   - rewriteInternalLinks now resolves relative markdown links
 *     against the *current* document's path itself (OKF's bundle
 *     pre-resolved links to root-relative ids at generation time;
 *     here that resolution happens client-side against live node
 *     ids, which include the .md extension).
 *   - Search keeps the original instant substring dim, and adds a
 *     debounced semantic search against `api/search` that highlights
 *     hits with a `.search-hit` class and fits the viewport to them.
 *   - Edges carry a `kind` (`markdown` | `semantic`) and semantic
 *     edges carry a `score`; styling distinguishes the two.
 *   - All fetch URLs are relative (`api/...`) so the app keeps working
 *     behind a path-prefixed reverse proxy.
 *   - A small integration surface is exposed on `window.KBViz` (not part
 *     of the OKF original) so assets/edit.js — the in-browser editor,
 *     loaded as a separate script — can read the current graph state and
 *     push updates back into it after a save/create/delete, without
 *     either file reaching into the other's internals directly. See the
 *     `window.KBViz = {...}` assignment at the bottom of this file.
 *   - marked does not sanitize its output, and document bodies are
 *     arbitrary user-authored markdown, so `renderMarkdown` (vendored
 *     DOMPurify, loaded before this script — see index.html) wraps
 *     every marked.parse() call before the result is assigned to
 *     innerHTML. Not part of the OKF original, which never rendered
 *     untrusted markdown to live HTML at view time.
 *   - The Cytoscape canvas is drawn from a JS style array, which can't
 *     read the CSS custom properties viz.css's light/dark palette is
 *     built on. `themeColors()`/`buildStyle()` supply the handful of
 *     canvas-relevant colors (node label/border, default edge color) for
 *     whichever mode `prefers-color-scheme` currently reports, and a
 *     `change` listener on that media query re-applies the style to the
 *     live `cy` instance so the graph repaints if the OS theme flips
 *     while the page is open. Not part of the OKF original, which had no
 *     dark mode.
 */
(function () {
  "use strict";

  const SEARCH_DEBOUNCE_MS = 300;

  const darkMq = window.matchMedia("(prefers-color-scheme: dark)");

  /** Canvas-relevant colors Cytoscape's style JSON can't pull from CSS
   * custom properties. Mirrors viz.css's --text/--border/--border-strong
   * tokens; the accent colors (selection highlight, search-hit ring) stay
   * identical in both modes, so they're inlined directly in buildStyle()
   * rather than threaded through here. */
  function themeColors() {
    return darkMq.matches
      ? { nodeLabel: "#e2e8f0", nodeBorder: "#e2e8f0", edgeLine: "#475569" }
      : { nodeLabel: "#0f172a", nodeBorder: "#0f172a", edgeLine: "#cbd5e1" };
  }

  function buildStyle() {
    const t = themeColors();
    return [
      {
        selector: "node",
        style: {
          "background-color": "data(color)",
          "label": "data(label)",
          "color": t.nodeLabel,
          "font-size": 11,
          "text-valign": "bottom",
          "text-margin-y": 4,
          "text-wrap": "wrap",
          "text-max-width": 120,
          "width": "data(size)",
          "height": "data(size)",
          "border-width": 1,
          "border-color": t.nodeBorder,
        },
      },
      {
        selector: 'node[status = "archived"]',
        style: {
          "opacity": 0.55,
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 3,
          "border-color": "#f59e0b",
        },
      },
      {
        selector: "edge",
        style: {
          "width": 1.5,
          "line-color": t.edgeLine,
          "target-arrow-color": t.edgeLine,
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 0.9,
          "line-style": "solid",
          "opacity": 1,
        },
      },
      {
        selector: 'edge[kind = "semantic"]',
        style: {
          "line-style": "dashed",
          "target-arrow-shape": "none",
          "opacity": "mapData(score, 0, 1, 0.15, 0.65)",
        },
      },
      {
        selector: "edge:selected",
        style: {
          "line-color": "#f59e0b",
          "target-arrow-color": "#f59e0b",
          "width": 2.5,
        },
      },
      {
        selector: ".dim",
        style: { "opacity": 0.15 },
      },
      {
        selector: ".search-hit",
        style: {
          "border-width": 4,
          "border-color": "#0ea5e9",
          "border-style": "solid",
          "z-index": 999,
        },
      },
    ];
  }

  let cy = null;
  let nodeIndex = {};
  let backlinks = {};
  let searchDebounceTimer = null;
  let searchToken = 0;
  let currentDetailId = null;

  async function init() {
    const graphEl = document.getElementById("graph");
    const loadingEl = document.getElementById("graph-loading");
    const errorEl = document.getElementById("graph-error");

    let bundle;
    try {
      const res = await fetch("api/graph");
      if (!res.ok) {
        throw new Error(`api/graph returned ${res.status}`);
      }
      bundle = await res.json();
    } catch (err) {
      if (loadingEl) loadingEl.hidden = true;
      if (errorEl) {
        errorEl.hidden = false;
        errorEl.textContent = `Failed to load graph: ${err.message}`;
      }
      return;
    }
    if (loadingEl) loadingEl.hidden = true;

    document.title = "Knowledge Base Viewer";

    populateTypeFilter(bundle.types || []);
    rebuildBacklinks(bundle.edges || []);

    // Look up node label/type by id
    nodeIndex = {};
    for (const n of bundle.nodes || []) nodeIndex[n.data.id] = n.data;

    cy = cytoscape({
      container: graphEl,
      elements: [...(bundle.nodes || []), ...(bundle.edges || [])],
      style: buildStyle(),
      layout: { name: "cose", animate: false, padding: 30 },
      wheelSensitivity: 0.2,
    });

    // Repaint the canvas (label/border/edge colors) if the OS/browser
    // theme flips while the page is open — CSS handles the rest of the
    // page via var(--token) cascades, but Cytoscape's style is a JS
    // object snapshot that has to be explicitly rebuilt and reapplied.
    darkMq.addEventListener("change", () => {
      if (cy) cy.style(buildStyle());
    });

    cy.on("tap", "node", (evt) => {
      showDetail(evt.target.id());
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) clearSelection();
    });

    document.getElementById("layout").addEventListener("change", (e) => {
      cy.layout({ name: e.target.value, animate: false, padding: 30 }).run();
    });

    document.getElementById("reset").addEventListener("click", () => {
      cy.fit(null, 30);
      clearSelection();
    });

    document.getElementById("search").addEventListener("input", (e) => {
      const raw = e.target.value;
      const q = raw.trim().toLowerCase();

      // Instant substring dim — unchanged from the OKF original.
      if (!q) {
        cy.elements().removeClass("dim");
      } else {
        cy.nodes().forEach((n) => {
          const d = n.data();
          const hay =
            (d.label || "").toLowerCase() + " " +
            d.id.toLowerCase() + " " +
            (d.tags || []).join(" ").toLowerCase();
          n.toggleClass("dim", !hay.includes(q));
        });
        cy.edges().forEach((edge) => {
          const src = edge.source();
          const tgt = edge.target();
          edge.toggleClass("dim", src.hasClass("dim") || tgt.hasClass("dim"));
        });
      }

      // Debounced semantic search — populates the results panel and
      // highlights hits distinctly from dim.
      clearTimeout(searchDebounceTimer);
      if (!q) {
        cy.elements().removeClass("search-hit");
        searchToken++; // invalidate any in-flight request
        hideSearchResults();
        return;
      }
      searchDebounceTimer = setTimeout(() => {
        runSemanticSearch(raw.trim());
      }, SEARCH_DEBOUNCE_MS);
    });

    document.getElementById("search").addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        hideSearchResults();
      } else if (e.key === "Enter") {
        // Open the top hit, if the results panel is showing any.
        const first = document.querySelector("#search-results .search-result");
        if (first) first.click();
      }
    });

    // Click anywhere outside the search box/panel dismisses the results.
    document.addEventListener("pointerdown", (e) => {
      if (!e.target.closest(".search-wrap")) hideSearchResults();
    });

    document.getElementById("filter-type").addEventListener("change", (e) => {
      const t = e.target.value;
      if (!t) {
        cy.elements().removeClass("dim");
        return;
      }
      cy.nodes().forEach((n) => {
        n.toggleClass("dim", n.data("type") !== t);
      });
      cy.edges().forEach((edge) => {
        edge.toggleClass("dim", edge.source().hasClass("dim") || edge.target().hasClass("dim"));
      });
    });

    // Auto-show the first node.
    const initial = bundle.nodes && bundle.nodes[0];
    if (initial) showDetail(initial.data.id);
  }

  async function runSemanticSearch(q) {
    const token = ++searchToken;
    let data;
    try {
      const res = await fetch(`api/search?q=${encodeURIComponent(q)}`);
      if (!res.ok) {
        if (token === searchToken) renderSearchResults(null);
        return;
      }
      data = await res.json();
    } catch (err) {
      if (token === searchToken) renderSearchResults(null);
      return;
    }
    if (token !== searchToken) return; // a newer query has since been issued

    cy.elements().removeClass("search-hit");
    const hitIds = new Set((data.results || []).map((r) => r.file_path));
    const hits = cy.nodes().filter((n) => hitIds.has(n.id()));
    hits.addClass("search-hit");
    if (hits.length) {
      cy.animate({ fit: { eles: hits, padding: 40 } }, { duration: 200 });
    }

    renderSearchResults(data.results || []);
  }

  /** Render semantic search hits as a clickable results list under the
   * search box. `api/search` returns chunk-level hits ordered by score, so
   * a document can appear more than once — keep only its best (first)
   * chunk. `results` semantics: null = the search request itself failed,
   * [] = it succeeded with no hits. */
  function renderSearchResults(results) {
    const panel = document.getElementById("search-results");
    panel.innerHTML = "";
    panel.hidden = false;

    if (results === null) {
      const p = document.createElement("div");
      p.className = "search-results-empty";
      p.textContent = "Search unavailable.";
      panel.appendChild(p);
      return;
    }

    const seen = new Set();
    const docs = results.filter((r) => {
      if (seen.has(r.file_path)) return false;
      seen.add(r.file_path);
      return true;
    });

    if (!docs.length) {
      const p = document.createElement("div");
      p.className = "search-results-empty";
      p.textContent = "No results.";
      panel.appendChild(p);
      return;
    }

    for (const r of docs) {
      const item = document.createElement("div");
      item.className = "search-result";
      item.tabIndex = 0;

      const head = document.createElement("div");
      head.className = "search-result-head";
      const title = document.createElement("span");
      title.className = "search-result-title";
      title.textContent = r.title || r.file_path;
      const score = document.createElement("span");
      score.className = "search-result-score";
      score.textContent = r.score.toFixed(2);
      head.appendChild(title);
      head.appendChild(score);
      item.appendChild(head);

      const path = document.createElement("div");
      path.className = "search-result-path";
      path.textContent = r.file_path;
      item.appendChild(path);

      const snippet = (r.text || "").replace(/\s+/g, " ").trim();
      if (snippet) {
        const snip = document.createElement("div");
        snip.className = "search-result-snippet";
        snip.textContent = snippet.length > 160 ? snippet.slice(0, 160) + "…" : snippet;
        item.appendChild(snip);
      }

      const open = () => {
        hideSearchResults();
        showDetail(r.file_path);
      };
      item.addEventListener("click", open);
      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter") open();
      });
      panel.appendChild(item);
    }
  }

  function hideSearchResults() {
    const panel = document.getElementById("search-results");
    panel.hidden = true;
    panel.innerHTML = "";
  }

  function clearSelection() {
    cy.elements().unselect();
    currentDetailId = null;
    document.getElementById("detail-empty").hidden = false;
    document.getElementById("detail-content").hidden = true;
  }

  /** Populate #filter-type with any type strings not already present as an
   * <option>, preserving the current selection. Used both at initial load
   * and after a graph refresh (a save can introduce a brand-new type). */
  function populateTypeFilter(types) {
    const typeSelect = document.getElementById("filter-type");
    const existing = new Set(
      Array.from(typeSelect.options).map((o) => o.value)
    );
    for (const t of types) {
      if (existing.has(t)) continue;
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      typeSelect.appendChild(opt);
    }
  }

  /** Build the reverse-link index for backlinks (markdown edges only —
   * semantic edges are "related", not "cited by"). */
  function rebuildBacklinks(edges) {
    backlinks = {};
    for (const edge of edges) {
      const { source, target, kind } = edge.data;
      if (kind && kind !== "markdown") continue;
      (backlinks[target] ||= []).push(source);
    }
  }

  /** Merge a freshly-fetched `api/graph` bundle into the live cytoscape
   * instance: update data on nodes that already exist (label/type/tags/etc.
   * may have changed), add brand-new nodes, drop nodes no longer present
   * (e.g. after a delete elsewhere), and replace the edge set wholesale
   * (cheap, and edges carry no client-only state). Existing node positions
   * are preserved — a layout only reruns when a node was actually added, so
   * a plain content edit doesn't reshuffle the graph the user is looking
   * at. */
  function applyGraphUpdate(bundle) {
    const newIndex = {};
    for (const n of bundle.nodes || []) newIndex[n.data.id] = n.data;

    let added = false;
    for (const id of Object.keys(newIndex)) {
      const existing = cy.getElementById(id);
      if (existing && existing.length) {
        existing.data(newIndex[id]);
      } else {
        cy.add({ group: "nodes", data: newIndex[id] });
        added = true;
      }
    }
    for (const id of Object.keys(nodeIndex)) {
      if (!(id in newIndex)) {
        cy.remove(cy.getElementById(id));
        if (currentDetailId === id) clearSelection();
      }
    }
    nodeIndex = newIndex;

    cy.edges().remove();
    cy.add((bundle.edges || []).map((e) => ({ group: "edge", data: e.data })));
    rebuildBacklinks(bundle.edges || []);
    populateTypeFilter(bundle.types || []);

    if (added) {
      cy.layout({ name: "cose", animate: false, padding: 30 }).run();
    }
  }

  /** Re-fetch `api/graph` and merge it into the live graph. Returns the raw
   * bundle (so a caller can also read the freshly-updated node data), or
   * `null` if the fetch failed. */
  async function refreshGraph() {
    try {
      const res = await fetch("api/graph");
      if (!res.ok) return null;
      const bundle = await res.json();
      applyGraphUpdate(bundle);
      return bundle;
    } catch (err) {
      return null;
    }
  }

  /** Remove a single node (and its edges) from the live graph, e.g. after a
   * successful delete. Clears the detail panel first if it was showing
   * that node. */
  function removeNodeById(id) {
    if (currentDetailId === id) clearSelection();
    const el = cy.getElementById(id);
    if (el && el.length) cy.remove(el);
    delete nodeIndex[id];
    delete backlinks[id];
    for (const key of Object.keys(backlinks)) {
      backlinks[key] = backlinks[key].filter((src) => src !== id);
    }
  }

  /** Repo-relative doc/schema path segments must be individually
   * percent-encoded (not the whole path) so literal "/" keeps
   * separating path segments for the axum `{*path}` wildcard route. */
  function encodePathForApi(path) {
    return path.split("/").map(encodeURIComponent).join("/");
  }

  /** Strip a single leading `---\n...\n---` YAML frontmatter block, if
   * present, from raw markdown file content. The closing fence line may
   * carry trailing spaces/tabs (and a CR before its line ending, or no
   * line ending at all if the document ends there). */
  function stripFrontmatter(content) {
    if (!content.startsWith("---")) return content;
    const rest = content.slice(3);
    const m = /\n---[ \t]*(\r\n|\r|\n|$)/.exec(rest);
    if (!m) return content;
    return rest.slice(m.index + m[0].length);
  }

  /** Render markdown to sanitized HTML: marked.parse() followed by
   * DOMPurify.sanitize(). marked does not sanitize its output, so every
   * call site that assigns parsed markdown to innerHTML must go through
   * this helper (or DOMPurify directly) rather than calling marked.parse
   * alone. Exposed on window.KBViz so assets/edit.js's preview pane uses
   * the same sanitization path. */
  function renderMarkdown(md) {
    const html = marked.parse(md || "", { breaks: false, gfm: true });
    return DOMPurify.sanitize(html);
  }

  async function showDetail(id) {
    const data = nodeIndex[id];
    if (!data) return;
    currentDetailId = id;
    cy.elements().unselect();
    const node = cy.getElementById(id);
    if (node) node.select();

    document.getElementById("detail-empty").hidden = true;
    const content = document.getElementById("detail-content");
    content.hidden = false;

    const chip = document.getElementById("detail-type");
    chip.textContent = data.type || "";
    chip.style.background = data.color || "";

    document.getElementById("detail-title").textContent = data.label || id;
    document.getElementById("detail-id").textContent = id;
    document.getElementById("detail-description").textContent = data.description || "—";
    document.getElementById("detail-domain").textContent = data.domain || "—";

    const tagsEl = document.getElementById("detail-tags");
    tagsEl.innerHTML = "";
    if (data.tags && data.tags.length) {
      for (const t of data.tags) {
        const span = document.createElement("span");
        span.className = "tag";
        span.textContent = t;
        tagsEl.appendChild(span);
      }
    } else {
      tagsEl.textContent = "—";
    }

    document.getElementById("detail-mtime").textContent = formatMtime(data.mtime);

    const badgesEl = document.getElementById("detail-badges");
    badgesEl.innerHTML = "";
    if (data.status) {
      badgesEl.appendChild(makeBadge(data.status, "status-" + data.status));
    }

    const bl = backlinks[id] || [];
    const blSection = document.getElementById("detail-backlinks");
    const blList = document.getElementById("backlinks-list");
    blList.innerHTML = "";
    if (bl.length) {
      blSection.hidden = false;
      for (const src of bl) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.textContent = nodeIndex[src]?.label || src;
        a.dataset.target = src;
        a.addEventListener("click", () => showDetail(src));
        li.appendChild(a);
        const muted = document.createElement("span");
        muted.className = "muted";
        muted.textContent = ` (${src})`;
        li.appendChild(muted);
        blList.appendChild(li);
      }
    } else {
      blSection.hidden = true;
    }

    cy.animate({ center: { eles: node }, zoom: Math.max(cy.zoom(), 1.0) }, { duration: 200 });

    // Lazy body fetch — independent of the synchronous metadata above.
    const bodyEl = document.getElementById("detail-body");
    bodyEl.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const res = await fetch(`api/doc/${encodePathForApi(id)}`);
      if (!res.ok) {
        bodyEl.innerHTML = `<p class="muted">Could not load document (HTTP ${res.status}).</p>`;
        return;
      }
      const doc = await res.json();
      const body = stripFrontmatter(doc.content || "");
      bodyEl.innerHTML = renderMarkdown(body);
      rewriteInternalLinks(bodyEl, id);
    } catch (err) {
      bodyEl.innerHTML = '<p class="muted">Failed to load document.</p>';
    }
  }

  function makeBadge(text, cls) {
    const span = document.createElement("span");
    span.className = "badge " + cls;
    span.textContent = text;
    return span;
  }

  function formatMtime(mtime) {
    if (!mtime && mtime !== 0) return "—";
    const d = new Date(mtime * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toISOString().slice(0, 10);
  }

  /** Resolve a markdown link href against the directory of `sourceId`,
   * mirroring the relative-path resolution ingest.rs applies when it
   * extracts markdown-link edges at index time (./ and ../, fragment
   * stripping, absolute/external skip). Returns the resolved
   * repo-relative id, or null if the href isn't a resolvable internal
   * .md link. */
  function resolveRelativeLink(sourceId, href) {
    const hashIdx = href.indexOf("#");
    const clean = hashIdx >= 0 ? href.slice(0, hashIdx) : href;
    if (!clean) return null;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(clean)) return null; // http:, https:, mailto:, etc.
    if (clean.startsWith("//")) return null; // protocol-relative
    if (clean.startsWith("/")) return null; // absolute path — can't resolve to a repo-relative id here
    if (!clean.toLowerCase().endsWith(".md")) return null;

    const baseParts = sourceId.split("/");
    baseParts.pop(); // drop the filename, keep the containing directory
    for (const part of clean.split("/")) {
      if (part === "" || part === ".") continue;
      if (part === "..") {
        if (baseParts.length === 0) return null;
        baseParts.pop();
      } else {
        baseParts.push(part);
      }
    }
    return baseParts.join("/");
  }

  function rewriteInternalLinks(root, sourceId) {
    root.querySelectorAll("a[href]").forEach((a) => {
      const href = a.getAttribute("href");
      if (!href) return;
      const target = resolveRelativeLink(sourceId, href);
      if (target && nodeIndex[target]) {
        a.className = "internal";
        a.setAttribute("href", "javascript:void(0)");
        a.addEventListener("click", (e) => {
          e.preventDefault();
          showDetail(target);
        });
        return;
      }
      a.className = "external";
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener");
    });
  }

  /** Integration surface for assets/edit.js (a separate script — see this
   * file's header comment). Deliberately small: edit.js drives its own DOM
   * (the editor overlay) and only needs to read the current graph/detail
   * state and push updates back into the live cytoscape instance. */
  window.KBViz = {
    getNode: (id) => nodeIndex[id],
    getNodeIds: () => Object.keys(nodeIndex),
    getCurrentId: () => currentDetailId,
    showDetail,
    refreshGraph,
    removeNodeById,
    encodePathForApi,
    stripFrontmatter,
    renderMarkdown,
  };

  init();
})();
