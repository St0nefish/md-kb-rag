/*
 * assets/viz.js — the knowledge-base web app: data layer, hash router,
 * home/browse/doc views, sidebar document tree, and the (now secondary)
 * Cytoscape graph views.
 *
 * Provenance: the graph canvas — Cytoscape styling, the dim/highlight
 * search classes, the detail rendering that is now the doc view, and the
 * markdown link rewriting — is adapted from Google's Open Knowledge
 * Format (OKF) reference viewer (reference_agent/bundle/static/viz.js),
 * licensed under the Apache License, Version 2.0. See LICENSE-okf.md in
 * this directory. The docs-first shell around it (router, home/browse
 * views, sidebar tree, graph toolbar, lazy graph construction, semantic
 * search results panel) is md-kb-rag's own.
 *
 * Architecture notes:
 *   - `api/graph` is fetched once at startup and is the single data
 *     source for every non-body surface: the sidebar tree, home page,
 *     browse listings, doc metadata, and the graph itself. Document
 *     bodies are fetched lazily per doc from `api/doc/<path>`.
 *   - Views are hash-routed: #/ (home), #/browse/<dir>, #/doc/<path>,
 *     #/graph (full KB), #/graph/<path> (that doc's neighborhood).
 *     Path segments are individually percent-encoded, mirroring
 *     encodePathForApi.
 *   - The Cytoscape instance is created lazily on first entry to a
 *     graph view — the common read-a-doc path never pays for a force
 *     layout. Once created it persists across view switches; the local
 *     (neighborhood) mode is expressed as an `.excluded` display:none
 *     class rather than element removal, so switching between local and
 *     global is cheap and non-destructive.
 *   - Graph labels are level-of-detail: a `.no-label` class is toggled
 *     by visible node count (LABEL_MAX_NODES — zoom can't help, see the
 *     comment there), so a doc's neighbourhood is labelled while the
 *     full KB is dots; a `.hover-label` class (mouseover) force-shows
 *     the hovered node's label with a themed backing, so the old
 *     every-label-overlapping-every-other mess can't come back.
 *   - fcose (vendored, with its layout-base/cose-base dependencies —
 *     see index.html's script order) is the default layout because it
 *     packs the KB's many disconnected components far better than the
 *     built-in cose; runGraphLayout falls back to cose if the plugin
 *     didn't register.
 *   - The search box does two things: an instant substring dim on the
 *     graph (when it exists), and a debounced semantic query against
 *     `api/search` rendered as a results panel under the input (per-doc
 *     best chunk, click to open). Graph hit-highlighting still happens,
 *     but only matters when the graph view is active.
 *   - marked does not sanitize its output, and document bodies are
 *     arbitrary user-authored markdown, so `renderMarkdown` (vendored
 *     DOMPurify, loaded before this script) wraps every marked.parse()
 *     call before the result is assigned to innerHTML.
 *   - `window.KBViz` is the integration surface assets/edit.js (the
 *     in-browser editor, a separate script) drives: it is unchanged from
 *     the graph-era UI — showDetail(path) now navigates to the doc view,
 *     refreshGraph() re-fetches api/graph and re-renders whatever is on
 *     screen, removeNodeById(path) drops a deleted doc everywhere.
 *   - The Cytoscape canvas is drawn from a JS style array, which can't
 *     read CSS custom properties; themeColors()/buildStyle() supply the
 *     canvas-relevant colors for whichever mode prefers-color-scheme
 *     reports, and a `change` listener re-applies the style if the OS
 *     theme flips while the page is open.
 */
(function () {
  "use strict";

  const SEARCH_DEBOUNCE_MS = 300;
  const RECENT_DOCS_COUNT = 10;

  /** Graph labels are drawn only when at most this many nodes are
   * visible (see updateLabelVisibility). Zoom is deliberately NOT the
   * trigger: `text-max-width` is in model units, so labels and the gaps
   * between nodes scale together — a label ~3x wider than its node's
   * share of space overlaps its neighbours identically at every zoom
   * level. Only node count changes that ratio. Chosen so a doc's
   * neighbourhood view is labelled while the full KB stays dots. */
  const LABEL_MAX_NODES = 60;

  /** Minimum model-unit gap between nodes when labels are drawn, in the
   * same units as the node style's `text-max-width` (120) plus margin.
   * fcose packs edge-less nodes about 40 units apart — fine for dots,
   * but every label would then overlap its neighbours. See
   * spreadForLabels. */
  const LABEL_MIN_SPACING = 150;

  const darkMq = window.matchMedia("(prefers-color-scheme: dark)");

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------

  let bundleCache = null; // latest api/graph payload
  let nodeIndex = {}; // id -> node data
  let allEdges = []; // latest edge element list
  let backlinks = {}; // target id -> [source ids] (markdown edges only)

  let cy = null; // lazily-created Cytoscape instance
  let cyNeedsLayout = false; // graph structure changed while view inactive
  let lastGraphKey = null; // mode/root/depth/semantic of the last layout
  let labelsVisible = null; // last applied label state (see updateLabelVisibility)

  let activeRoute = { view: "home" };
  let currentDetailId = null; // doc shown in the doc view
  let docFetchToken = 0; // invalidates in-flight body fetches
  let searchDebounceTimer = null;
  let searchToken = 0;

  // -------------------------------------------------------------------
  // Theme-aware Cytoscape style
  // -------------------------------------------------------------------

  /** Canvas-relevant colors Cytoscape's style JSON can't pull from CSS
   * custom properties. Mirrors viz.css's tokens; the accent colors
   * (selection highlight, search-hit ring, local-graph root ring) are
   * identical in both modes and inlined in buildStyle(). */
  function themeColors() {
    return darkMq.matches
      ? { nodeLabel: "#e2e8f0", nodeBorder: "#e2e8f0", edgeLine: "#475569", labelBg: "#0b1120" }
      : { nodeLabel: "#0f172a", nodeBorder: "#0f172a", edgeLine: "#cbd5e1", labelBg: "#f8fafc" };
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
      // Level-of-detail, toggled by updateLabelVisibility. Cytoscape's
      // own `min-zoomed-font-size` can't express this: it compares
      // against `font-size * 2^ceil(log2(zoom * devicePixelRatio))`, so
      // the effective size jumps in power-of-two steps AND depends on
      // the viewer's display (11px at zoom 0.77 computes as 22px on a
      // retina screen, 11px elsewhere) — and zoom is the wrong trigger
      // anyway (see LABEL_MAX_NODES). The three rules after this one
      // re-show individual labels and must stay later in the array —
      // equal specificity, last wins.
      {
        selector: "node.no-label",
        style: { "text-opacity": 0 },
      },
      {
        // Force-show the hovered node's label regardless of zoom, with a
        // themed backing plate so it reads over neighboring elements.
        selector: "node.hover-label",
        style: {
          "text-opacity": 1,
          "font-size": 12,
          "text-background-color": t.labelBg,
          "text-background-opacity": 0.9,
          "text-background-padding": 3,
          "text-background-shape": "round-rectangle",
          "z-index": 998,
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 3,
          "border-color": "#f59e0b",
          "text-opacity": 1,
        },
      },
      {
        // The focused document in a local (neighborhood) graph.
        selector: "node.root-node",
        style: {
          "border-width": 4,
          "border-color": "#f59e0b",
          "text-opacity": 1,
          "z-index": 997,
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
      // Local-mode exclusion and the semantic-edge toggle, as classes so
      // mode switches never add/remove elements.
      {
        selector: ".excluded",
        style: { "display": "none" },
      },
      {
        selector: "edge.sem-hidden",
        style: { "display": "none" },
      },
    ];
  }

  // -------------------------------------------------------------------
  // Data layer
  // -------------------------------------------------------------------

  /** Adopt a fresh api/graph payload as the app-wide data source. */
  function setBundle(bundle) {
    bundleCache = bundle;
    nodeIndex = {};
    for (const n of bundle.nodes || []) nodeIndex[n.data.id] = n.data;
    allEdges = bundle.edges || [];
    rebuildBacklinks(allEdges);
    populateTypeFilter(bundle.types || []);
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

  /** Re-fetch `api/graph` and fold it into every live surface: the data
   * layer, the sidebar, the current view, and (if built) the Cytoscape
   * instance. Returns the raw bundle, or `null` if the fetch failed. */
  async function refreshGraph() {
    let bundle;
    try {
      const res = await fetch("api/graph");
      if (!res.ok) return null;
      bundle = await res.json();
    } catch (err) {
      return null;
    }

    const prevIds = Object.keys(nodeIndex);
    setBundle(bundle);
    if (cy) mergeIntoCy(bundle, prevIds);

    renderSidebar();
    if (activeRoute.view === "home") renderHome();
    else if (activeRoute.view === "browse") renderBrowse(activeRoute.prefix);
    return bundle;
  }

  /** Merge a fresh bundle into the live Cytoscape instance: update data on
   * existing nodes, add new ones, drop removed ones, and replace the edge
   * set wholesale (cheap, and edges carry no client-only state). Existing
   * node positions are preserved — a layout only reruns when the node set
   * actually changed, and only immediately when the graph view is on
   * screen (otherwise it is deferred to the next entry). */
  function mergeIntoCy(bundle, prevIds) {
    let changed = false;
    cy.batch(() => {
      for (const id of Object.keys(nodeIndex)) {
        const existing = cy.getElementById(id);
        if (existing && existing.length) {
          existing.data(nodeIndex[id]);
        } else {
          // Match the current label state, or a node added into a
          // dots-only graph would draw its label alone.
          const added = cy.add({ group: "nodes", data: nodeIndex[id] });
          if (labelsVisible === false) added.addClass("no-label");
          changed = true;
        }
      }
      for (const id of prevIds) {
        if (!(id in nodeIndex)) {
          cy.remove(cy.getElementById(id));
          changed = true;
        }
      }
      cy.edges().remove();
      cy.add(allEdges.map((e) => ({ group: "edges", data: e.data })));
    });

    if (activeRoute.view === "graph") {
      // Re-apply mode classes (the wholesale edge replacement dropped
      // them) and rerun the layout only if the node set changed.
      applyGraphView(changed);
      lastGraphKey = graphViewKey();
    } else {
      // Classes need re-applying on next entry even without a layout.
      cyNeedsLayout = true;
    }
  }

  /** Remove a single document from every surface, e.g. after a delete.
   * Navigates home first if the doc is currently being read. */
  function removeNodeById(id) {
    if (currentDetailId === id) {
      currentDetailId = null;
      navigate("#/");
    }
    if (bundleCache) {
      bundleCache.nodes = (bundleCache.nodes || []).filter((n) => n.data.id !== id);
      bundleCache.edges = (bundleCache.edges || []).filter(
        (e) => e.data.source !== id && e.data.target !== id
      );
    }
    delete nodeIndex[id];
    allEdges = allEdges.filter((e) => e.data.source !== id && e.data.target !== id);
    rebuildBacklinks(allEdges);
    if (cy) {
      const el = cy.getElementById(id);
      if (el && el.length) cy.remove(el);
    }
    renderSidebar();
    if (activeRoute.view === "home") renderHome();
    else if (activeRoute.view === "browse") renderBrowse(activeRoute.prefix);
  }

  // -------------------------------------------------------------------
  // Routing
  // -------------------------------------------------------------------

  /** Repo-relative doc/schema path segments must be individually
   * percent-encoded (not the whole path) so literal "/" keeps separating
   * path segments for the axum `{*path}` wildcard route. */
  function encodePathForApi(path) {
    return path.split("/").map(encodeURIComponent).join("/");
  }

  /** Same per-segment encoding for paths embedded in location.hash. */
  function encodeHashPath(path) {
    return path.split("/").map(encodeURIComponent).join("/");
  }

  function parseHash() {
    const h = window.location.hash.replace(/^#\/?/, "");
    if (!h) return { view: "home" };
    const segments = h.split("/");
    const kind = segments.shift();
    let rest;
    try {
      rest = segments.map(decodeURIComponent).join("/");
    } catch (err) {
      return { view: "home" }; // malformed percent-encoding
    }
    if (kind === "doc" && rest) return { view: "doc", path: rest };
    if (kind === "browse" && rest) return { view: "browse", prefix: rest };
    if (kind === "graph") return { view: "graph", root: rest || null };
    return { view: "home" };
  }

  /** Go to a hash route. Falls back to an explicit re-render when the
   * hash is already current (hashchange won't fire). */
  function navigate(hash) {
    if (window.location.hash === hash) route();
    else window.location.hash = hash;
  }

  /** Open a document in the doc view. Name and signature preserved from
   * the graph-era UI because edit.js calls this via window.KBViz. */
  function showDetail(id) {
    navigate(`#/doc/${encodeHashPath(id)}`);
  }

  const VIEW_IDS = ["view-home", "view-browse", "view-doc", "view-graph"];

  function route() {
    activeRoute = parseHash();

    for (const vid of VIEW_IDS) {
      document.getElementById(vid).hidden = vid !== `view-${activeRoute.view}`;
    }
    document.getElementById("app-error").hidden = true;

    switch (activeRoute.view) {
      case "doc":
        renderDoc(activeRoute.path);
        break;
      case "browse":
        renderBrowse(activeRoute.prefix);
        document.title = `${activeRoute.prefix} — Knowledge Base`;
        break;
      case "graph":
        renderGraphView(activeRoute.root);
        document.title = "Graph — Knowledge Base";
        break;
      default:
        renderHome();
        document.title = "Knowledge Base";
    }
    updateSidebarActive();
  }

  // -------------------------------------------------------------------
  // Sidebar document tree
  // -------------------------------------------------------------------

  function renderSidebar() {
    const treeEl = document.getElementById("doc-tree");
    const openDirs = new Set(
      Array.from(treeEl.querySelectorAll("details[open]")).map((d) => d.dataset.dir)
    );

    // Nested {dirs: Map<name, node>, docs: [nodeData]} tree from ids.
    const rootNode = { dirs: new Map(), docs: [] };
    for (const id of Object.keys(nodeIndex)) {
      const segments = id.split("/");
      let node = rootNode;
      for (let i = 0; i < segments.length - 1; i++) {
        if (!node.dirs.has(segments[i])) {
          node.dirs.set(segments[i], { dirs: new Map(), docs: [] });
        }
        node = node.dirs.get(segments[i]);
      }
      node.docs.push(nodeIndex[id]);
    }

    treeEl.innerHTML = "";
    appendTreeChildren(treeEl, rootNode, "", openDirs);
    updateSidebarActive();
  }

  function countDocs(node) {
    let n = node.docs.length;
    for (const child of node.dirs.values()) n += countDocs(child);
    return n;
  }

  function appendTreeChildren(parentEl, node, dirPath, openDirs) {
    const dirNames = Array.from(node.dirs.keys()).sort();
    for (const name of dirNames) {
      const child = node.dirs.get(name);
      const childPath = dirPath ? `${dirPath}/${name}` : name;
      const details = document.createElement("details");
      details.dataset.dir = childPath;
      if (openDirs.has(childPath)) details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = name + " ";
      const count = document.createElement("span");
      count.className = "tree-count";
      count.textContent = `(${countDocs(child)})`;
      summary.appendChild(count);
      details.appendChild(summary);
      const children = document.createElement("div");
      children.className = "tree-children";
      appendTreeChildren(children, child, childPath, openDirs);
      details.appendChild(children);
      parentEl.appendChild(details);
    }

    const docs = node.docs.slice().sort((a, b) =>
      (a.label || a.id).localeCompare(b.label || b.id)
    );
    for (const d of docs) {
      const a = document.createElement("a");
      a.className = "tree-doc";
      a.dataset.id = d.id;
      a.title = d.id;
      a.textContent = d.label || d.id;
      a.addEventListener("click", () => showDetail(d.id));
      parentEl.appendChild(a);
    }
  }

  /** Highlight the doc being read and expand its ancestor directories. */
  function updateSidebarActive() {
    const treeEl = document.getElementById("doc-tree");
    treeEl.querySelectorAll("a.tree-doc.active").forEach((a) => a.classList.remove("active"));
    if (activeRoute.view !== "doc") return;
    const active = treeEl.querySelector(
      `a.tree-doc[data-id="${CSS.escape(activeRoute.path)}"]`
    );
    if (!active) return;
    active.classList.add("active");
    let el = active.parentElement;
    while (el && el !== treeEl) {
      if (el.tagName === "DETAILS") el.open = true;
      el = el.parentElement;
    }
    active.scrollIntoView({ block: "nearest" });
  }

  // -------------------------------------------------------------------
  // Home view
  // -------------------------------------------------------------------

  function renderHome() {
    const docs = Object.values(nodeIndex);
    const domains = new Set();
    for (const d of docs) {
      if (d.id.includes("/")) domains.add(d.id.split("/")[0]);
    }
    const mdLinks = allEdges.filter((e) => e.data.kind !== "semantic").length;
    document.getElementById("home-stats").textContent =
      `${docs.length} documents · ${domains.size} domains · ${mdLinks} links`;

    const recentEl = document.getElementById("home-recent");
    recentEl.innerHTML = "";
    const recent = docs
      .slice()
      .sort((a, b) => (b.mtime || 0) - (a.mtime || 0))
      .slice(0, RECENT_DOCS_COUNT);
    for (const d of recent) recentEl.appendChild(makeDocRow(d, { showDesc: false }));

    const domainsEl = document.getElementById("home-domains");
    domainsEl.innerHTML = "";
    const counts = {};
    for (const d of docs) {
      if (!d.id.includes("/")) continue;
      const domain = d.id.split("/")[0];
      counts[domain] = (counts[domain] || 0) + 1;
    }
    for (const domain of Object.keys(counts).sort()) {
      const card = document.createElement("a");
      card.className = "domain-card";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = domain;
      const count = document.createElement("div");
      count.className = "count";
      count.textContent = `${counts[domain]} document${counts[domain] === 1 ? "" : "s"}`;
      card.appendChild(name);
      card.appendChild(count);
      card.addEventListener("click", () => navigate(`#/browse/${encodeHashPath(domain)}`));
      domainsEl.appendChild(card);
    }
  }

  // -------------------------------------------------------------------
  // Browse view
  // -------------------------------------------------------------------

  function renderBrowse(prefix) {
    document.getElementById("browse-title").textContent = prefix;
    const listEl = document.getElementById("browse-list");
    listEl.innerHTML = "";
    const docs = Object.values(nodeIndex)
      .filter((d) => d.id === prefix || d.id.startsWith(prefix + "/"))
      .sort((a, b) => a.id.localeCompare(b.id));
    document.getElementById("browse-count").textContent =
      `${docs.length} document${docs.length === 1 ? "" : "s"}`;
    for (const d of docs) listEl.appendChild(makeDocRow(d, { showDesc: true }));
  }

  /** A clickable one-doc row shared by the home (recent) and browse
   * listings: colored type dot, title, path, modified date, and
   * optionally the frontmatter description. */
  function makeDocRow(d, opts) {
    const row = document.createElement("a");
    row.className = "doc-row";

    const head = document.createElement("div");
    head.className = "doc-row-head";
    const dot = document.createElement("span");
    dot.className = "type-dot";
    dot.style.background = d.color || "var(--chip-bg)";
    dot.title = d.type || "";
    head.appendChild(dot);
    const title = document.createElement("span");
    title.className = "doc-row-title";
    title.textContent = d.label || d.id;
    head.appendChild(title);
    const path = document.createElement("span");
    path.className = "doc-row-path";
    path.textContent = d.id;
    head.appendChild(path);
    const meta = document.createElement("span");
    meta.className = "doc-row-meta";
    meta.textContent = formatMtime(d.mtime);
    head.appendChild(meta);
    row.appendChild(head);

    if (opts.showDesc && d.description) {
      const desc = document.createElement("div");
      desc.className = "doc-row-desc";
      const text = d.description;
      desc.textContent = text.length > 160 ? text.slice(0, 160) + "…" : text;
      row.appendChild(desc);
    }

    row.addEventListener("click", () => showDetail(d.id));
    return row;
  }

  // -------------------------------------------------------------------
  // Doc view
  // -------------------------------------------------------------------

  async function renderDoc(id) {
    currentDetailId = id;
    // Fallback data keeps the view functional for a path the index
    // doesn't know (stale bookmark, mid-reindex): the body fetch below
    // is independent of the metadata index.
    const data = nodeIndex[id] || { id, label: id.split("/").pop() };
    document.title = `${data.label || id} — Knowledge Base`;

    const chip = document.getElementById("detail-type");
    chip.textContent = data.type || "";
    chip.hidden = !data.type;
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

    // Lazy body fetch — token-guarded so navigating to another doc while
    // this one is in flight can't paint the stale body over the new one.
    const token = ++docFetchToken;
    const bodyEl = document.getElementById("detail-body");
    bodyEl.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const res = await fetch(`api/doc/${encodePathForApi(id)}`);
      if (token !== docFetchToken) return;
      if (!res.ok) {
        bodyEl.innerHTML = `<p class="muted">Could not load document (HTTP ${res.status}).</p>`;
        return;
      }
      const doc = await res.json();
      if (token !== docFetchToken) return;
      const body = stripFrontmatter(doc.content || "");
      bodyEl.innerHTML = renderMarkdown(body);
      rewriteInternalLinks(bodyEl, id);
    } catch (err) {
      if (token === docFetchToken) {
        bodyEl.innerHTML = '<p class="muted">Failed to load document.</p>';
      }
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

  // -------------------------------------------------------------------
  // Markdown rendering + internal links
  // -------------------------------------------------------------------

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

  // -------------------------------------------------------------------
  // Graph views (global + local)
  // -------------------------------------------------------------------

  /** Create the Cytoscape instance on first use. The grid layout here is
   * a cheap placeholder: every entry into the graph view that needs a
   * layout runs a real one via applyGraphView. */
  function ensureCy() {
    if (cy) return;
    cy = cytoscape({
      container: document.getElementById("graph"),
      elements: [...(bundleCache.nodes || []), ...(bundleCache.edges || [])],
      style: buildStyle(),
      layout: { name: "grid" },
      wheelSensitivity: 0.2,
    });
    cyNeedsLayout = true;

    // Repaint the canvas if the OS/browser theme flips while the page is
    // open — CSS handles the rest of the page via var(--token) cascades,
    // but Cytoscape's style is a JS object snapshot that has to be
    // explicitly rebuilt and reapplied.
    darkMq.addEventListener("change", () => {
      if (cy) cy.style(buildStyle());
    });

    cy.on("tap", "node", (evt) => {
      showDetail(evt.target.id());
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) cy.elements().unselect();
    });
    cy.on("mouseover", "node", (evt) => {
      evt.target.addClass("hover-label");
      document.getElementById("graph").style.cursor = "pointer";
    });
    cy.on("mouseout", "node", (evt) => {
      evt.target.removeClass("hover-label");
      document.getElementById("graph").style.cursor = "";
    });
  }

  function includeSemantic() {
    return document.getElementById("toggle-semantic").checked;
  }

  function graphDepth() {
    return parseInt(document.getElementById("graph-depth").value, 10) || 1;
  }

  function graphViewKey() {
    const root = activeRoute.view === "graph" ? activeRoute.root : null;
    return [root ? "local" : "global", root || "", graphDepth(), includeSemantic()].join("|");
  }

  /** The nodes within `depth` hops of `rootId`, following markdown edges
   * and (optionally) semantic edges. */
  function neighborhoodOf(rootId, depth, withSemantic) {
    const root = cy.getElementById(rootId);
    if (!root || !root.length) return cy.collection();
    let nodes = root;
    let frontier = root;
    for (let i = 0; i < depth; i++) {
      let edges = frontier.connectedEdges();
      if (!withSemantic) edges = edges.filter('[kind != "semantic"]');
      const next = edges.connectedNodes().difference(nodes);
      if (!next.length) break;
      nodes = nodes.union(next);
      frontier = next;
    }
    return nodes;
  }

  /** Apply the current graph mode (global vs local root/depth, semantic
   * toggle) as classes on the persistent element set, then optionally
   * rerun the layout over what's visible. */
  function applyGraphView(runLayout) {
    const root = activeRoute.view === "graph" ? activeRoute.root : null;
    cy.batch(() => {
      cy.elements().removeClass("excluded root-node");
      cy.edges('[kind = "semantic"]').toggleClass("sem-hidden", !includeSemantic());
      if (root) {
        const keep = neighborhoodOf(root, graphDepth(), includeSemantic());
        if (keep.length) {
          cy.nodes().not(keep).addClass("excluded");
          cy.getElementById(root).addClass("root-node");
        } else {
          // Root not in the graph (stale path): show nothing rather than
          // silently falling back to the full KB.
          cy.nodes().addClass("excluded");
        }
      }
    });
    updateLabelVisibility();
    if (runLayout) runGraphLayout();
  }

  function visibleGraphEles() {
    return cy.elements().not(".excluded").not(".sem-hidden");
  }

  /** Fit the viewport to `eles`, but never closer than 1.5x — fitting a
   * one-node neighborhood would otherwise fill the screen with a single
   * giant dot. */
  function fitGraph(eles) {
    cy.fit(eles, 30);
    if (cy.zoom() > 1.5) {
      cy.zoom(1.5);
      cy.center(eles);
    }
  }

  /** Scale a laid-out node set apart until neighbours are at least
   * LABEL_MIN_SPACING apart, so the labels drawn over them don't
   * collide. Uniform scaling about the centroid, so it preserves the
   * layout's structure exactly — it only changes the scale, and the
   * subsequent fit absorbs that. Only runs when labels are actually
   * drawn, which bounds the O(n^2) distance scan to LABEL_MAX_NODES
   * (and is why the full-KB view — where spreading would shrink the fit
   * to unreadable dust — never reaches it). */
  function spreadForLabels(eles) {
    if (!labelsVisible) return;
    const nodes = eles.nodes();
    if (nodes.length < 2) return;

    const pos = nodes.map((n) => n.position());
    let minD = Infinity;
    for (let i = 0; i < pos.length; i++) {
      for (let j = i + 1; j < pos.length; j++) {
        const d = Math.hypot(pos[i].x - pos[j].x, pos[i].y - pos[j].y);
        if (d < minD) minD = d;
      }
    }
    if (!(minD > 0) || minD >= LABEL_MIN_SPACING) return;

    const factor = LABEL_MIN_SPACING / minD;
    const bb = nodes.boundingBox();
    const originX = (bb.x1 + bb.x2) / 2;
    const originY = (bb.y1 + bb.y2) / 2;
    cy.batch(() => {
      nodes.forEach((n) => {
        const p = n.position();
        n.position({
          x: originX + (p.x - originX) * factor,
          y: originY + (p.y - originY) * factor,
        });
      });
    });
  }

  /** Draw node labels only while the visible set is small enough to read
   * (see LABEL_MAX_NODES) — in practice, a doc's neighbourhood but not
   * the full KB. Hovered, selected, and neighbourhood-root nodes keep
   * their labels either way via later style rules, so a dots-only graph
   * is still fully identifiable by pointing at it. Called whenever the
   * visible set changes, not on zoom. */
  function updateLabelVisibility() {
    const show = cy.nodes().not(".excluded").length <= LABEL_MAX_NODES;
    if (show === labelsVisible) return;
    labelsVisible = show;
    cy.batch(() => {
      cy.nodes().toggleClass("no-label", !show);
    });
  }

  function runGraphLayout() {
    const eles = visibleGraphEles();
    if (!eles.nodes().length) return;
    const name = document.getElementById("layout").value;
    // fit: false is load-bearing: every cytoscape layout fits the
    // viewport itself by default when it finishes, and fcose finishes
    // asynchronously — its built-in fit would land AFTER (and silently
    // overwrite) fitGraph's capped fit, zooming a densely-packed
    // full-KB layout past the label threshold. Instead, fit exactly
    // once per layout, capped, on layoutstop (which fires within
    // run() for synchronous layouts and at completion for async ones).
    const opts = { name, animate: false, padding: 30, fit: false };
    if (name === "fcose") {
      opts.quality = "default";
      // Pack the KB's many disconnected components instead of scattering
      // them — the main reason fcose is the default over built-in cose.
      opts.packComponents = true;
    }
    let layout;
    try {
      layout = eles.layout(opts);
    } catch (err) {
      // fcose plugin missing/failed — built-in force layout still works.
      layout = eles.layout({ name: "cose", animate: false, padding: 30, fit: false });
    }
    layout.one("layoutstop", () => {
      spreadForLabels(eles);
      fitGraph(eles);
    });
    layout.run();
  }

  function renderGraphView(root) {
    ensureCy();
    cy.resize();

    const label = document.getElementById("graph-mode-label");
    const localControls = document.getElementById("graph-local-controls");
    if (root) {
      const data = nodeIndex[root];
      label.textContent = `Neighborhood: ${data ? data.label || root : root}`;
      localControls.hidden = false;
    } else {
      label.textContent = `Full knowledge base — ${Object.keys(nodeIndex).length} documents`;
      localControls.hidden = true;
    }

    // Only recompute classes + rerun the layout when the view actually
    // changed (mode/root/depth/semantic) or the structure changed while
    // the view was inactive; plain re-entry preserves positions and the
    // user's pan/zoom.
    const key = graphViewKey();
    if (key !== lastGraphKey || cyNeedsLayout) {
      applyGraphView(true);
      lastGraphKey = key;
      cyNeedsLayout = false;
    }
  }

  // -------------------------------------------------------------------
  // Search: instant substring dim on the graph + debounced semantic
  // query rendered as a results panel (and highlighted on the graph).
  // -------------------------------------------------------------------

  function onSearchInput(e) {
    const raw = e.target.value;
    const q = raw.trim().toLowerCase();

    if (cy) {
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
    }

    clearTimeout(searchDebounceTimer);
    if (!q) {
      if (cy) cy.elements().removeClass("search-hit");
      searchToken++; // invalidate any in-flight request
      hideSearchResults();
      return;
    }
    searchDebounceTimer = setTimeout(() => {
      runSemanticSearch(raw.trim());
    }, SEARCH_DEBOUNCE_MS);
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

    if (cy) {
      cy.elements().removeClass("search-hit");
      const hitIds = new Set((data.results || []).map((r) => r.file_path));
      const hits = cy.nodes().filter((n) => hitIds.has(n.id()));
      hits.addClass("search-hit");
      if (hits.length && activeRoute.view === "graph") {
        cy.animate({ fit: { eles: hits, padding: 40 } }, { duration: 200 });
      }
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

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------

  async function init() {
    const loadingEl = document.getElementById("app-loading");
    const errorEl = document.getElementById("app-error");

    let bundle;
    try {
      const res = await fetch("api/graph");
      if (!res.ok) {
        throw new Error(`api/graph returned ${res.status}`);
      }
      bundle = await res.json();
    } catch (err) {
      loadingEl.hidden = true;
      errorEl.hidden = false;
      errorEl.textContent = `Failed to load the knowledge base: ${err.message}`;
      return;
    }
    loadingEl.hidden = true;

    setBundle(bundle);
    renderSidebar();

    document.getElementById("search").addEventListener("input", onSearchInput);
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

    document.getElementById("local-graph-btn").addEventListener("click", () => {
      if (currentDetailId) navigate(`#/graph/${encodeHashPath(currentDetailId)}`);
    });

    // Graph toolbar. Depth/semantic changes alter the visible set, so
    // they re-render (the view key change triggers a fresh layout);
    // layout-select changes rerun the layout in place.
    document.getElementById("graph-depth").addEventListener("change", () => {
      if (activeRoute.view === "graph") renderGraphView(activeRoute.root);
    });
    document.getElementById("toggle-semantic").addEventListener("change", () => {
      if (activeRoute.view === "graph") renderGraphView(activeRoute.root);
    });
    document.getElementById("layout").addEventListener("change", () => {
      if (cy && activeRoute.view === "graph") runGraphLayout();
    });
    document.getElementById("reset").addEventListener("click", () => {
      if (!cy) return;
      fitGraph(visibleGraphEles());
      cy.elements().unselect();
    });
    document.getElementById("filter-type").addEventListener("change", (e) => {
      if (!cy) return;
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

    window.addEventListener("hashchange", route);
    route();
  }

  /** Integration surface for assets/edit.js (a separate script — see this
   * file's header comment). Deliberately small: edit.js drives its own DOM
   * (the editor overlay) and only needs to read the current doc/graph
   * state and push updates back in after a save/create/delete. The names
   * and semantics predate the docs-first shell and are kept stable. */
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
