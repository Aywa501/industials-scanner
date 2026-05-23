// v2 phase3 candidate labeler — one chip per candidate, keyboard-driven.

const state = {
  queue: [],
  labels: new Map(),
  notes: new Map(),
  flags: new Set(),    // tile_ids flagged "follow_up"
  idx: 0,
  wide: false,         // current zoom toggle
  showPins: true,      // overlay POI pins
};

const CHIP_HALF_M = { default: 1280, wide: 2560 };

const KEY_TO_LABEL = { "1": "not_industrial", "2": "industrial", "3": "unsure" };
const PROB_BUCKETS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.0];  // descending

async function init() {
  const [queue, labels, notes, flags] = await Promise.all([
    fetch("/api/queue").then(r => r.json()),
    fetch("/api/labels").then(r => r.json()),
    fetch("/api/notes").then(r => r.json()),
    fetch("/api/flags").then(r => r.json()),
  ]);
  state.queue = queue;
  for (const r of labels) state.labels.set(r.tile_id, r.label);
  for (const r of notes) state.notes.set(r.tile_id, r.note);
  for (const r of flags) if (r.flag === "follow_up") state.flags.add(r.tile_id);

  const firstUnlabeled = state.queue.findIndex(c => !state.labels.has(c.tile_id));
  state.idx = firstUnlabeled >= 0 ? firstUnlabeled : 0;

  render();
  window.addEventListener("keydown", onKey);
}

function current() { return state.queue[state.idx]; }

function probBucketClass(p) {
  if (p >= 0.9) return "hi";
  if (p >= 0.7) return "mid";
  return "lo";
}

function render() {
  const c = current();
  if (!c) {
    document.getElementById("tile-id").textContent = "all done!";
    document.getElementById("chip").src = "";
    return;
  }

  document.getElementById("prog").textContent =
    `[${state.idx + 1} / ${state.queue.length}]`;
  document.getElementById("tile-id").textContent = c.tile_id;

  const probBadge = document.getElementById("prob-badge");
  probBadge.textContent = `prob ${c.prob.toFixed(3)}`;
  probBadge.className = probBucketClass(c.prob);

  document.getElementById("mgrs-badge").textContent = c.mgrs_tile;
  document.getElementById("scene-date").textContent = c.scene_date || "";

  const qBadge = document.getElementById("quality-badge");
  const q = c.chip_quality || "ok";
  qBadge.textContent = q === "ok" ? "" : q;
  qBadge.className = q;

  const fBadge = document.getElementById("flag-badge");
  if (state.flags.has(c.tile_id)) {
    fBadge.textContent = "⚑ follow-up";
    fBadge.className = "";
  } else {
    fBadge.className = "hidden";
  }

  const lat = c.lat.toFixed(5);
  const lon = c.lon.toFixed(5);
  const coords = document.getElementById("coords");
  coords.textContent = `${lat}, ${lon}`;
  coords.href = `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
  document.getElementById("gmaps").href =
    `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;

  const noteInput = document.getElementById("note-input");
  if (document.activeElement !== noteInput) {
    noteInput.value = state.notes.get(c.tile_id) || "";
  }

  const labeled = state.labels.size;
  const total = state.queue.length;
  const byLabel = { industrial: 0, not_industrial: 0, unsure: 0 };
  for (const v of state.labels.values()) if (byLabel[v] !== undefined) byLabel[v]++;
  document.getElementById("counts").textContent =
    `${labeled} / ${total} · ind=${byLabel.industrial} not=${byLabel.not_industrial} unsure=${byLabel.unsure}`;

  const img = document.getElementById("chip");
  const prefix = state.wide ? "/chips_wide" : "/chips";
  img.src = `${prefix}/${c.tile_id}.png`;
  img.alt = c.tile_id;
  prefetchNeighbors();
  renderPins(c);

  const lab = state.labels.get(c.tile_id);
  img.className = lab ? `label-${lab}` : "";
  const badge = document.getElementById("chip-label-badge");
  if (lab) {
    badge.textContent = lab.replace("_", " ");
    badge.className = `label-${lab}`;
    badge.style.display = "";
  } else {
    badge.style.display = "none";
  }

  renderContext(c);
}

function renderContext(c) {
  const anchorEl = document.getElementById("ctx-anchor");
  const indEl = document.getElementById("ctx-industrial");
  const namedEl = document.getElementById("ctx-named");

  if (c.nearest_anchor) {
    const a = c.nearest_anchor;
    const bits = [a.company, a.sector, a.site_type].filter(Boolean).join(" · ");
    anchorEl.className = "ctx-section anchor";
    anchorEl.innerHTML = `
      <div class="ctx-title"><span>nearest known site</span><span>${a.distance_m} m</span></div>
      <div class="ctx-name">${escapeHtml(a.name || "(unnamed)")}</div>
      <div class="ctx-meta">${escapeHtml(bits || "—")}${a.state ? " · " + a.state : ""}</div>`;
  } else {
    anchorEl.className = "ctx-section empty";
    anchorEl.innerHTML = "";
  }

  if (c.nearest_industrial || c.industrial_buildings_500m) {
    const n = c.nearest_industrial;
    const count = c.industrial_buildings_500m || 0;
    indEl.className = "ctx-section industrial";
    if (n) {
      const nm = n.name ? escapeHtml(n.name) : `(${escapeHtml(n.class || "industrial")})`;
      const area = n.area_m2 ? `${n.area_m2.toLocaleString()} m²` : "";
      indEl.innerHTML = `
        <div class="ctx-title"><span>nearest industrial bldg</span><span>${n.distance_m} m</span></div>
        <div class="ctx-name">${nm}</div>
        <div class="ctx-meta">${escapeHtml(n.class || "")}${area ? " · " + area : ""} · ${count} within 500m</div>`;
    } else {
      indEl.innerHTML = `
        <div class="ctx-title"><span>industrial buildings nearby</span></div>
        <div class="ctx-meta">${count} within 500m</div>`;
    }
  } else {
    indEl.className = "ctx-section empty";
    indEl.innerHTML = "";
  }

  if (c.nearest_named_building) {
    const n = c.nearest_named_building;
    const area = n.area_m2 ? `${n.area_m2.toLocaleString()} m²` : "";
    const kind = [n.class, n.subtype].filter(Boolean).join(" / ");
    namedEl.className = "ctx-section";
    namedEl.innerHTML = `
      <div class="ctx-title"><span>nearest named POI</span><span>${n.distance_m} m</span></div>
      <div class="ctx-name">${escapeHtml(n.name)}</div>
      <div class="ctx-meta">${escapeHtml(kind || "—")}${area ? " · " + area : ""}</div>`;
  } else {
    namedEl.className = "ctx-section empty";
    namedEl.innerHTML = "";
  }
}

function poiToNormalized(poiLat, poiLon, cLat, cLon, halfM) {
  const dy = (poiLat - cLat) * 111320;
  const dx = (poiLon - cLon) * 111320 * Math.cos(cLat * Math.PI / 180);
  const x = 0.5 + dx / (2 * halfM);
  const y = 0.5 - dy / (2 * halfM);  // SVG y grows down
  return { x, y };
}

function renderPins(c) {
  const svg = document.getElementById("chip-overlay");
  const NS = "http://www.w3.org/2000/svg";
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  svg.classList.toggle("hidden", !state.showPins);
  if (!state.showPins) return;

  const halfM = state.wide ? CHIP_HALF_M.wide : CHIP_HALF_M.default;

  // Center crosshair = candidate itself
  const cross = document.createElementNS(NS, "g");
  cross.setAttribute("class", "pin-center");
  const cx = document.createElementNS(NS, "line");
  cx.setAttribute("x1", "0.485"); cx.setAttribute("x2", "0.515");
  cx.setAttribute("y1", "0.5"); cx.setAttribute("y2", "0.5");
  cx.setAttribute("stroke", "#fff"); cx.setAttribute("stroke-width", "0.004");
  const cy = document.createElementNS(NS, "line");
  cy.setAttribute("x1", "0.5"); cy.setAttribute("x2", "0.5");
  cy.setAttribute("y1", "0.485"); cy.setAttribute("y2", "0.515");
  cy.setAttribute("stroke", "#fff"); cy.setAttribute("stroke-width", "0.004");
  cross.appendChild(cx); cross.appendChild(cy);
  svg.appendChild(cross);

  const sets = [
    { items: c.anchor_pins || [], cls: "pin-anchor", r: 0.018, label: p => p.name || "(anchor)" },
    { items: c.industrial_pins || [], cls: "pin-industrial", r: 0.014, label: p => p.name || `(${p.class})` },
    { items: c.named_pins || [], cls: "pin-named", r: 0.011, label: p => p.name },
  ];
  for (const set of sets) {
    for (const p of set.items) {
      const { x, y } = poiToNormalized(p.lat, p.lon, c.lat, c.lon, halfM);
      // Allow a small overshoot so pins right at the edge still render
      if (x < -0.02 || x > 1.02 || y < -0.02 || y > 1.02) continue;
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", `pin ${set.cls}`);
      const cir = document.createElementNS(NS, "circle");
      cir.setAttribute("cx", x); cir.setAttribute("cy", y);
      cir.setAttribute("r", set.r);
      const tt = document.createElementNS(NS, "title");
      const dist = p.distance_m != null ? ` (${p.distance_m} m)` : "";
      tt.textContent = `${set.label(p)}${dist}`;
      g.appendChild(cir); g.appendChild(tt);
      svg.appendChild(g);
    }
  }
}

function prefetchNeighbors() {
  const prefix = state.wide ? "/chips_wide" : "/chips";
  for (let d = 1; d <= 3; d++) {
    const next = state.queue[state.idx + d];
    if (next) new Image().src = `${prefix}/${next.tile_id}.png`;
    const prev = state.queue[state.idx - d];
    if (prev) new Image().src = `${prefix}/${prev.tile_id}.png`;
  }
}

function jumpProbBucket(dir) {
  // dir = +1 → toward lower prob (next boundary down), -1 → higher prob
  const cur = current().prob;
  if (dir > 0) {
    // find first index with prob strictly less than the next boundary below cur
    const nextBound = PROB_BUCKETS.find(b => b < cur);
    if (nextBound === undefined) return state.idx;
    for (let i = state.idx + 1; i < state.queue.length; i++) {
      if (state.queue[i].prob < nextBound + 1e-9) return i;
    }
    return state.queue.length - 1;
  } else {
    const prevBound = [...PROB_BUCKETS].reverse().find(b => b > cur);
    if (prevBound === undefined) return state.idx;
    for (let i = state.idx - 1; i >= 0; i--) {
      if (state.queue[i].prob >= prevBound - 1e-9) return i;
    }
    return 0;
  }
}

async function postFlag(tileId, flag) {
  const has = state.flags.has(tileId);
  const method = has ? "DELETE" : "POST";
  const res = await fetch("/api/flag", {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tile_id: tileId, flag }),
  });
  if (!res.ok) return false;
  if (has) state.flags.delete(tileId);
  else state.flags.add(tileId);
  setStatus(has ? "unflagged" : "flagged for follow-up");
  return true;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

async function postLabel(tileId, label) {
  const res = await fetch("/api/label", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tile_id: tileId, label }),
  });
  if (!res.ok) { setStatus(`label POST failed: ${res.status}`); return false; }
  const j = await res.json();
  setStatus(`saved · ${j.n_labels} total`);
  return true;
}

async function deleteLabel(tileId) {
  const res = await fetch("/api/label", {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tile_id: tileId }),
  });
  if (!res.ok) return false;
  const j = await res.json();
  setStatus(`cleared · ${j.n_labels} total`);
  return true;
}

async function postNote(tileId, note) {
  const res = await fetch("/api/note", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tile_id: tileId, note }),
  });
  if (!res.ok) { setStatus(`note POST failed: ${res.status}`); return false; }
  const j = await res.json();
  setStatus(note.trim() ? `note saved · ${j.n_notes}` : `note cleared · ${j.n_notes}`);
  return true;
}

function setStatus(s) {
  document.getElementById("status").textContent = s;
}

function nextUnlabeled() {
  for (let i = state.idx + 1; i < state.queue.length; i++) {
    if (!state.labels.has(state.queue[i].tile_id)) return i;
  }
  for (let i = 0; i < state.idx; i++) {
    if (!state.labels.has(state.queue[i].tile_id)) return i;
  }
  return state.idx;
}

async function onKey(e) {
  const c = current();
  if (!c) return;
  const noteInput = document.getElementById("note-input");

  if (document.activeElement === noteInput) {
    if (e.key === "Enter" || e.key === "Escape") {
      const text = noteInput.value;
      if (text.trim()) state.notes.set(c.tile_id, text.trim());
      else state.notes.delete(c.tile_id);
      noteInput.blur();
      await postNote(c.tile_id, text);
      e.preventDefault();
    }
    return;
  }

  if (e.key === "ArrowRight" || e.key === "Enter" || e.key === "ArrowDown") {
    state.idx = Math.min(state.queue.length - 1, state.idx + 1);
    render(); e.preventDefault();
  } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
    state.idx = Math.max(0, state.idx - 1);
    render(); e.preventDefault();
  } else if (e.key === "j") {
    state.idx = nextUnlabeled();
    render(); e.preventDefault();
  } else if (KEY_TO_LABEL[e.key]) {
    const label = KEY_TO_LABEL[e.key];
    state.labels.set(c.tile_id, label);
    render();
    await postLabel(c.tile_id, label);
    state.idx = nextUnlabeled();
    render();
    e.preventDefault();
  } else if (e.key === "x") {
    state.labels.delete(c.tile_id);
    render();
    await deleteLabel(c.tile_id);
    e.preventDefault();
  } else if (e.key === "n") {
    noteInput.focus();
    noteInput.select();
    e.preventDefault();
  } else if (e.key === "g") {
    window.open(document.getElementById("gmaps").href, "_blank");
    e.preventDefault();
  } else if (e.key === "z") {
    state.wide = !state.wide;
    setStatus(state.wide ? "wide view (5.1 km)" : "default view (2.56 km)");
    render(); e.preventDefault();
  } else if (e.key === "p") {
    state.showPins = !state.showPins;
    setStatus(state.showPins ? "pins on" : "pins off");
    renderPins(current()); e.preventDefault();
  } else if (e.key === "f") {
    await postFlag(c.tile_id, "follow_up");
    render(); e.preventDefault();
  } else if (e.key === "[") {
    state.idx = jumpProbBucket(-1);
    render(); e.preventDefault();
  } else if (e.key === "]") {
    state.idx = jumpProbBucket(+1);
    render(); e.preventDefault();
  }
}

init();
