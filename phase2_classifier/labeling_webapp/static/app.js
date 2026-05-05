// Manual S2 lifecycle labeler.
// Strip view: one site per screen, all years horizontally, keyboard-driven.

const state = {
  queue: [],
  labels: new Map(), // key: `${site_id}|${year}` -> label
  flags: new Map(),  // key: `${site_id}|${flag}` -> true
  notes: new Map(),  // key: site_id -> note text
  outlines: new Map(), // key: site_id -> [[x,y], ...] in normalized [0,1]
  draftOutline: null,  // [[x,y], ...] while in draw mode; null otherwise
  siteIdx: 0,
  yearIdx: 0,
  showHeatmap: false,  // toggle with 'h'
};

const KEY_TO_LABEL = { "1": "not_a_site", "2": "partial", "3": "complete", "u": "unsure" };

async function init() {
  const [queue, labels, flags, notes, outlines] = await Promise.all([
    fetch("/api/queue").then(r => r.json()),
    fetch("/api/labels").then(r => r.json()),
    fetch("/api/flags").then(r => r.json()),
    fetch("/api/notes").then(r => r.json()),
    fetch("/api/outlines").then(r => r.json()),
  ]);
  state.queue = queue;
  for (const r of labels) state.labels.set(`${r.site_id}|${r.year}`, r.label);
  for (const r of flags) state.flags.set(`${r.site_id}|${r.flag}`, true);
  for (const r of notes) state.notes.set(r.site_id, r.note);
  for (const r of outlines) state.outlines.set(r.site_id, r.polygon);

  // Land on the first site that has any unlabeled chip
  for (let i = 0; i < state.queue.length; i++) {
    const s = state.queue[i];
    if (s.years.some(y => !state.labels.has(`${s.site_id}|${y}`))) {
      state.siteIdx = i;
      break;
    }
  }
  render();
  window.addEventListener("keydown", onKey);
}

function currentSite() { return state.queue[state.siteIdx]; }

function render() {
  const site = currentSite();
  if (!site) {
    document.getElementById("site-name").textContent = "all done!";
    document.getElementById("strip").innerHTML = "";
    return;
  }

  document.getElementById("site-progress").textContent =
    `[${state.siteIdx + 1} / ${state.queue.length}]`;
  document.getElementById("site-name").textContent =
    site.canonical_project_name || "(random CONUS)";
  document.getElementById("site-state").textContent =
    site.state ? `${site.state}` : "";
  const badge = document.getElementById("site-type-badge");
  badge.textContent = site.site_type;
  badge.className = site.site_type === "anchor" ? "" : "negative";
  const projBadge = document.getElementById("project-type-badge");
  if (site.project_type) {
    projBadge.textContent = site.project_type.replace("_", " ");
    projBadge.className = site.project_type;
    projBadge.style.display = "";
  } else {
    projBadge.textContent = "";
    projBadge.style.display = "none";
  }
  document.getElementById("site-ann").textContent =
    site.ann_year ? `announced ${site.ann_year}` : "";
  document.getElementById("site-flag").textContent =
    state.flags.get(`${site.site_id}|bad_geocode`) ? "⚠ bad geocode" : "";

  const coords = document.getElementById("site-coords");
  if (coords) {
    if (typeof site.lat === "number" && typeof site.lng === "number") {
      const lat = site.lat.toFixed(5);
      const lng = site.lng.toFixed(5);
      coords.textContent = `${lat}, ${lng} ↗`;
      coords.href = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
      coords.title = "open in Google Maps";
      coords.style.display = "";
    } else {
      coords.textContent = "";
      coords.style.display = "none";
    }
  }

  const noteInput = document.getElementById("note-input");
  if (document.activeElement !== noteInput) {
    noteInput.value = state.notes.get(site.site_id) || "";
  }

  // Counts
  const total = state.queue.reduce((acc, s) => acc + s.years.length, 0);
  const labeled = state.labels.size;
  document.getElementById("counts").textContent =
    `${labeled} / ${total} labels`;

  // Reference thumbnail: always show the latest available year for this site
  const latestYear = site.years[site.years.length - 1];
  document.getElementById("ref-year").textContent = latestYear;
  document.getElementById("ref-img").src = `/chips/${site.site_id}/${latestYear}.png`;

  // Reference outline overlay
  const refOutline = state.outlines.get(site.site_id);
  renderOutlineOverlay(document.getElementById("ref-outline"), refOutline, null);

  // Strip
  const strip = document.getElementById("strip");
  strip.innerHTML = "";
  if (state.yearIdx >= site.years.length) state.yearIdx = site.years.length - 1;
  if (state.yearIdx < 0) state.yearIdx = 0;
  const draftOnFocused = state.draftOutline;
  const savedOutline = state.outlines.get(site.site_id) || null;
  site.years.forEach((y, i) => {
    const div = document.createElement("div");
    div.className = "chip";
    if (i === state.yearIdx) div.classList.add("focused");
    if (draftOnFocused && i === state.yearIdx) div.classList.add("drawing");
    const lab = state.labels.get(`${site.site_id}|${y}`);
    if (lab) div.classList.add(`label-${lab}`);

    const annTag = document.createElement("div");
    annTag.className = "ann-tag";
    const flagged = Array.isArray(site.flagged_years) && site.flagged_years.includes(y);
    if (site.ann_year === y) {
      annTag.textContent = "★ announced";
      div.classList.add("ann-year");
    } else if (flagged) {
      annTag.textContent = "⚑ shortlist";
      div.classList.add("flagged-year");
    }
    div.appendChild(annTag);

    const imgWrap = document.createElement("div");
    imgWrap.className = "img-wrap";
    const img = document.createElement("img");
    img.src = `/chips/${site.site_id}/${y}.png`;
    img.alt = `${site.site_id} ${y}`;
    imgWrap.appendChild(img);

    if (state.showHeatmap) {
      const heat = document.createElement("img");
      heat.className = "heatmap-overlay";
      heat.src = `/heatmaps/${site.site_id}/${y}.png`;
      heat.alt = "model heatmap";
      heat.onerror = () => { heat.style.display = "none"; };
      imgWrap.appendChild(heat);
    }

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 1 1");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.classList.add("outline-svg");
    imgWrap.appendChild(svg);
    const showDraft = draftOnFocused && i === state.yearIdx;
    renderOutlineOverlay(svg, showDraft ? null : savedOutline, showDraft ? draftOnFocused : null);

    if (showDraft) {
      img.addEventListener("click", (e) => {
        const rect = img.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const yN = (e.clientY - rect.top) / rect.height;
        state.draftOutline.push([Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, yN))]);
        render();
        e.stopPropagation();
      });
    }
    div.appendChild(imgWrap);

    if (lab) {
      const b = document.createElement("div");
      b.className = "badge";
      b.textContent = lab.replace("_", " ");
      div.appendChild(b);
    }

    const yr = document.createElement("div");
    yr.className = "year";
    yr.textContent = y;
    div.appendChild(yr);

    div.addEventListener("click", () => {
      state.yearIdx = i;
      render();
    });
    strip.appendChild(div);
  });
  // Center the focused chip in the strip viewport
  const focused = strip.querySelector(".chip.focused");
  if (focused) {
    focused.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }
}

function renderOutlineOverlay(svg, savedPoly, draftPoly) {
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const NS = "http://www.w3.org/2000/svg";
  const drawClosed = (pts, cls) => {
    if (!pts || pts.length < 2) return;
    const el = document.createElementNS(NS, pts.length >= 3 ? "polygon" : "polyline");
    el.setAttribute("points", pts.map(p => `${p[0]},${p[1]}`).join(" "));
    el.classList.add(cls);
    svg.appendChild(el);
  };
  const drawDots = (pts, cls) => {
    for (const p of pts) {
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", p[0]); c.setAttribute("cy", p[1]); c.setAttribute("r", 0.012);
      c.classList.add(cls);
      svg.appendChild(c);
    }
  };
  if (savedPoly) drawClosed(savedPoly, "outline-saved");
  if (draftPoly) {
    drawClosed(draftPoly, "outline-draft");
    drawDots(draftPoly, "outline-draft-dot");
  }
}

async function postOutline(siteId, polygon) {
  const res = await fetch("/api/outline", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId, polygon }),
  });
  if (!res.ok) {
    setStatus(`outline POST failed: ${res.status}`);
    return false;
  }
  const j = await res.json();
  setStatus(`outline saved · ${j.n_outlines} total`);
  return true;
}

async function deleteOutlineApi(siteId) {
  const res = await fetch("/api/outline", {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId }),
  });
  if (!res.ok) return false;
  const j = await res.json();
  setStatus(`outline cleared · ${j.n_outlines} total`);
  return true;
}

async function postLabel(siteId, year, label) {
  const res = await fetch("/api/label", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId, year, label }),
  });
  if (!res.ok) {
    setStatus(`label POST failed: ${res.status}`);
    return false;
  }
  const j = await res.json();
  setStatus(`saved · ${j.n_labels} total`);
  return true;
}

async function deleteLabel(siteId, year) {
  const res = await fetch("/api/label", {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId, year }),
  });
  if (!res.ok) return false;
  const j = await res.json();
  setStatus(`cleared · ${j.n_labels} total`);
  return true;
}

async function postNote(siteId, note) {
  const res = await fetch("/api/note", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId, note }),
  });
  if (!res.ok) {
    setStatus(`note POST failed: ${res.status}`);
    return false;
  }
  const j = await res.json();
  setStatus(note.trim() ? `note saved · ${j.n_notes} total` : `note cleared · ${j.n_notes} total`);
  return true;
}

async function postFlag(siteId, flag) {
  const key = `${siteId}|${flag}`;
  const has = state.flags.get(key);
  const method = has ? "DELETE" : "POST";
  const res = await fetch("/api/flag", {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ site_id: siteId, flag }),
  });
  if (!res.ok) return false;
  if (has) state.flags.delete(key);
  else state.flags.set(key, true);
  setStatus(has ? `unflagged ${flag}` : `flagged ${flag}`);
  return true;
}

function setStatus(s) {
  document.getElementById("status").textContent = s;
}

async function onKey(e) {
  const site = currentSite();
  if (!site) return;
  const year = site.years[state.yearIdx];
  const noteInput = document.getElementById("note-input");

  // While typing in the note field, intercept Enter/Esc to commit; ignore other keys
  if (document.activeElement === noteInput) {
    if (e.key === "Enter" || e.key === "Escape") {
      const text = noteInput.value;
      if (text.trim()) state.notes.set(site.site_id, text.trim());
      else state.notes.delete(site.site_id);
      noteInput.blur();
      await postNote(site.site_id, text);
      e.preventDefault();
    }
    return;
  }

  // While drawing an outline, only Enter/Escape/Backspace/Shift+O are meaningful
  if (state.draftOutline) {
    if (e.key === "Enter") {
      if (state.draftOutline.length >= 3) {
        const poly = state.draftOutline;
        state.outlines.set(site.site_id, poly);
        state.draftOutline = null;
        render();
        await postOutline(site.site_id, poly);
      } else {
        setStatus(`need ≥3 points (have ${state.draftOutline.length})`);
      }
      e.preventDefault();
      return;
    }
    if (e.key === "Escape") {
      state.draftOutline = null;
      setStatus("outline draw cancelled");
      render();
      e.preventDefault();
      return;
    }
    if (e.key === "Backspace") {
      state.draftOutline.pop();
      render();
      e.preventDefault();
      return;
    }
    if (e.key === "O") {
      state.draftOutline = null;
      const hadSaved = state.outlines.has(site.site_id);
      if (hadSaved) state.outlines.delete(site.site_id);
      render();
      if (hadSaved) await deleteOutlineApi(site.site_id);
      else setStatus("draft cleared");
      e.preventDefault();
      return;
    }
    e.preventDefault();
    return;
  }

  if (e.key === "ArrowLeft") {
    state.yearIdx = Math.max(0, state.yearIdx - 1);
    render(); e.preventDefault();
  } else if (e.key === "ArrowRight") {
    state.yearIdx = Math.min(site.years.length - 1, state.yearIdx + 1);
    render(); e.preventDefault();
  } else if (e.key === "Enter" || e.key === "ArrowDown") {
    state.siteIdx = Math.min(state.queue.length, state.siteIdx + 1);
    state.yearIdx = 0;
    render(); e.preventDefault();
  } else if (e.key === "ArrowUp") {
    state.siteIdx = Math.max(0, state.siteIdx - 1);
    state.yearIdx = 0;
    render(); e.preventDefault();
  } else if (KEY_TO_LABEL[e.key]) {
    const label = KEY_TO_LABEL[e.key];
    state.labels.set(`${site.site_id}|${year}`, label);
    render();
    await postLabel(site.site_id, year, label);
    // auto-advance focus to next year within site
    if (state.yearIdx < site.years.length - 1) {
      state.yearIdx += 1;
      render();
    }
    e.preventDefault();
  } else if (e.key === "x") {
    state.labels.delete(`${site.site_id}|${year}`);
    render();
    await deleteLabel(site.site_id, year);
    e.preventDefault();
  } else if (e.key === "s") {
    await postFlag(site.site_id, "bad_geocode");
    render();
    e.preventDefault();
  } else if (e.key === "n") {
    noteInput.focus();
    noteInput.select();
    e.preventDefault();
  } else if (e.key === "h") {
    state.showHeatmap = !state.showHeatmap;
    setStatus(state.showHeatmap ? "heatmap on (h to toggle)" : "heatmap off");
    render();
    e.preventDefault();
  } else if (e.key === "o") {
    state.draftOutline = [];
    setStatus("draw outline: click vertices on focused chip · Enter to save · Backspace undo · Esc cancel");
    render();
    e.preventDefault();
  } else if (e.key === "O") {
    if (state.outlines.has(site.site_id)) {
      state.outlines.delete(site.site_id);
      render();
      await deleteOutlineApi(site.site_id);
    } else {
      setStatus("no outline to clear");
    }
    e.preventDefault();
  }
}

init();
