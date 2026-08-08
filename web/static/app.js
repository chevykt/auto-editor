const $ = (id) => document.getElementById(id);
const show = (id) => { $(id).hidden = false; };
const hide = (id) => { $(id).hidden = true; };

let file = null;
let job = null;
let report = null;
let applied = new Set();

const ts = (s) => {
  const m = Math.floor(s / 60);
  const r = (s % 60).toFixed(2).padStart(5, "0");
  return `${String(m).padStart(2, "0")}:${r}`;
};

// A merged span can carry several reasons. Retake outranks disfluency, which
// outranks silence, so the badge names the most consequential thing in it.
const reasonOf = (cut) =>
  cut.reasons.includes("suspect") ? "suspect"
  : cut.reasons.includes("retake_loser") ? "retake"
  : cut.reasons.includes("disfluency") ? "disfluency"
  : "silence";

const isSuspect = (cut) => cut.reasons.includes("suspect");

const text = (value) => {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
};

// --- start screen --------------------------------------------------------

$("drop").addEventListener("click", () => $("file").click());
$("drop").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop").classList.add("over");
});
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  if (e.dataTransfer.files.length) takeFile(e.dataTransfer.files[0]);
});
$("file").addEventListener("change", (e) => {
  if (e.target.files.length) takeFile(e.target.files[0]);
});
$("script").addEventListener("input", refreshStart);

function takeFile(f) {
  file = f;
  const mb = (f.size / 1048576).toFixed(0);
  $("drop-text").textContent = `${f.name} — ${mb} MB`;
  $("drop").classList.add("loaded");
  refreshStart();
}

function refreshStart() {
  $("analyse").disabled = !(file && $("script").value.trim());
}

function failStart(message) {
  $("start-error").textContent = message;
  show("start-error");
  hide("working");
  show("start");
  $("analyse").disabled = false;
}

/**
 * Upload via XHR rather than fetch, purely for upload progress.
 * fetch cannot report it, and a 124 MB file otherwise leaves the page looking
 * dead for as long as the transfer takes.
 */
function upload(body, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");

    request.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    });

    // A network error - server down, connection dropped - fires onerror, not a
    // response. Without this the click handler died silently and the page just
    // sat there, which is what "nothing happened" looks like.
    request.addEventListener("error", () =>
      reject(new Error("can't reach the server — is `python -m web` still running?")));
    request.addEventListener("abort", () =>
      reject(new Error("upload cancelled")));

    request.addEventListener("load", () => {
      let payload = {};
      try {
        payload = JSON.parse(request.responseText);
      } catch { /* a non-JSON body means the server errored badly */ }

      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.detail || `upload failed (${request.status})`));
    });

    request.send(body);
  });
}

$("analyse").addEventListener("click", async () => {
  $("analyse").disabled = true;
  hide("start-error");

  const body = new FormData();
  body.append("video", file);
  body.append("script", $("script").value);

  hide("start");
  show("working");
  $("stage").textContent = "Uploading";
  $("bar-fill").classList.remove("indeterminate");
  $("bar-fill").style.width = "0%";

  try {
    const created = await upload(body, (fraction) => {
      $("stage").textContent = `Uploading ${Math.round(fraction * 100)}%`;
      $("bar-fill").style.width = `${Math.round(fraction * 100)}%`;
    });
    job = created.id;
  } catch (err) {
    failStart(err.message);
    return;
  }

  $("stage").textContent = "Queued";
  $("bar-fill").style.width = "0%";
  poll();
});

// --- polling -------------------------------------------------------------

const STAGE_TEXT = {
  queued: "Queued",
  probing: "Reading the file",
  transcribing: "Transcribing",
  analysing: "Matching your script",
  rendering: "Rendering",
};

async function poll() {
  let state;
  try {
    const response = await fetch(`/api/jobs/${job}`);
    if (!response.ok) throw new Error(`server said ${response.status}`);
    state = await response.json();
  } catch {
    // Losing the server mid-job used to throw out of the handler and leave the
    // page frozen on whatever it last showed.
    hide("review");
    failStart("lost contact with the server — is `python -m web` still running?");
    return;
  }

  if (state.state === "failed") {
    hide("review");
    failStart(state.error);
    return;
  }

  if (state.state === "ready") {
    report = state.report;
    // Suspects are reported, not applied. They arrive unchecked so a guess
    // about a hidden false start can never silently remove real speech.
    applied = new Set(
      report.cuts.map((c, i) => (isSuspect(c) ? null : i)).filter((i) => i !== null));
    hide("working");
    drawReview();
    show("review");
    return;
  }

  if (state.state === "done") {
    hide("working");
    $("finished-text").textContent =
      `Done — ${ts(report.duration)} cut down to ${ts(outputSeconds())}.`;
    $("result").src = `/api/jobs/${job}/download`;
    $("download").href = `/api/jobs/${job}/download`;
    show("finished");
    return;
  }

  // Whisper reports nothing until its first segment, and loading the model
  // takes ~17s. Showing "Transcribing 0%" against a dead bar for that long
  // reads as a hang - which is the exact thing the progress bar exists to
  // prevent. Name what is actually happening instead.
  const loadingModel = state.state === "transcribing" && state.progress === 0;
  $("stage").textContent = loadingModel
    ? "Loading the speech model…"
    : STAGE_TEXT[state.state] || state.state;

  $("bar-fill").classList.toggle("indeterminate", loadingModel);
  $("bar-fill").style.width =
    loadingModel ? "100%" : `${Math.round(state.progress * 100)}%`;

  setTimeout(poll, 500);
}

// --- review --------------------------------------------------------------

function activeCuts() {
  return report.cuts.filter((_, i) => applied.has(i));
}

function outputSeconds() {
  const removed = activeCuts().reduce((sum, c) => sum + (c.end - c.start), 0);
  return report.duration - removed;
}

function drawReview() {
  $("player").src = `/api/jobs/${job}/video`;
  $("filename").textContent =
    report.video ? report.video.split(/[\\/]/).pop() : "";
  drawTimeline();
  drawCuts();
  drawAdlibs();
  drawLines();
  refreshTotals();
}

function drawTimeline() {
  const bar = $("timeline");
  bar.innerHTML = "";
  let at = 0;

  const push = (cls, width) => {
    if (width <= 0) return;
    const d = document.createElement("div");
    d.className = cls;
    d.style.flex = String(width);
    bar.appendChild(d);
  };

  report.cuts.forEach((cut, i) => {
    push("kept", cut.start - at);
    push(applied.has(i) ? reasonOf(cut) : "kept", cut.end - cut.start);
    at = cut.end;
  });
  push("kept", report.duration - at);
}

function drawCuts() {
  const judged = report.cuts
    .map((cut, index) => ({ cut, index }))
    .filter(({ cut }) => reasonOf(cut) !== "silence" && !isSuspect(cut));

  const suspects = report.cuts
    .map((cut, index) => ({ cut, index }))
    .filter(({ cut }) => isSuspect(cut));
  drawSuspects(suspects);

  const silence = report.cuts.filter((c) => reasonOf(c) === "silence");
  const total = silence.reduce((s, c) => s + (c.end - c.start), 0);
  $("silence-summary").textContent =
    `Silence: ${silence.length} cuts, ${total.toFixed(1)}s total — never a ` +
    `judgment call, so removed without asking.`;

  const host = $("cuts");
  host.innerHTML = "";
  if (!judged.length) {
    host.innerHTML = '<p class="empty">None — nothing needed a judgment call.</p>';
    return;
  }

  judged.forEach(({ cut, index }) => {
    const row = document.createElement("div");
    row.className = "row";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.setAttribute("aria-label", `keep cut at ${ts(cut.start)}`);
    box.addEventListener("change", () => {
      if (box.checked) applied.add(index); else applied.delete(index);
      row.classList.toggle("off", !box.checked);
      drawTimeline();
      refreshTotals();
    });

    const play = document.createElement("button");
    play.className = "play";
    play.textContent = "▶";
    play.title = "Play this moment";
    play.addEventListener("click", () => {
      const player = $("player");
      player.currentTime = Math.max(0, cut.start - 1.5);
      player.play();
    });

    row.innerHTML =
      `<span class="time">${ts(cut.start)}</span>` +
      `<span class="badge ${reasonOf(cut)}">${reasonOf(cut)}</span>` +
      `<span class="detail">${text(cut.detail)}</span>`;
    row.prepend(box);
    row.appendChild(play);
    host.appendChild(row);
  });
}

function drawAdlibs() {
  const host = $("adlibs");
  host.innerHTML = "";
  if (!report.adlibs.length) {
    host.innerHTML = '<p class="empty">None.</p>';
    return;
  }
  report.adlibs.forEach((a) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<span class="time">${ts(a.start)}</span>` +
      `<span class="detail">${text(a.text)}</span>`;
    host.appendChild(row);
  });
}

function drawLines() {
  const host = $("lines");
  host.innerHTML = "";
  report.lines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<span class="time">${line.start === null ? "--:--" : ts(line.start)}</span>` +
      `<span class="status ${line.status}">${line.status}</span>` +
      `<span class="detail">${text(line.matched_text || line.script)}</span>`;
    host.appendChild(row);
  });
}

function refreshTotals() {
  $("duration").textContent = `${ts(report.duration)} → ${ts(outputSeconds())}`;
  $("render").textContent = `Render ${outputSeconds().toFixed(1)}s`;
}

// --- actions -------------------------------------------------------------

$("render").addEventListener("click", async () => {
  $("render").disabled = true;
  hide("review-error");

  try {
    const response = await fetch(`/api/jobs/${job}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cuts: [...applied] }),
    });
    if (!response.ok) throw new Error((await response.json()).detail);
  } catch (err) {
    $("review-error").textContent =
      err.message || "can't reach the server — is `python -m web` still running?";
    show("review-error");
    $("render").disabled = false;
    return;
  }

  hide("review");
  $("stage").textContent = "Rendering";
  $("bar-fill").style.width = "100%";
  show("working");
  poll();
});

$("xml").addEventListener("click", async () => {
  const response = await fetch(`/api/jobs/${job}/xml`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cuts: [...applied] }),
  });
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "timeline.xml";
  link.click();
  URL.revokeObjectURL(link.href);
});

$("again").addEventListener("click", () => location.reload());

/**
 * Suspects: tokens long enough that Whisper may have absorbed a discarded
 * attempt. Rendered unchecked, because the boundary is an estimate and a wrong
 * guess here removes real speech.
 */
function drawSuspects(entries) {
  const host = $("suspects");
  if (!host) return;

  host.innerHTML = "";
  if (!entries.length) {
    host.innerHTML = '<p class="empty">None.</p>';
    return;
  }

  entries.forEach(({ cut, index }) => {
    const row = document.createElement("div");
    row.className = "row off";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = false;
    box.setAttribute("aria-label", `remove suspected false start at ${ts(cut.start)}`);
    box.addEventListener("change", () => {
      if (box.checked) applied.add(index); else applied.delete(index);
      row.classList.toggle("off", !box.checked);
      drawTimeline();
      refreshTotals();
    });

    const play = document.createElement("button");
    play.className = "play";
    play.textContent = "▶";
    play.title = "Play this moment";
    play.addEventListener("click", () => {
      const player = $("player");
      player.currentTime = Math.max(0, cut.start - 1.5);
      player.play();
    });

    row.innerHTML =
      `<span class="time">${ts(cut.start)}</span>` +
      `<span class="badge suspect">suspect</span>` +
      `<span class="detail">${text(cut.detail)}</span>`;
    row.prepend(box);
    row.appendChild(play);
    host.appendChild(row);
  });
}
