/* BarBeCue dashboard client: polls /api/state, draws numbers, beeps reps. */
(function () {
  "use strict";

  var state = null;
  var lastReps = 0;
  var lastBad = 0;
  var audioCtx = null;

  function el(id) { return document.getElementById(id); }

  function beep(freq, dur) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      var osc = audioCtx.createOscillator();
      var gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + dur + 0.05);
    } catch (e) { /* audio unavailable: stay visual */ }
  }

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json(); });
  }

  function fmtStatus(s) {
    var head = s.exercise + " · " + s.profile + " · " + s.fps + " fps";
    if (s.status === "no_person") { return head + " — no person detected"; }
    if (s.status === "step_back") { return head + " — step back, full body visible"; }
    if (s.status === "bad_view") { return head + " — turn sideways to the camera"; }
    if (s.status === "paused") { return head + " — paused, no person detected"; }
    if (s.status === "calibrating") { return head + " — calibrating…"; }
    return head;
  }

  function render(s) {
    state = s;
    el("sessionline").textContent = fmtStatus(s);

    var tally = "REPS " + s.reps;
    var sub = s.stage;
    if (s.bad_reps > 0) {
      sub += " · " + s.bad_reps + " bad" + (s.bad_reason ? " (" + s.bad_reason + ")" : "");
    }
    if (s.scenario) { sub += " · " + s.scenario; }
    el("tally").textContent = tally;
    if (s.bad_reps > 0) {
      var bad = document.createElement("span");
      bad.className = "bad";
      bad.textContent = "  BAD " + s.bad_reps;
      el("tally").appendChild(bad);
    }
    el("tallysub").textContent = sub;

    var cues = el("cues");
    cues.textContent = "";
    if (s.status === "ok" && s.msgs.length === 0) {
      var li = document.createElement("li");
      li.className = "good";
      li.textContent = "Good form";
      cues.appendChild(li);
    } else {
      s.msgs.slice(0, 2).forEach(function (m) {
        var li = document.createElement("li");
        li.textContent = m;
        cues.appendChild(li);
      });
    }

    var prof = el("profile");
    if (prof.options.length !== s.profiles.length) {
      prof.textContent = "";
      s.profiles.forEach(function (p) {
        var o = document.createElement("option");
        o.value = p;
        o.textContent = p;
        prof.appendChild(o);
      });
    }
    prof.value = s.profile;

    document.querySelectorAll("#sidebar .ex").forEach(function (b) {
      b.classList.toggle("active", b.dataset.exercise === s.exercise);
    });
  }

  function poll() {
    fetch("/api/state").then(function (r) { return r.json(); }).then(function (s) {
      if (state && s.reps > state.reps) { beep(660, 0.12); }
      if (state && s.bad_reps > state.bad_reps) { beep(220, 0.2); }
      lastReps = s.reps;
      lastBad = s.bad_reps;
      render(s);
    }).catch(function () { /* server restarting: keep last frame */ });
  }

  function buildSidebar(cats) {
    var bar = el("sidebar");
    bar.textContent = "";
    Object.keys(cats).forEach(function (cat) {
      var div = document.createElement("div");
      div.className = "cat";
      var h = document.createElement("h3");
      h.className = "cat-name";
      h.textContent = cat;
      div.appendChild(h);
      cats[cat].forEach(function (name) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "ex";
        b.dataset.exercise = name;
        b.textContent = name;
        b.addEventListener("click", function () {
          post("/api/select", { exercise: name }).then(render);
        });
        div.appendChild(b);
      });
      bar.appendChild(div);
    });
  }

  function refreshLog() {
    fetch("/api/log").then(function (r) { return r.json(); }).then(function (d) {
      var body = el("logbody");
      body.textContent = "";
      d.sets.forEach(function (row) {
        var tr = document.createElement("tr");
        [row.at, row.exercise, row.reps, row.bad_reps,
         typeof row.hold === "number" ? row.hold.toFixed(1) + "s" : row.hold
        ].forEach(function (v) {
          var td = document.createElement("td");
          td.textContent = v;
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      el("logempty").style.display = d.sets.length ? "none" : "";
    });
  }

  el("profile").addEventListener("change", function (e) {
    post("/api/profile", { profile: e.target.value }).then(render);
  });
  el("btn-reset").addEventListener("click", function () {
    post("/api/reset").then(render);
  });
  el("btn-calibrate").addEventListener("click", function () {
    post("/api/calibrate");
  });
  el("btn-log").addEventListener("click", function () {
    post("/api/log_set").then(function (d) {
      refreshLog();
      poll();
    });
  });

  fetch("/api/exercises").then(function (r) { return r.json(); }).then(function (d) {
    buildSidebar(d.categories);
    poll();
    refreshLog();
    setInterval(poll, 300);
  });
})();
