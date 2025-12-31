// Self-tests / debug harness for MuJoCo WASM Drawing Arm
// - Runs in browser, triggered from HTML button or URL param (?selftest=basic|full)
// - Requires main.js to expose window.appReady

function $(id) {
  return document.getElementById(id);
}

function nowMs() {
  return performance.now();
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function fmt(n, d = 4) {
  if (!Number.isFinite(n)) return String(n);
  return n.toFixed(d);
}

function line(outEl, s = "") {
  outEl.textContent += s + "\n";
  outEl.scrollTop = outEl.scrollHeight;
}

function section(outEl, title) {
  line(outEl, "");
  line(outEl, "== " + title + " ==");
}

async function runOne(outEl, name, fn) {
  const t0 = nowMs();
  try {
    const info = await fn();
    const dt = nowMs() - t0;
    line(outEl, `✅ ${name}  (${dt.toFixed(1)} ms)`);
    if (info) {
      for (const l of String(info).split("\n")) line(outEl, "   " + l);
    }
    return { name, ok: true, ms: dt };
  } catch (e) {
    const dt = nowMs() - t0;
    line(outEl, `❌ ${name}  (${dt.toFixed(1)} ms)`);
    line(outEl, "   " + (e?.message || String(e)));
    return { name, ok: false, ms: dt, err: e };
  }
}

function getQuerySelftestMode() {
  const sp = new URLSearchParams(location.search);
  const v = sp.get("selftest");
  if (!v) return null;
  const s = v.toLowerCase();
  if (s === "1" || s === "true" || s === "basic") return "basic";
  if (s === "full" || s === "all") return "full";
  return "basic";
}

function openDrawer() {
  const drawer = $("testDrawer");
  drawer.classList.remove("hidden");
  drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  const drawer = $("testDrawer");
  drawer.classList.add("hidden");
  drawer.setAttribute("aria-hidden", "true");
}

function clearLog() {
  const outEl = $("testOutput");
  outEl.textContent = "";
}

function copyLog() {
  const outEl = $("testOutput");
  const text = outEl.textContent || "";
  navigator.clipboard?.writeText(text).catch(() => {});
}

function rectInfo(canvas) {
  const r = canvas.getBoundingClientRect();
  return { w: Math.max(1, Math.floor(r.width)), h: Math.max(1, Math.floor(r.height)) };
}

function calcRoundTrip(app, px, py, drawCssW, drawCssH) {
  // Use the same CSS px convention as main.js expects.
  const w = app.canvasCssToWorld(px, py);
  const c = app.worldToCanvasCss(w.x, w.y);
  const dx = c.px - px;
  const dy = c.py - py;
  // normalized error (0..1)
  const ndx = dx / drawCssW;
  const ndy = dy / drawCssH;
  return { w, c, dx, dy, ndx, ndy };
}

function fk3Link(q1, q2, q3, L1, L2, L3) {
  const a1 = q1;
  const a2 = q1 + q2;
  const a3 = q1 + q2 + q3;
  const x = L1 * Math.cos(a1) + L2 * Math.cos(a2) + L3 * Math.cos(a3);
  const y = L1 * Math.sin(a1) + L2 * Math.sin(a2) + L3 * Math.sin(a3);
  return { x, y, phi: a3 };
}

function pickBestIK(app, x, y) {
  // mirror main.js behavior: phi faces target direction, try elbow up/down and pick smaller FK error
  const phi = Math.atan2(y, x);
  const qDown = app.ik3LinkPlanar(x, y, app.L1, app.L2, app.L3, phi, "down");
  const qUp = app.ik3LinkPlanar(x, y, app.L1, app.L2, app.L3, phi, "up");
  const cand = [];
  if (qDown) cand.push(qDown);
  if (qUp) cand.push(qUp);
  if (cand.length === 0) return null;
  let best = cand[0];
  let bestErr = Infinity;
  for (const q of cand) {
    const fk = fk3Link(q[0], q[1], q[2], app.L1, app.L2, app.L3);
    const err = Math.hypot(fk.x - x, fk.y - y);
    if (err < bestErr) {
      bestErr = err;
      best = q;
    }
  }
  return best;
}

async function runBasicTests(app, outEl) {
  section(outEl, "Environment");
  await runOne(outEl, "appReady / model loaded", async () => {
    if (!app || !app.model || !app.data) throw new Error("app/model/data not ready");
    return `nu=${app.model.nu ?? "?"}, nq=${app.model.nq ?? "?"}, nsite=${app.model.nsite ?? "?"}`;
  });

  await runOne(outEl, "Pen site lookup (by name)", async () => {
    const mj = app.mujoco;
    const model = app.model;
    const data = app.data;
    const name = app.PEN_SITE_NAME || "pen_tip_site";

    if (!mj.mj_name2id || !mj.mjtObj) {
      return "mj_name2id not available in this binding; skip strict check";
    }
    const sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name);
    if (sid < 0) throw new Error(`site "${name}" not found in model (check XML site name)`);
    const i = 3 * sid;
    const p = { x: data.site_xpos[i + 0], y: data.site_xpos[i + 1], z: data.site_xpos[i + 2] };
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
      throw new Error("site_xpos contains non-finite values");
    }
    const pApp = app.getPenWorld();
    const d = Math.hypot(p.x - pApp.x, p.y - pApp.y, p.z - pApp.z);
    return `siteId=${sid}, penWorld=${fmt(p.x)},${fmt(p.y)},${fmt(p.z)}  | app.getPenWorld diff=${fmt(d, 6)}`;
  });

  section(outEl, "Canvas sizing / coordinate transforms");

  await runOne(outEl, "Canvas CSS size vs internal size (drawCanvas)", async () => {
    const c = $("drawCanvas");
    const r = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const expectW = Math.floor(Math.max(1, Math.floor(r.width)) * dpr);
    const expectH = Math.floor(Math.max(1, Math.floor(r.height)) * dpr);
    const okW = Math.abs(c.width - expectW) <= 2;
    const okH = Math.abs(c.height - expectH) <= 2;
    if (!okW || !okH) {
      throw new Error(`canvas.width/height mismatch: actual ${c.width}x${c.height}, expected ~${expectW}x${expectH} (DPR=${dpr})`);
    }
    return `rect=${Math.floor(r.width)}x${Math.floor(r.height)} css, canvas=${c.width}x${c.height}, dpr=${dpr}`;
  });

  await runOne(outEl, "Round-trip: canvas→world→canvas (5 points)", async () => {
    const c = $("drawCanvas");
    const { w: drawCssW, h: drawCssH } = rectInfo(c);
    let maxAbs = 0;
    const samples = [];
    for (let k = 0; k < 5; k++) {
      const px = Math.random() * (drawCssW - 1);
      const py = Math.random() * (drawCssH - 1);
      const rt = calcRoundTrip(app, px, py, drawCssW, drawCssH);
      maxAbs = Math.max(maxAbs, Math.abs(rt.dx), Math.abs(rt.dy));
      samples.push(`p=(${fmt(px,1)},${fmt(py,1)}) -> world=(${fmt(rt.w.x)},${fmt(rt.w.y)}) -> p2=(${fmt(rt.c.px,1)},${fmt(rt.c.py,1)})  d=(${fmt(rt.dx,3)},${fmt(rt.dy,3)})`);
    }
    if (maxAbs > 1e-3) {
      throw new Error(`round-trip error too large: max |d| = ${maxAbs}`);
    }
    return samples.join("\n");
  });

  section(outEl, "IK sanity (no MuJoCo control involved)");

  await runOne(outEl, "IK→FK error (20 random reachable points)", async () => {
    const ws = app.workspace;
    const reach = app.L1 + app.L2 + app.L3 - 1e-3;
    let maxErr = 0;
    let tries = 0;
    for (let k = 0; k < 20; k++) {
      // sample inside workspace but ensure reachable by radius
      let x, y;
      for (;;) {
        tries++;
        x = ws.xMin + Math.random() * (ws.xMax - ws.xMin);
        y = ws.yMin + Math.random() * (ws.yMax - ws.yMin);
        if (Math.hypot(x, y) <= reach) break;
        if (tries > 2000) throw new Error("failed to sample reachable points (workspace may be outside reach)");
      }
      const q = pickBestIK(app, x, y);
      if (!q) throw new Error(`IK returned null at (${x},${y})`);
      const fk = fk3Link(q[0], q[1], q[2], app.L1, app.L2, app.L3);
      const err = Math.hypot(fk.x - x, fk.y - y);
      maxErr = Math.max(maxErr, err);
    }
    if (maxErr > 5e-3) {
      throw new Error(`IK→FK error too large (max=${maxErr})`);
    }
    return `max IK→FK position error = ${fmt(maxErr, 6)}`;
  });

  section(outEl, "Quick diagnostic snapshots");

  await runOne(outEl, "Pen world position snapshot", async () => {
    const p = app.getPenWorld();
    const c = app.worldToCanvasCss(p.x, p.y);
    return `penWorld=(${fmt(p.x)},${fmt(p.y)},${fmt(p.z)})  penCanvas=(${fmt(c.px,1)},${fmt(c.py,1)})`;
  });

  await runOne(outEl, "Target/pen axis sanity (vary q1)", async () => {
    // This test temporarily writes qpos and runs mj_forward, then restores.
    const mj = app.mujoco;
    const qpos0 = app.data.qpos.slice(); // typed array -> copy
    const ctrl0 = app.data.ctrl.slice();

    const setPose = (q1, q2, q3, pen) => {
      app.data.qpos[0] = q1;
      app.data.qpos[1] = q2;
      app.data.qpos[2] = q3;
      app.data.qpos[3] = pen;
      mj.mj_forward(app.model, app.data);
      return app.getPenWorld();
    };

    const pA = setPose(0, 0.7, -0.7, app.PEN_UP);
    const pB = setPose(Math.PI / 2, 0.7, -0.7, app.PEN_UP);

    // restore
    for (let i = 0; i < qpos0.length; i++) app.data.qpos[i] = qpos0[i];
    for (let i = 0; i < ctrl0.length; i++) app.data.ctrl[i] = ctrl0[i];
    mj.mj_forward(app.model, app.data);

    const d = Math.hypot(pB.x - pA.x, pB.y - pA.y);
    return `p(q1=0)=${fmt(pA.x)},${fmt(pA.y)}  p(q1=pi/2)=${fmt(pB.x)},${fmt(pB.y)}  planarMoveDist=${fmt(d)}`;
  });
}

async function runFullTests(app, outEl) {
  // Full tests are more invasive but restore state afterwards.
  section(outEl, "Full tests (state will be restored)");

  const mj = app.mujoco;
  const model = app.model;
  const data = app.data;

  const backup = {
    paused: app.isPaused?.() ?? false,
    qpos: data.qpos.slice(),
    qvel: data.qvel ? data.qvel.slice() : null,
    ctrl: data.ctrl ? data.ctrl.slice() : null,
    queue: (app.commandQueue || []).slice(),
    targetStrokes: JSON.parse(JSON.stringify(app.targetStrokes || [])),
    executedStrokes: JSON.parse(JSON.stringify(app.executedStrokes || [])),
  };

  const restore = () => {
    try {
      app.setPaused?.(backup.paused);
      for (let i = 0; i < backup.qpos.length; i++) data.qpos[i] = backup.qpos[i];
      if (backup.qvel && data.qvel) for (let i = 0; i < backup.qvel.length; i++) data.qvel[i] = backup.qvel[i];
      if (backup.ctrl && data.ctrl) for (let i = 0; i < backup.ctrl.length; i++) data.ctrl[i] = backup.ctrl[i];
      app.commandQueue = backup.queue;
      app.targetStrokes = backup.targetStrokes;
      app.executedStrokes = backup.executedStrokes;
      mj.mj_forward(model, data);
    } catch (e) {
      console.warn("restore failed", e);
    }
  };

  app.setPaused?.(true);

  try {
    await runOne(outEl, "Actuator mapping sanity (ctrl[i] affects some qpos)", async () => {
      const nu = model.nu ?? 0;
      if (!nu || !data.ctrl) return "no actuators/ctrl available; skip";

      const lines = [];
      // settle
      mj.mj_forward(model, data);

      // Try each actuator: nudge ctrl and step for a short time; see which qpos moved most.
      for (let ai = 0; ai < Math.min(nu, 8); ai++) {
        const q0 = data.qpos.slice();
        const c0 = data.ctrl.slice();

        // heuristic: ctrl=target position for position actuators
        data.ctrl[ai] = (data.ctrl[ai] ?? 0) + 0.25;

        for (let k = 0; k < 80; k++) mj.mj_step(model, data);

        // find max qpos change index
        let bestJ = -1;
        let bestD = 0;
        for (let j = 0; j < Math.min(data.qpos.length, 12); j++) {
          const d = Math.abs(data.qpos[j] - q0[j]);
          if (d > bestD) { bestD = d; bestJ = j; }
        }
        lines.push(`act[${ai}] -> max Δqpos at j=${bestJ}, Δ=${fmt(bestD,4)}`);

        // restore for next actuator
        for (let j = 0; j < q0.length; j++) data.qpos[j] = q0[j];
        for (let j = 0; j < c0.length; j++) data.ctrl[j] = c0[j];
        mj.mj_forward(model, data);
      }

      // If every actuator causes ~0 movement, ctrl wiring is likely broken.
      const anyMove = lines.some((s) => {
        const m = s.match(/Δ=([0-9\.]+)/);
        return m && parseFloat(m[1]) > 1e-3;
      });
      if (!anyMove) throw new Error("No actuator seemed to move qpos. ctrl mapping or gains may be wrong.");
      return lines.join("\n");
    });

    await runOne(outEl, "Mini path follow (square, 2s sim) - queue should drain", async () => {
      // Build a tiny square in reachable region
      const ws = app.workspace;
      const x0 = (ws.xMin + ws.xMax) * 0.5;
      const y0 = 0.0;
      const s = Math.min(ws.xMax - ws.xMin, ws.yMax - ws.yMin) * 0.25;

      const pts = [
        { x: x0 - s, y: y0 - s },
        { x: x0 + s, y: y0 - s },
        { x: x0 + s, y: y0 + s },
        { x: x0 - s, y: y0 + s },
        { x: x0 - s, y: y0 - s },
      ];

      // Replace queue with: move pen up to start, pen down along points
      const q = [];
      q.push({ x: pts[0].x, y: pts[0].y, penDown: false });
      q.push({ x: pts[0].x, y: pts[0].y, penDown: true });
      for (let i = 1; i < pts.length; i++) q.push({ x: pts[i].x, y: pts[i].y, penDown: true });
      q.push({ x: pts[pts.length - 1].x, y: pts[pts.length - 1].y, penDown: false });

      app.commandQueue = q;
      app.targetStrokes = [];
      app.executedStrokes = [];

      // Run stepping using app.stepOnce to exercise the same logic as real-time loop
      const steps = Math.floor(2.0 / (model.opt?.timestep ?? 0.002));
      for (let k = 0; k < steps; k++) {
        app.stepOnce();
        // yield occasionally to keep UI alive
        if (k % 200 === 0) await sleep(0);
      }

      const qlen = (app.commandQueue || []).length;
      const p = app.getPenWorld();
      const err = Math.hypot(p.x - pts[pts.length - 1].x, p.y - pts[pts.length - 1].y);
      return `queueRemaining=${qlen}, finalErr≈${fmt(err,4)}  (executedStrokes=${(app.executedStrokes||[]).length})`;
    });
  } finally {
    restore();
  }
}

async function main() {
  const outEl = $("testOutput");
  const btnOpen = $("btnRunTests");
  const btnClose = $("btnCloseTests");
  const btnBasic = $("btnRunBasicTests");
  const btnFull = $("btnRunFullTests");
  const btnCopy = $("btnCopyTestLog");

  btnOpen?.addEventListener("click", () => {
    openDrawer();
  });
  btnClose?.addEventListener("click", () => closeDrawer());
  btnCopy?.addEventListener("click", () => copyLog());

  clearLog();
  line(outEl, "Waiting for appReady…");

  let app = null;
  try {
    app = await window.appReady;
  } catch (e) {
    line(outEl, "appReady failed: " + (e?.message || String(e)));
    return;
  }

  clearLog();
  line(outEl, "appReady: OK");
  line(outEl, "Tip: open the drawer and run tests.");

  btnBasic?.addEventListener("click", async () => {
    openDrawer();
    clearLog();
    app.setPaused?.(true);
    await runBasicTests(app, outEl);
    line(outEl, "");
    line(outEl, "(basic tests finished)");
    app.setPaused?.(false);
  });

  btnFull?.addEventListener("click", async () => {
    openDrawer();
    clearLog();
    await runBasicTests(app, outEl);
    await runFullTests(app, outEl);
    line(outEl, "");
    line(outEl, "(full tests finished; state restored)");
  });

  // Auto-run from URL
  const mode = getQuerySelftestMode();
  if (mode) {
    openDrawer();
    clearLog();
    line(outEl, `Auto selftest: ${mode}`);
    await runBasicTests(app, outEl);
    if (mode === "full") await runFullTests(app, outEl);
    line(outEl, "");
    line(outEl, "(auto selftest finished)");
  }
}

main().catch((e) => console.error(e));
