import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";
import { ik3LinkPlanar, clamp } from "./ik.js";

// =============================
// UI elements
// =============================
const simCanvas = document.getElementById("simCanvas");
const drawCanvas = document.getElementById("drawCanvas");
const statusEl = document.getElementById("status");

const btnClear = document.getElementById("btnClear");
const btnHome = document.getElementById("btnHome");

const chkFollow = document.getElementById("chkFollow");
const chkShowTarget = document.getElementById("chkShowTarget");
const chkShowExec = document.getElementById("chkShowExec");

// 2D canvas context (CSSピクセルで描くために dpr を吸収する)
const ctx2d = drawCanvas.getContext("2d");

// =============================
// Mapping: 2D canvas <-> world
// =============================
// モデル内の「キャンバス」(描画面)を想定したワークスペース範囲（world XY）
// ※ arm_pen.xml のキャンバス geom のサイズ/位置に合わせています。
const workspace = {
  // NOTE:
  // This is a *reachable* sub-rectangle of the visual canvas geom.
  // The arm max reach is L1+L2+L3 = 0.46, so keep (x,y) within ~0.455.
  xMin: 0.03,
  xMax: 0.43,
  yMin: -0.15,
  yMax: 0.15,
};

// 2DキャンバスのCSSピクセルサイズ
let drawCssW = 1;
let drawCssH = 1;

function canvasCssToWorld(px, py) {
  const u = px / drawCssW;
  const v = py / drawCssH;
  const x = workspace.xMin + u * (workspace.xMax - workspace.xMin);
  const y = workspace.yMax - v * (workspace.yMax - workspace.yMin); // y反転
  return { x, y };
}

function worldToCanvasCss(x, y) {
  const u = (x - workspace.xMin) / (workspace.xMax - workspace.xMin);
  const v = (workspace.yMax - y) / (workspace.yMax - workspace.yMin);
  return { px: u * drawCssW, py: v * drawCssH };
}

// =============================
// Path data (target & executed)
// =============================
/** @type {Array<Array<{x:number,y:number}>>} */
let targetStrokes = [];
/** @type {Array<Array<{x:number,y:number}>>} */
let executedStrokes = [];

// command queue (world coordinates)
let commandQueue = []; // {x, y, penDown}

// pointer drawing state
let isPointerDown = false;
let currentTargetStroke = null;
let currentStrokeCmds = []; // followWhileDrawing=false のときに溜めて、最後にまとめてキューへ入れる

// options
let followWhileDrawing = true;
let showTarget = true;
let showExecuted = true;

// =============================
// MuJoCo + Three state
// =============================
let mujoco = null;
let model = null;
let data = null;



// =============================
// Debug/Test hooks
// =============================
let paused = false;
function setPaused(v) { paused = !!v; }
function isPaused() { return paused; }

// Expose a promise that resolves when MuJoCo model/data are ready (for tests)
// tests.js can await window.appReady
if (!window.appReady) {
  window.appReady = new Promise((resolve, reject) => {
    window.__resolveAppReady = resolve;
    window.__rejectAppReady = reject;
  });
}

// Pen site lookup (DO NOT assume site index 0)
const PEN_SITE_NAME = "pen_tip_site";
let penSiteId = -1;

function toMjtEnumInt(v) {
  // Some mujoco-js builds expose enums as embind objects, not plain numbers.
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  if (v && typeof v === "object") {
    if (typeof v.value === "number") return v.value;
    if (typeof v.value === "function") {
      const n = v.value();
      if (typeof n === "number") return n;
    }
    if (typeof v.valueOf === "function") {
      const n = v.valueOf();
      if (typeof n === "number") return n;
    }
    if ("__value" in v && typeof v.__value === "number") return v.__value;
  }
  return NaN;
}


function lookupSiteIdByName(name) {
  if (!mujoco || !model) return -1;
  try {
    if (mujoco.mj_name2id && mujoco.mjtObj && mujoco.mjtObj.mjOBJ_SITE !== undefined) {
      const t = toMjtEnumInt(mujoco.mjtObj.mjOBJ_SITE);
      if (!Number.isFinite(t)) throw new Error("mjtObj.mjOBJ_SITE not numeric");
      return mujoco.mj_name2id(model, t, name);
    }
  } catch (e) {}
  // fallback for potential bindings
  try {
    if (model.site) {
      return model.site(name);
    }
  } catch (e) {}
  return -1;
}

function getSiteXpos(siteId) {
  const i = 3 * siteId;
  return {
    x: data.site_xpos[i + 0],
    y: data.site_xpos[i + 1],
    z: data.site_xpos[i + 2],
  };
}

function getPenWorld() {
  if (!model || !data) return { x: 0, y: 0, z: 0 };
  if (penSiteId >= 0) return getSiteXpos(penSiteId);
  // last resort fallback (debug only)
  return { x: data.site_xpos[0], y: data.site_xpos[1], z: data.site_xpos[2] };
}

// IK parameters (XML のリンク長に合わせる)
const L1 = 0.20;
const L2 = 0.20;
const L3 = 0.06;

// 末端の平面内姿勢
// 以前は 0 固定にしていましたが、そうすると「位置的には届くが姿勢制約でIKが解けない」領域が出ます。
// そこでターゲット方向に向ける（phi=atan2(y,x)）ことで到達不能点を大幅に減らします。

// Pen (slide joint) desired positions
const PEN_UP = 0.03;
const PEN_DOWN = -0.012;

// When pen tip z <= this (world), treat as "down" (visual)
// NOTE: this is only used as a backup; main down/up uses the pen slide joint (qpos[3]).
const PEN_CONTACT_Z = 0.003;

// IK reach tolerance for popping commands
// 3mm は厳しすぎてキューが詰まりやすいので、少し緩めます。
const TARGET_EPS = 0.010;
const MOVE_EPS = 0.012;

function wrapToPi(rad) {
  const twoPi = Math.PI * 2;
  let r = rad % twoPi;
  if (r <= -Math.PI) r += twoPi;
  if (r > Math.PI) r -= twoPi;
  return r;
}

function angleDiff(a, b) {
  // smallest signed diff (a-b) in (-pi,pi]
  return wrapToPi(a - b);
}

function pickIKSolution(cmdX, cmdY) {
  // Choose phi to face the target direction.
  const phi = Math.atan2(cmdY, cmdX);

  const qDown = ik3LinkPlanar(cmdX, cmdY, L1, L2, L3, phi, "down");
  const qUp = ik3LinkPlanar(cmdX, cmdY, L1, L2, L3, phi, "up");
  if (!qDown && !qUp) return null;
  if (qDown && !qUp) return qDown;
  if (qUp && !qDown) return qUp;

  // pick the solution closest to current joint angles to reduce flipping/jitter
  const qcur = [data.qpos[0], data.qpos[1], data.qpos[2]];
  const cost = (q) =>
    Math.abs(angleDiff(q[0], qcur[0])) +
    Math.abs(angleDiff(q[1], qcur[1])) +
    Math.abs(angleDiff(q[2], qcur[2]));
  return cost(qDown) <= cost(qUp) ? qDown : qUp;
}

// =============================
// UI wiring
// =============================
chkFollow.addEventListener("change", () => {
  followWhileDrawing = chkFollow.checked;
});
chkShowTarget.addEventListener("change", () => {
  showTarget = chkShowTarget.checked;
});
chkShowExec.addEventListener("change", () => {
  showExecuted = chkShowExec.checked;
});

btnClear.addEventListener("click", () => {
  targetStrokes = [];
  executedStrokes = [];
  commandQueue = [];
  isPointerDown = false;
  currentTargetStroke = null;
  currentStrokeCmds = [];
});

btnHome.addEventListener("click", () => {
  commandQueue = [];
  currentStrokeCmds = [];
  // ホーム姿勢へ（ctrl に入れるだけ）
  if (data) {
    data.ctrl[0] = 0.0;
    data.ctrl[1] = 0.8;
    data.ctrl[2] = -0.8;
    data.ctrl[3] = PEN_UP;
  }
});

// =============================
// Pointer input on draw canvas
// =============================
function getPointerPosCss(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  return { px, py };
}

function pushTargetPoint(px, py) {
  if (!currentTargetStroke) return;

  // 距離が近すぎる点は間引く（CPU/キュー増大対策）
  const pts = currentTargetStroke;
  const last = pts[pts.length - 1];
  const minDist = 2.5; // px
  if (last) {
    const d = Math.hypot(px - last.x, py - last.y);
    if (d < minDist) return;
  }

  const isFirst = pts.length === 0;
  pts.push({ x: px, y: py });

  const { x, y } = canvasCssToWorld(px, py);

  if (isFirst) {
    // まず「ペンを上げたまま」スタート点へ移動 → その場でペンを下ろす。
    // （これを入れないと、最初の点まで移動する途中も線が引かれてしまう）
    const move = { x, y, penDown: false };
    const down = { x, y, penDown: true };
    currentStrokeCmds.push(move, down);
    if (followWhileDrawing) commandQueue.push(move, down);
  } else {
    const down = { x, y, penDown: true };
    currentStrokeCmds.push(down);
    if (followWhileDrawing) commandQueue.push(down);
  }
}

drawCanvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  drawCanvas.setPointerCapture(e.pointerId);

  isPointerDown = true;
  currentStrokeCmds = [];
  currentTargetStroke = [];
  targetStrokes.push(currentTargetStroke);

  const { px, py } = getPointerPosCss(e);
  pushTargetPoint(px, py);
});

drawCanvas.addEventListener("pointermove", (e) => {
  if (!isPointerDown) return;
  e.preventDefault();

  const { px, py } = getPointerPosCss(e);
  pushTargetPoint(px, py);
});

function endStroke(e) {
  if (!isPointerDown) return;
  e.preventDefault();

  isPointerDown = false;

  // 最後の点を penUp コマンドで「ストローク終了」扱いにする
  const lastStroke = currentTargetStroke;
  if (lastStroke && lastStroke.length > 0) {
    const lastPt = lastStroke[lastStroke.length - 1];
    const { x, y } = canvasCssToWorld(lastPt.x, lastPt.y);

    // followWhileDrawing=false の場合は、このタイミングでキューに投入
    if (!followWhileDrawing) {
      for (const cmd of currentStrokeCmds) commandQueue.push(cmd);
    }

    // 仕上げ：ペンを上げる（同じ位置で penDown=false）
    commandQueue.push({ x, y, penDown: false });
  }

  currentTargetStroke = null;
  currentStrokeCmds = [];
}

drawCanvas.addEventListener("pointerup", endStroke);
drawCanvas.addEventListener("pointercancel", endStroke);

// =============================
// Resize handling
// =============================
let renderer = null;
let scene = null;
let camera = null;
let controls = null;

function resizeAll() {
  // 2D draw canvas
  {
    const rect = drawCanvas.getBoundingClientRect();
    drawCssW = Math.max(1, Math.floor(rect.width));
    drawCssH = Math.max(1, Math.floor(rect.height));

    const dpr = window.devicePixelRatio || 1;
    drawCanvas.width = Math.floor(drawCssW * dpr);
    drawCanvas.height = Math.floor(drawCssH * dpr);

    // CSS px で描けるようにスケールを吸収
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Three canvas
  if (renderer && camera) {
    const rect = simCanvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

window.addEventListener("resize", resizeAll);

// =============================
// Simple renderer: MuJoCo geoms -> Three primitives
// =============================
/** @type {Array<{geomIndex:number, mesh:THREE.Mesh}>} */
let geomMeshes = [];

function mujocoGeomToThreeGeometry(geomType, sizeVec3) {
  // MuJoCo geom enum (stable):
  // 0: plane, 2: sphere, 3: capsule, 5: cylinder, 6: box, 7: mesh
  // https://mujoco.readthedocs.io/ (mjtGeom)
  switch (geomType) {
    case 2: {
      const r = sizeVec3[0];
      return new THREE.SphereGeometry(r, 20, 14);
    }
    case 3: {
      const r = sizeVec3[0];
      const half = sizeVec3[1];
      const g = new THREE.CapsuleGeometry(r, 2 * half, 8, 16);
      // three の CapsuleGeometry は Y 軸方向。MuJoCo は Z 軸方向が基本なので回転。
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 5: {
      const r = sizeVec3[0];
      const half = sizeVec3[1];
      const g = new THREE.CylinderGeometry(r, r, 2 * half, 20);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 6: {
      const hx = sizeVec3[0];
      const hy = sizeVec3[1];
      const hz = sizeVec3[2];
      return new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
    }
    default:
      return null;
  }
}

function createThreeSceneFromModel() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0f12);

  // Camera
  camera = new THREE.PerspectiveCamera(45, 1.0, 0.01, 10.0);
  camera.position.set(0.45, -0.55, 0.45);
  camera.lookAt(new THREE.Vector3(0.25, 0, 0.0));

  // Lights
  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.9);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(1.0, -1.0, 1.2);
  scene.add(dir);

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas: simCanvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  // Controls
  controls = new OrbitControls(camera, simCanvas);
  controls.target.set(0.25, 0, 0.02);
  controls.update();

  // Geoms
  geomMeshes = [];
  const ngeom = model.ngeom;

  for (let i = 0; i < ngeom; i++) {
    const geomType = model.geom_type[i];
    const size = model.geom_size.subarray(3 * i, 3 * i + 3);
    const rgba = model.geom_rgba.subarray(4 * i, 4 * i + 4);

    const geometry = mujocoGeomToThreeGeometry(geomType, size);
    if (!geometry) continue;

    const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
    const material = new THREE.MeshStandardMaterial({
      color,
      transparent: rgba[3] < 0.999,
      opacity: rgba[3],
      metalness: 0.0,
      roughness: 0.8,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;

    scene.add(mesh);
    geomMeshes.push({ geomIndex: i, mesh });
  }
}

function updateThreeTransformsFromData() {
  const gxpos = data.geom_xpos; // 3*ngeom
  const gxmat = data.geom_xmat; // 9*ngeom (row-major 3x3)

  for (const { geomIndex, mesh } of geomMeshes) {
    const p = 3 * geomIndex;
    const r = 9 * geomIndex;

    const px = gxpos[p + 0];
    const py = gxpos[p + 1];
    const pz = gxpos[p + 2];

    // MuJoCo: row-major 3x3
    const m00 = gxmat[r + 0],
      m01 = gxmat[r + 1],
      m02 = gxmat[r + 2];
    const m10 = gxmat[r + 3],
      m11 = gxmat[r + 4],
      m12 = gxmat[r + 5];
    const m20 = gxmat[r + 6],
      m21 = gxmat[r + 7],
      m22 = gxmat[r + 8];

    // THREE.Matrix4.set は「引数は行優先 (row-major)」で指定する。
    // 同次変換の並びは以下:
    // [ R00 R01 R02 Tx ]
    // [ R10 R11 R12 Ty ]
    // [ R20 R21 R22 Tz ]
    // [  0   0   0  1 ]
    // ※ 以前ここで Tx,Ty,Tz を最下段に入れてしまっていたため、
    //   位置が反映されず「赤/緑のバーが原点付近でカクカクするだけ」になっていました。
    mesh.matrix.set(m00, m01, m02, px, m10, m11, m12, py, m20, m21, m22, pz, 0, 0, 0, 1);
    mesh.matrixWorldNeedsUpdate = true;
  }
}

// =============================
// Drawing 2D helper
// =============================
function drawStrokes(ctx, strokes) {
  for (const stroke of strokes) {
    if (!stroke || stroke.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) {
      ctx.lineTo(stroke[i].x, stroke[i].y);
    }
    ctx.stroke();
  }
}

function redraw2DOverlay(penWorld, targetCmd) {
  // background
  ctx2d.clearRect(0, 0, drawCssW, drawCssH);
  ctx2d.fillStyle = "#ffffff";
  ctx2d.fillRect(0, 0, drawCssW, drawCssH);

  // workspace frame
  ctx2d.strokeStyle = "rgba(0,0,0,0.25)";
  ctx2d.lineWidth = 1;
  ctx2d.setLineDash([]);
  ctx2d.strokeRect(0.5, 0.5, drawCssW - 1, drawCssH - 1);

  // target strokes (dashed)
  if (showTarget) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.22)";
    ctx2d.lineWidth = 2;
    ctx2d.setLineDash([6, 5]);
    drawStrokes(ctx2d, targetStrokes);
  }

  // executed strokes (solid)
  if (showExecuted) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.92)";
    ctx2d.lineWidth = 2.2;
    ctx2d.setLineDash([]);
    drawStrokes(ctx2d, executedStrokes);
  }

  // current target point
  if (targetCmd) {
    const p = worldToCanvasCss(targetCmd.x, targetCmd.y);
    ctx2d.fillStyle = "rgba(255,0,0,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  // current pen tip
  if (penWorld) {
    const p = worldToCanvasCss(penWorld.x, penWorld.y);
    ctx2d.fillStyle = "rgba(0,120,255,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  // info overlay
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = "rgba(0,0,0,0.55)";
  ctx2d.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  const qlen = commandQueue.length;
  ctx2d.fillText(`queue: ${qlen}`, 10, 18);
}

// =============================
// Simulation loop
// =============================
let lastFrameTime = performance.now();
let accumulator = 0;
let fpsCounter = { frames: 0, last: performance.now(), fps: 0 };

let execPenDown = false;
let currentExecStroke = null;

function recordExecutedPoint(penWorld, penIsDown) {
  // pen down -> start or continue stroke
  if (penIsDown) {
    if (!execPenDown) {
      currentExecStroke = [];
      executedStrokes.push(currentExecStroke);
      execPenDown = true;
    }
    if (currentExecStroke) {
      const { px, py } = worldToCanvasCss(penWorld.x, penWorld.y);
      const last = currentExecStroke[currentExecStroke.length - 1];
      const minDist = 2.0;
      if (!last || Math.hypot(px - last.x, py - last.y) >= minDist) {
        currentExecStroke.push({ x: px, y: py });
      }
    }
  } else {
    execPenDown = false;
    currentExecStroke = null;
  }
}

function stepOnce() {
  // decide current command
  const cmd = commandQueue.length > 0 ? commandQueue[0] : null;

  // desired pen height
  const penDesired = cmd && cmd.penDown ? PEN_DOWN : PEN_UP;

  // IK
  if (cmd) {
    const q = pickIKSolution(cmd.x, cmd.y);
    if (q) {
      data.ctrl[0] = q[0];
      data.ctrl[1] = q[1];
      data.ctrl[2] = q[2];
    } else {
      // unreachable -> drop
      commandQueue.shift();
    }
  } else {
    // hold current
    data.ctrl[0] = data.qpos[0];
    data.ctrl[1] = data.qpos[1];
    data.ctrl[2] = data.qpos[2];
  }
  data.ctrl[3] = penDesired;

  // physics step
  mujoco.mj_step(model, data);

  // pen tip position
  const penWorld = getPenWorld();
  const sx = penWorld.x;
  const sy = penWorld.y;
  const sz = penWorld.z;

  // pop command if reached
  if (cmd) {
    const err = Math.hypot(sx - cmd.x, sy - cmd.y);
    if (cmd.penDown) {
      // drawing points: position tolerance only
      if (err < TARGET_EPS) {
        // allow skipping multiple nearby points (prevents queue congestion)
        do {
          commandQueue.shift();
        } while (
          commandQueue.length > 0 &&
          commandQueue[0].penDown &&
          Math.hypot(sx - commandQueue[0].x, sy - commandQueue[0].y) < TARGET_EPS
        );
      }
    } else {
      // pen-up move/finish: require both "close enough" and "pen lifted"
      if (err < MOVE_EPS && data.qpos[3] > PEN_UP * 0.7) {
        commandQueue.shift();
      }
    }
  }

  // record executed path
  // In this demo, the canvas is non-colliding for stability, so we treat "down" mainly
  // by the slide joint value (qpos[3]). z-threshold is a fallback.
  const penIsDown = data.qpos[3] < (PEN_UP + PEN_DOWN) * 0.5 || sz <= PEN_CONTACT_Z;
  recordExecutedPoint({ x: sx, y: sy, z: sz }, penIsDown);
}

function animate(now) {
  requestAnimationFrame(animate);

  const dt = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  // safety clamp (タブ復帰直後の巨大dt対策)
  accumulator += Math.min(0.05, dt);

  const simDt = model ? model.opt.timestep : 0.002;
  while (model && data && accumulator >= simDt) {
    if (!paused) stepOnce();
    accumulator -= simDt;
  }

  if (model && data) {
    updateThreeTransformsFromData();
    renderer.render(scene, camera);

    const penWorld = getPenWorld();
    const targetCmd = commandQueue.length > 0 ? commandQueue[0] : null;
    redraw2DOverlay(penWorld, targetCmd);

    // status (fps + queue)
    fpsCounter.frames++;
    const t = performance.now();
    if (t - fpsCounter.last > 500) {
      fpsCounter.fps = (fpsCounter.frames * 1000) / (t - fpsCounter.last);
      fpsCounter.frames = 0;
      fpsCounter.last = t;
      statusEl.textContent = `fps ${fpsCounter.fps.toFixed(1)} | queue ${commandQueue.length}`;
    }
  }
}

// =============================
// Boot
// =============================
async function boot() {
  try {
    statusEl.textContent = "loading mujoco-js…";

    // Load MuJoCo WASM module
    mujoco = await load_mujoco();

    statusEl.textContent = "loading model…";

    // Setup virtual FS + load XML
    mujoco.FS.mkdir("/working");
    mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

    const xmlText = await (await fetch("./models/arm_pen.xml")).text();
    mujoco.FS.writeFile("/working/arm_pen.xml", xmlText);

    model = mujoco.MjModel.loadFromXML("/working/arm_pen.xml");
    data = new mujoco.MjData(model);

// lookup pen tip site id
penSiteId = lookupSiteIdByName(PEN_SITE_NAME);
if (penSiteId < 0) {
  console.warn(`[warn] site "${PEN_SITE_NAME}" not found. Falling back to site 0 (debug only).`);
}



    // init pose
    data.qpos[0] = 0.0;
    data.qpos[1] = 0.8;
    data.qpos[2] = -0.8;
    data.qpos[3] = PEN_UP;

    data.ctrl[0] = data.qpos[0];
    data.ctrl[1] = data.qpos[1];
    data.ctrl[2] = data.qpos[2];
    data.ctrl[3] = data.qpos[3];

    mujoco.mj_forward(model, data);

    statusEl.textContent = "init renderer…";

    createThreeSceneFromModel();
    resizeAll();

    statusEl.textContent = "ready";

// expose for tests/debug
window.app = {
  mujoco,
  model,
  data,
  workspace,
  L1, L2, L3,
  PEN_UP, PEN_DOWN,
  PEN_SITE_NAME,
  getPenWorld,
  getSiteXpos,
  canvasCssToWorld,
  worldToCanvasCss,
  ik3LinkPlanar,
  clamp,
  // simulation helpers
  stepOnce,
  setPaused,
  isPaused,
  // access to queues/strokes (read/write)
  get commandQueue() { return commandQueue; },
  set commandQueue(v) { commandQueue = v; },
  get targetStrokes() { return targetStrokes; },
  set targetStrokes(v) { targetStrokes = v; },
  get executedStrokes() { return executedStrokes; },
  set executedStrokes(v) { executedStrokes = v; },
  // low-level mujoco helpers
  mj_forward: () => mujoco.mj_forward(model, data),
  mj_step: () => mujoco.mj_step(model, data),
};
if (window.__resolveAppReady) window.__resolveAppReady(window.app);



    requestAnimationFrame(animate);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "failed to load (see console)";
    if (window.__rejectAppReady) window.__rejectAppReady(err);
  }
}

resizeAll();
boot();
