import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";
import { ik3LinkPlanar, clamp, wrapToPi } from "./ik.js";

/**
 * MuJoCo WASM Drawing Arm (Kinematic only)
 *
 * - No dynamics (mj_step) is used.
 * - We directly update qpos towards IK targets and call mj_forward.
 * - 2D: user draws on the right canvas.
 * - 3D: the arm follows the path and drops tiny "ink" spheres on the canvas surface.
 */

// =============================
// DOM
// =============================
const simCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById("simCanvas"));
const drawCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById("drawCanvas"));
const statusEl = document.getElementById("status");

const btnClear = document.getElementById("btnClear");
const btnHome = document.getElementById("btnHome");
const btnDiag = document.getElementById("btnDiag");

const chkFollow = /** @type {HTMLInputElement} */ (document.getElementById("chkFollow"));
const chkShowTarget = /** @type {HTMLInputElement} */ (document.getElementById("chkShowTarget"));
const chkShowExec = /** @type {HTMLInputElement} */ (document.getElementById("chkShowExec"));
const chkShowInk3D = /** @type {HTMLInputElement} */ (document.getElementById("chkShowInk3D"));

// Diagnostics panel
const diagPanel = document.getElementById("diagPanel");
const btnDiagClose = document.getElementById("btnDiagClose");
const btnTestBasic = document.getElementById("btnTestBasic");
const btnTestInk = document.getElementById("btnTestInk");
const btnTestPath = document.getElementById("btnTestPath");
const btnClearLog = document.getElementById("btnClearLog");
const btnCopyLog = document.getElementById("btnCopyLog");
const diagLog = document.getElementById("diagLog");

const ctx2d = drawCanvas.getContext("2d");

// =============================
// Workspace mapping (2D canvas <-> world XY)
// =============================
// この範囲は models/arm_pen.xml のキャンバス(geom)内の、アームが届く領域に合わせています。
const workspace = {
  xMin: 0.03,
  xMax: 0.43,
  yMin: -0.15,
  yMax: 0.15,
};

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
// Target / executed strokes
// =============================
/** @type {Array<Array<{x:number,y:number}>>} */
let targetStrokes = [];
/** @type {Array<Array<{x:number,y:number}>>} */
let executedStrokes = [];

/** @type {Array<{x:number,y:number,penDown:boolean}>} */
let commandQueue = [];

let isPointerDown = false;
/** @type {Array<{x:number,y:number}> | null} */
let currentTargetStroke = null;
/** @type {Array<{x:number,y:number,penDown:boolean}>} */
let currentStrokeCmds = [];

/** @type {{x:number,y:number} | null} */
let lastWorldInStroke = null;

let followWhileDrawing = true;
let showTarget = true;
let showExecuted = true;
let showInk3D = true;

// =============================
// Diagnostics helpers
// =============================
function setDiagVisible(v) {
  if (!diagPanel) return;
  diagPanel.classList.toggle("hidden", !v);
  diagPanel.setAttribute("aria-hidden", v ? "false" : "true");
}

function diagClear() {
  if (diagLog) diagLog.textContent = "";
}

function diagLine(s) {
  const line = String(s ?? "");
  if (diagLog) {
    diagLog.textContent += (diagLog.textContent ? "\n" : "") + line;
    diagLog.scrollTop = diagLog.scrollHeight;
  }
  console.log("[diag]", line);
}

async function diagCopyToClipboard() {
  try {
    const text = diagLog ? diagLog.textContent : "";
    await navigator.clipboard.writeText(text);
    diagLine("(copied to clipboard)");
  } catch (e) {
    diagLine(`(copy failed: ${e})`);
  }
}

// =============================
// MuJoCo state
// =============================
let mujoco = null;
let model = null;
let data = null;

/** @type {Array<{lo:number,hi:number}>} */
let hingeRanges = [];

const PEN_SITE_NAME = "pen_tip_site";
let penSiteId = -1;

function toMjtEnumInt(v) {
  // Some mujoco-js builds expose enums as embind objects, not plain numbers.
  if (typeof v === "number") return v;
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
  } catch (e) {
    // ignore, fallback
  }
  return -1;
}

function lookupGeomIdByName(name) {
  if (!mujoco || !model) return -1;
  try {
    if (mujoco.mj_name2id && mujoco.mjtObj && mujoco.mjtObj.mjOBJ_GEOM !== undefined) {
      const t = toMjtEnumInt(mujoco.mjtObj.mjOBJ_GEOM);
      if (!Number.isFinite(t)) throw new Error("mjtObj.mjOBJ_GEOM not numeric");
      return mujoco.mj_name2id(model, t, name);
    }
  } catch (e) {
    // ignore
  }
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
  if (!data) return { x: 0, y: 0, z: 0 };
  if (penSiteId >= 0) return getSiteXpos(penSiteId);
  // If site lookup fails but there is exactly 1 site, this fallback is still correct.
  return { x: data.site_xpos[0], y: data.site_xpos[1], z: data.site_xpos[2] };
}

function computeCanvasPlaneFromModel() {
  if (!model || !data) return;
  if (canvasGeomId < 0) {
    canvasGeomId = lookupGeomIdByName(CANVAS_GEOM_NAME);
  }
  if (canvasGeomId < 0) {
    canvasPlane = null;
    return;
  }

  const gi = canvasGeomId;
  const p = 3 * gi;
  const r = 9 * gi;
  const cx = data.geom_xpos[p + 0];
  const cy = data.geom_xpos[p + 1];
  const cz = data.geom_xpos[p + 2];

  // local +Z axis in world = third column of row-major rotation matrix
  const m02 = data.geom_xmat[r + 2];
  const m12 = data.geom_xmat[r + 5];
  const m22 = data.geom_xmat[r + 8];
  let nx = m02, ny = m12, nz = m22;
  const nlen = Math.hypot(nx, ny, nz);
  if (!(nlen > 1e-12)) {
    nx = 0;
    ny = 0;
    nz = 1;
  } else {
    nx /= nlen;
    ny /= nlen;
    nz /= nlen;
  }

  // for box geoms, geom_size stores half-sizes in local coordinates
  const hz = model.geom_size[p + 2] ?? 0.0;

  const p0 = new THREE.Vector3(cx + nx * hz, cy + ny * hz, cz + nz * hz);
  const n = new THREE.Vector3(nx, ny, nz);
  canvasPlane = { p0, n };
}

// =============================
// IK / kinematic control params
// =============================
const L1 = 0.20;
const L2 = 0.20;
const L3 = 0.06;
const MAX_REACH = L1 + L2 + L3 - 0.005;

const PEN_UP = 0.03;
const PEN_DOWN = -0.012;

// Resampling step (world meters) to make path tracking smooth even when input points are sparse.
const PATH_STEP = 0.006;

// When using a resampled path, keep the reach threshold smaller than the step to avoid skipping many points.
const TARGET_EPS = 0.0045;
const MOVE_EPS = 0.008;

const JOINT_MAX_RATE = 7.0; // rad/s (kinematic slew)
const PEN_MAX_RATE = 0.30; // m/s

function angleDiff(a, b) {
  // smallest signed diff (a-b) in (-pi,pi]
  return wrapToPi(a - b);
}

function clampToReach(x, y) {
  const r = Math.hypot(x, y);
  if (r < 1e-9) return { x, y };
  if (r <= MAX_REACH) return { x, y };
  const s = MAX_REACH / r;
  return { x: x * s, y: y * s };
}

function pickIKSolution(x, y) {
  // Choose end-effector orientation to face the target direction.
  const phi = Math.atan2(y, x);

  const qDown = ik3LinkPlanar(x, y, L1, L2, L3, phi, "down");
  const qUp = ik3LinkPlanar(x, y, L1, L2, L3, phi, "up");
  if (!qDown && !qUp) return null;
  if (qDown && !qUp) return qDown;
  if (qUp && !qDown) return qUp;

  // pick the one closest to current pose to avoid elbow flipping
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
chkShowInk3D.addEventListener("change", () => {
  showInk3D = chkShowInk3D.checked;
  if (inkMesh) inkMesh.visible = showInk3D;
});

btnClear.addEventListener("click", () => {
  targetStrokes = [];
  executedStrokes = [];
  commandQueue = [];
  isPointerDown = false;
  currentTargetStroke = null;
  currentStrokeCmds = [];
  lastWorldInStroke = null;
  clearInk3D();
});

btnHome.addEventListener("click", () => {
  commandQueue = [];
  currentStrokeCmds = [];
  // Home pose
  if (data && mujoco && model) {
    data.qpos[0] = 0.0;
    data.qpos[1] = 0.8;
    data.qpos[2] = -0.8;
    data.qpos[3] = PEN_UP;
    mujoco.mj_forward(model, data);

    // Compute the canvas plane for 3D ink placement.
    computeCanvasPlaneFromModel();
  }
});

// Diagnostics panel UI
btnDiag?.addEventListener("click", () => {
  const isHidden = diagPanel?.classList.contains("hidden");
  setDiagVisible(isHidden);
});
btnDiagClose?.addEventListener("click", () => setDiagVisible(false));
btnClearLog?.addEventListener("click", () => diagClear());
btnCopyLog?.addEventListener("click", () => diagCopyToClipboard());
btnTestBasic?.addEventListener("click", () => runSelfTestBasic());
btnTestInk?.addEventListener("click", () => addTestInk());
btnTestPath?.addEventListener("click", () => runSelfTestPath());

// =============================
// Pointer input on draw canvas
// =============================
function getPointerPosCss(e) {
  const rect = drawCanvas.getBoundingClientRect();
  return { px: e.clientX - rect.left, py: e.clientY - rect.top };
}

function resampleSegment(a, b, step) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy);
  if (!(dist > 1e-9)) return [];
  const n = Math.max(0, Math.floor(dist / step));
  const pts = [];
  for (let i = 1; i <= n; i++) {
    const t = (i * step) / dist;
    pts.push({ x: a.x + dx * t, y: a.y + dy * t });
  }
  // always include end point
  pts.push({ x: b.x, y: b.y });
  return pts;
}

function pushTargetPoint(px, py) {
  if (!currentTargetStroke) return;

  // decimate
  const pts = currentTargetStroke;
  const last = pts[pts.length - 1];
  const minDistPx = 2.5;
  if (last) {
    const d = Math.hypot(px - last.x, py - last.y);
    if (d < minDistPx) return;
  }

  const isFirst = pts.length === 0;
  pts.push({ x: px, y: py });

  let { x, y } = canvasCssToWorld(px, py);
  ({ x, y } = clampToReach(x, y));

  if (isFirst) {
    // pen-up move to start, then pen-down at start
    const move = { x, y, penDown: false };
    const down = { x, y, penDown: true };
    currentStrokeCmds.push(move, down);
    if (followWhileDrawing) commandQueue.push(move, down);
    lastWorldInStroke = { x, y };
    return;
  }

  // For smooth tracking, resample the segment at nearly constant world spacing.
  const prev = lastWorldInStroke ?? { x, y };
  const segPts = resampleSegment(prev, { x, y }, PATH_STEP);
  for (const p of segPts) {
    const down = { x: p.x, y: p.y, penDown: true };
    currentStrokeCmds.push(down);
    if (followWhileDrawing) commandQueue.push(down);
  }
  lastWorldInStroke = { x, y };
}

drawCanvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  drawCanvas.setPointerCapture(e.pointerId);

  isPointerDown = true;
  currentStrokeCmds = [];
  currentTargetStroke = [];
  targetStrokes.push(currentTargetStroke);
  lastWorldInStroke = null;

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

  const lastStroke = currentTargetStroke;
  if (lastStroke && lastStroke.length > 0) {
    const lastPt = lastStroke[lastStroke.length - 1];
    let { x, y } = canvasCssToWorld(lastPt.x, lastPt.y);
    ({ x, y } = clampToReach(x, y));

    if (!followWhileDrawing) {
      for (const cmd of currentStrokeCmds) commandQueue.push(cmd);
    }

    // lift the pen at the end
    commandQueue.push({ x, y, penDown: false });
  }

  currentTargetStroke = null;
  currentStrokeCmds = [];
  lastWorldInStroke = null;
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
// Three.js rendering (MuJoCo geoms + 3D ink)
// =============================
/** @type {Array<{geomIndex:number, mesh:THREE.Mesh}>} */
let geomMeshes = [];

function mujocoGeomToThreeGeometry(geomType, sizeVec3) {
  // MuJoCo mjtGeom (common subset)
  // 2: sphere, 3: capsule, 5: cylinder, 6: box
  switch (geomType) {
    case 2: {
      const r = sizeVec3[0];
      return new THREE.SphereGeometry(r, 20, 14);
    }
    case 3: {
      const r = sizeVec3[0];
      const half = sizeVec3[1];
      const g = new THREE.CapsuleGeometry(r, 2 * half, 8, 16);
      // Three capsule is along Y; MuJoCo capsules are typically along Z.
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

  camera = new THREE.PerspectiveCamera(45, 1.0, 0.01, 10.0);
  camera.position.set(0.45, -0.55, 0.45);
  camera.lookAt(new THREE.Vector3(0.25, 0, 0.02));

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.9);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(1.0, -1.0, 1.2);
  scene.add(dir);

  renderer = new THREE.WebGLRenderer({ canvas: simCanvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  controls = new OrbitControls(camera, simCanvas);
  controls.target.set(0.25, 0, 0.02);
  controls.update();

  // MuJoCo geoms
  geomMeshes = [];
  for (let i = 0; i < model.ngeom; i++) {
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
      roughness: 0.85,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    scene.add(mesh);
    geomMeshes.push({ geomIndex: i, mesh });
  }

  // 3D ink dots
  createInk3D();
}

function updateThreeTransformsFromData() {
  const gxpos = data.geom_xpos;
  const gxmat = data.geom_xmat;

  for (const { geomIndex, mesh } of geomMeshes) {
    const p = 3 * geomIndex;
    const r = 9 * geomIndex;

    const px = gxpos[p + 0];
    const py = gxpos[p + 1];
    const pz = gxpos[p + 2];

    // MuJoCo: row-major 3x3
    const m00 = gxmat[r + 0], m01 = gxmat[r + 1], m02 = gxmat[r + 2];
    const m10 = gxmat[r + 3], m11 = gxmat[r + 4], m12 = gxmat[r + 5];
    const m20 = gxmat[r + 6], m21 = gxmat[r + 7], m22 = gxmat[r + 8];

    mesh.matrix.set(
      m00, m01, m02, px,
      m10, m11, m12, py,
      m20, m21, m22, pz,
      0,   0,   0,   1
    );
    mesh.matrixWorldNeedsUpdate = true;
  }
}

// ===== 3D ink (instanced spheres) =====
// We compute the canvas surface plane from the MuJoCo geom named "canvas", so that ink placement
// stays correct even if you move/rotate the canvas in the XML.
const CANVAS_GEOM_NAME = "canvas";
let canvasGeomId = -1;
/** @type {{p0:THREE.Vector3, n:THREE.Vector3} | null} */
let canvasPlane = null;

// Make ink clearly visible (bigger + unlit material).
const INK_RADIUS = 0.004;
const INK_Z_OFFSET = 0.005; // lift above the surface to avoid z-fighting
const MAX_INK_DOTS = 120000;

/** @type {THREE.InstancedMesh | null} */
let inkMesh = null;
let inkCount = 0;
let tmpMat4 = new THREE.Matrix4();
let tmpV3 = new THREE.Vector3();
let tmpV3b = new THREE.Vector3();

function createInk3D() {
  const geom = new THREE.SphereGeometry(INK_RADIUS, 12, 10);
  // Unlit material makes tiny dots visible under any lighting.
  const mat = new THREE.MeshBasicMaterial({ color: 0x111111 });

  inkMesh = new THREE.InstancedMesh(geom, mat, MAX_INK_DOTS);
  // Avoid frustum-culling bugs with instanced bounds (keeps ink visible when camera orbits).
  inkMesh.frustumCulled = false;
  inkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  inkMesh.count = 0;
  inkMesh.visible = showInk3D;
  scene.add(inkMesh);
  inkCount = 0;
}

function clearInk3D() {
  if (!inkMesh) return;
  inkCount = 0;
  inkMesh.count = 0;
  inkMesh.instanceMatrix.needsUpdate = true;
}

function addInkDotFromPoint(p) {
  if (!inkMesh) return;
  if (inkCount >= MAX_INK_DOTS) return;
  // Project to the canvas surface plane (if available), and lift slightly above it.
  if (canvasPlane) {
    tmpV3.set(p.x, p.y, p.z);
    tmpV3b.copy(tmpV3).sub(canvasPlane.p0);
    const d = tmpV3b.dot(canvasPlane.n);
    tmpV3.addScaledVector(canvasPlane.n, -d);
    tmpV3.addScaledVector(canvasPlane.n, INK_Z_OFFSET);
  } else {
    tmpV3.set(p.x, p.y, INK_Z_OFFSET);
  }

  tmpMat4.identity();
  tmpMat4.setPosition(tmpV3.x, tmpV3.y, tmpV3.z);
  inkMesh.setMatrixAt(inkCount, tmpMat4);
  inkCount++;
  inkMesh.count = inkCount;
  inkMesh.instanceMatrix.needsUpdate = true;
}

// =============================
// 2D overlay drawing
// =============================
function drawStrokes(ctx, strokes) {
  for (const stroke of strokes) {
    if (!stroke || stroke.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) ctx.lineTo(stroke[i].x, stroke[i].y);
    ctx.stroke();
  }
}

function redraw2DOverlay(penWorld, targetCmd) {
  // background
  ctx2d.clearRect(0, 0, drawCssW, drawCssH);
  ctx2d.fillStyle = "#ffffff";
  ctx2d.fillRect(0, 0, drawCssW, drawCssH);

  // frame
  ctx2d.strokeStyle = "rgba(0,0,0,0.25)";
  ctx2d.lineWidth = 1;
  ctx2d.setLineDash([]);
  ctx2d.strokeRect(0.5, 0.5, drawCssW - 1, drawCssH - 1);

  if (showTarget) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.22)";
    ctx2d.lineWidth = 2;
    ctx2d.setLineDash([6, 5]);
    drawStrokes(ctx2d, targetStrokes);
  }

  if (showExecuted) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.92)";
    ctx2d.lineWidth = 2.2;
    ctx2d.setLineDash([]);
    drawStrokes(ctx2d, executedStrokes);
  }

  if (targetCmd) {
    const p = worldToCanvasCss(targetCmd.x, targetCmd.y);
    ctx2d.fillStyle = "rgba(255,0,0,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  if (penWorld) {
    const p = worldToCanvasCss(penWorld.x, penWorld.y);
    ctx2d.fillStyle = "rgba(0,120,255,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  // text
  ctx2d.fillStyle = "rgba(0,0,0,0.55)";
  ctx2d.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx2d.fillText(`queue: ${commandQueue.length} | ink: ${inkCount}`, 10, 18);
}

// =============================
// Kinematic follow loop
// =============================
let lastFrameTime = performance.now();
let accumulator = 0;
let fpsCounter = { frames: 0, last: performance.now(), fps: 0 };

let execPenDown = false;
let currentExecStroke = null;
let lastInkCanvasPt = null;

function recordExecutedPoint(penWorld, penIsDown) {
  if (penIsDown) {
    if (!execPenDown) {
      currentExecStroke = [];
      executedStrokes.push(currentExecStroke);
      execPenDown = true;
      lastInkCanvasPt = null;
    }

    const { px, py } = worldToCanvasCss(penWorld.x, penWorld.y);
    const last = currentExecStroke[currentExecStroke.length - 1];
    const minDistPx = 2.0;

    if (!last || Math.hypot(px - last.x, py - last.y) >= minDistPx) {
      currentExecStroke.push({ x: px, y: py });

      // 3D ink: also decimate by the same criterion (in canvas space)
      if (!lastInkCanvasPt || Math.hypot(px - lastInkCanvasPt.x, py - lastInkCanvasPt.y) >= minDistPx) {
        addInkDotFromPoint(penWorld);
        lastInkCanvasPt = { x: px, y: py };
      }
    }
  } else {
    execPenDown = false;
    currentExecStroke = null;
    lastInkCanvasPt = null;
  }
}

function stepOnce() {
  if (!model || !data) return;

  const cmd = commandQueue.length > 0 ? commandQueue[0] : null;
  const penDesired = cmd && cmd.penDown ? PEN_DOWN : PEN_UP;

  // IK target
  let qTarget = null;
  if (cmd) {
    let { x, y } = clampToReach(cmd.x, cmd.y);
    qTarget = pickIKSolution(x, y);
    if (!qTarget) {
      // unreachable (should be rare with clamp) -> drop
      commandQueue.shift();
      qTarget = null;
    }
  }
  if (!qTarget) qTarget = [data.qpos[0], data.qpos[1], data.qpos[2]];

  const simDt = model.opt.timestep || 0.002;

  // Slew qpos towards target (no dynamics)
  const maxDq = JOINT_MAX_RATE * simDt;
  for (let i = 0; i < 3; i++) {
    const diff = angleDiff(qTarget[i], data.qpos[i]);
    const dq = clamp(diff, -maxDq, maxDq);
    let q = data.qpos[i] + dq;
    // Clamp to the MuJoCo joint range to avoid wrap-around jumps near ±pi.
    if (hingeRanges && hingeRanges[i]) {
      q = clamp(q, hingeRanges[i].lo, hingeRanges[i].hi);
    }
    data.qpos[i] = q;
  }

  const maxDp = PEN_MAX_RATE * simDt;
  data.qpos[3] = clamp(data.qpos[3] + clamp(penDesired - data.qpos[3], -maxDp, maxDp), -0.02, 0.05);

  mujoco.mj_forward(model, data);

  const penWorld = getPenWorld();

  // Dequeue when reached
  if (cmd) {
    const err = Math.hypot(penWorld.x - cmd.x, penWorld.y - cmd.y);

    if (cmd.penDown) {
      if (err < TARGET_EPS) {
        // advance exactly one point (path is already resampled at near-constant spacing)
        commandQueue.shift();
        // drop duplicates (rare)
        while (
          commandQueue.length > 0 &&
          cmd.penDown &&
          commandQueue[0].penDown &&
          Math.hypot(commandQueue[0].x - penWorld.x, commandQueue[0].y - penWorld.y) < 1e-4
        ) {
          commandQueue.shift();
        }
      }
    } else {
      // pen-up move / finish
      if (err < MOVE_EPS && data.qpos[3] > PEN_UP * 0.7) {
        commandQueue.shift();
      }
    }
  }

  // Record executed strokes.
  // In this demo (kinematic-only), we treat "pen-down" as a *commanded* state (cmd.penDown),
  // not a physical contact/force state. This makes drawing deterministic and avoids issues if
  // the pen-z slew is still in transit.
  const penIsDown = cmd ? !!cmd.penDown : false;
  recordExecutedPoint(penWorld, penIsDown);

  return { penWorld, targetCmd: cmd };
}

function animate(now) {
  requestAnimationFrame(animate);

  const dt = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  accumulator += Math.min(0.05, dt);

  const simDt = model ? model.opt.timestep : 0.002;
  let snapshot = null;
  while (model && data && accumulator >= simDt) {
    snapshot = stepOnce();
    accumulator -= simDt;
  }

  if (model && data) {
    updateThreeTransformsFromData();
    renderer.render(scene, camera);

    const penWorld = snapshot?.penWorld ?? getPenWorld();
    const targetCmd = snapshot?.targetCmd ?? (commandQueue.length > 0 ? commandQueue[0] : null);
    redraw2DOverlay(penWorld, targetCmd);

    fpsCounter.frames++;
    const t = performance.now();
    if (t - fpsCounter.last > 500) {
      fpsCounter.fps = (fpsCounter.frames * 1000) / (t - fpsCounter.last);
      fpsCounter.frames = 0;
      fpsCounter.last = t;
      statusEl.textContent = `fps ${fpsCounter.fps.toFixed(1)} | queue ${commandQueue.length} | ink ${inkCount}`;
    }
  }
}

// =============================
// Self tests (browser-triggered)
// =============================
async function runSelfTestBasic() {
  setDiagVisible(true);
  diagClear();
  diagLine("== Basic self-test ==");
  const t0 = performance.now();

  if (!mujoco || !model || !data) {
    diagLine("❌ MuJoCo/model not loaded yet.");
    return;
  }

  diagLine(`✅ model loaded | nq=${model.nq} | nu=${model.nu} | nsite=${model.nsite} | ngeom=${model.ngeom}`);

  // Pen site
  if (penSiteId >= 0) {
    const p = getPenWorld();
    diagLine(`✅ pen site: ${PEN_SITE_NAME} id=${penSiteId} | penWorld=(${p.x.toFixed(4)},${p.y.toFixed(4)},${p.z.toFixed(4)})`);
  } else {
    diagLine(`⚠️ pen site lookup failed (fallback) | nsite=${model.nsite}`);
  }

  // Canvas plane
  computeCanvasPlaneFromModel();
  if (canvasPlane) {
    diagLine(
      `✅ canvas plane (${CANVAS_GEOM_NAME}) id=${canvasGeomId} | p0=(${canvasPlane.p0.x.toFixed(4)},${canvasPlane.p0.y.toFixed(4)},${canvasPlane.p0.z.toFixed(4)}) n=(${canvasPlane.n.x.toFixed(3)},${canvasPlane.n.y.toFixed(3)},${canvasPlane.n.z.toFixed(3)})`
    );
  } else {
    diagLine(`⚠️ canvas geom "${CANVAS_GEOM_NAME}" not found; ink will use a fixed Z.`);
  }

  // Ink mesh
  if (inkMesh) {
    diagLine(`✅ ink mesh ready | inkCount=${inkCount} | visible=${inkMesh.visible}`);
  } else {
    diagLine("❌ ink mesh is not created (Three.js scene not ready?)");
  }

  // Transform round-trip
  let maxErr = 0;
  for (let i = 0; i < 6; i++) {
    const px = Math.random() * drawCssW;
    const py = Math.random() * drawCssH;
    const w = canvasCssToWorld(px, py);
    const p2 = worldToCanvasCss(w.x, w.y);
    const e = Math.hypot(px - p2.px, py - p2.py);
    if (e > maxErr) maxErr = e;
  }
  diagLine(`✅ canvas↔world round-trip maxErr=${maxErr.toFixed(4)} px`);

  const ms = performance.now() - t0;
  diagLine(`(done in ${ms.toFixed(1)} ms)`);
}

function addTestInk() {
  setDiagVisible(true);
  if (!mujoco || !model || !data) {
    diagLine("❌ MuJoCo/model not loaded yet.");
    return;
  }
  if (!inkMesh) {
    diagLine("❌ ink mesh not ready.");
    return;
  }
  computeCanvasPlaneFromModel();

  const before = inkCount;
  // A small cross at the center of the workspace.
  const cx = (workspace.xMin + workspace.xMax) * 0.5;
  const cy = 0.0;
  const dz = canvasPlane ? canvasPlane.p0.z : 0.0;

  const pts = [
    { x: cx, y: cy },
    { x: cx + 0.01, y: cy },
    { x: cx - 0.01, y: cy },
    { x: cx, y: cy + 0.01 },
    { x: cx, y: cy - 0.01 },
    { x: cx + 0.02, y: cy },
    { x: cx - 0.02, y: cy },
    { x: cx, y: cy + 0.02 },
    { x: cx, y: cy - 0.02 },
  ];
  for (const p of pts) addInkDotFromPoint({ x: p.x, y: p.y, z: dz });

  diagLine(`✅ addTestInk: +${inkCount - before} dots (inkCount=${inkCount})`);
}

async function runSelfTestPath() {
  setDiagVisible(true);
  diagClear();
  diagLine("== Path follow test (square) ==");
  if (!mujoco || !model || !data) {
    diagLine("❌ MuJoCo/model not loaded yet.");
    return;
  }

  // This test clears current strokes/ink/queue.
  targetStrokes = [];
  executedStrokes = [];
  commandQueue = [];
  clearInk3D();
  lastWorldInStroke = null;

  // Home pose
  data.qpos[0] = 0.0;
  data.qpos[1] = 0.8;
  data.qpos[2] = -0.8;
  data.qpos[3] = PEN_UP;
  mujoco.mj_forward(model, data);
  computeCanvasPlaneFromModel();

  // Build a square in a safe reachable region
  const cx = (workspace.xMin + workspace.xMax) * 0.5;
  const cy = 0.0;
  const s = 0.10; // 10cm
  const ptsW = [
    { x: cx - s * 0.5, y: cy - s * 0.5 },
    { x: cx + s * 0.5, y: cy - s * 0.5 },
    { x: cx + s * 0.5, y: cy + s * 0.5 },
    { x: cx - s * 0.5, y: cy + s * 0.5 },
    { x: cx - s * 0.5, y: cy - s * 0.5 },
  ];

  // For target overlay
  const stroke = ptsW.map((p) => {
    const c = worldToCanvasCss(p.x, p.y);
    return { x: c.px, y: c.py };
  });
  targetStrokes.push(stroke);

  // Queue: move to start, pen-down, then follow edges
  const p0 = ptsW[0];
  commandQueue.push({ x: p0.x, y: p0.y, penDown: false });
  commandQueue.push({ x: p0.x, y: p0.y, penDown: true });
  for (let i = 0; i < ptsW.length - 1; i++) {
    const seg = resampleSegment(ptsW[i], ptsW[i + 1], PATH_STEP);
    for (const p of seg) commandQueue.push({ x: p.x, y: p.y, penDown: true });
  }
  const plast = ptsW[ptsW.length - 1];
  commandQueue.push({ x: plast.x, y: plast.y, penDown: false });

  const simDt = model.opt.timestep || 0.002;
  const maxSteps = Math.floor(4.0 / simDt);
  for (let i = 0; i < maxSteps; i++) {
    stepOnce();
    // yield to keep the UI responsive
    if (i % 250 === 0) await new Promise((r) => setTimeout(r, 0));
    if (commandQueue.length === 0) break;
  }

  diagLine(`queueRemaining=${commandQueue.length}`);
  diagLine(`executedStrokes=${executedStrokes.length}`);
  diagLine(`inkCount=${inkCount}`);

  if (commandQueue.length === 0 && executedStrokes.length > 0 && inkCount > 0) {
    diagLine("✅ Path test OK (ink + executed strokes present)");
  } else {
    diagLine("❌ Path test failed: queue did not drain or ink was not produced");
  }
}

// =============================
// Boot
// =============================
async function boot() {
  try {
    statusEl.textContent = "loading mujoco-js…";
    mujoco = await load_mujoco();

    statusEl.textContent = "loading model…";

    mujoco.FS.mkdir("/working");
    mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

    const xmlText = await (await fetch("./models/arm_pen.xml")).text();
    mujoco.FS.writeFile("/working/arm_pen.xml", xmlText);

    model = mujoco.MjModel.loadFromXML("/working/arm_pen.xml");
    data = new mujoco.MjData(model);

    // Cache hinge joint ranges for kinematic clamping.
    hingeRanges = [
      { lo: model.jnt_range[0], hi: model.jnt_range[1] },
      { lo: model.jnt_range[2], hi: model.jnt_range[3] },
      { lo: model.jnt_range[4], hi: model.jnt_range[5] },
    ];

    // Resolve pen site ID (fallback to 0 if lookup fails)
    penSiteId = lookupSiteIdByName(PEN_SITE_NAME);
    if (penSiteId < 0) {
      console.warn(`[warn] site "${PEN_SITE_NAME}" not found via mj_name2id. Using site 0 fallback.`);
    }

    // Home pose
    data.qpos[0] = 0.0;
    data.qpos[1] = 0.8;
    data.qpos[2] = -0.8;
    data.qpos[3] = PEN_UP;
    mujoco.mj_forward(model, data);

    statusEl.textContent = "init renderer…";
    createThreeSceneFromModel();
    resizeAll();

    statusEl.textContent = "ready";
    requestAnimationFrame(animate);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "failed to load (see console)";
  }
}

resizeAll();
boot();
