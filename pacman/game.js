/* MuJoCo WASM Pac-Man-ish mini game.
 *
 * Design goals:
 * - Use MuJoCo for collision against maze walls (contact-rich part).
 * - Keep rendering lightweight using Canvas 2D top-down.
 * - Keep game logic simple (pellets + a chasing ghost).
 *
 * Controls:
 * - Arrow keys / WASD: move
 * - Space: pause
 * - R: restart
 */

// -----------------------------
// MuJoCo loader (CDN global build)
// -----------------------------
async function loadMujocoFromGlobal() {
  // mujoco-js builds have had small API differences across versions/bundles.
  // We try a few common global entry points to be robust.
  const candidates = [
    window.load_mujoco,          // common in mujoco_wasm.js examples
    window.loadMujoco,           // sometimes camelCase
    window.mujoco?.load_mujoco,  // sometimes nested under a namespace
    window.mujoco?.default,      // sometimes default-like
  ].filter(Boolean);

  if (candidates.length === 0) {
    throw new Error(
      "MuJoCo WASM script loaded, but no known global loader was found. " +
      "Inspect `window` in DevTools to see what symbol it exports."
    );
  }

  // Pick the first function-like candidate.
  const loader = candidates.find((x) => typeof x === "function");
  if (!loader) {
    throw new Error(
      "Found candidate globals but none were callable. " +
      "Inspect `window.load_mujoco` / `window.mujoco` in DevTools."
    );
  }

  // The loader typically returns a Promise that resolves to the mujoco module object.
  return await loader();
}


// -----------------------------
// Game configuration
// -----------------------------
const CELL = 1.0;                 // Grid cell size in MuJoCo world units
const PAC_SPEED = 4.0;            // Desired velocity in world units/sec
const GHOST_SPEED = 3.2;          // Ghost speed
const SNAP_EPS = 0.10;            // How close to a cell center to allow turning/snapping
const AI_UPDATE_SEC = 0.25;       // Ghost path update interval
const SIM_TIMESTEP = 0.01;        // MuJoCo timestep set in MJCF

const PAC_RADIUS = 0.28;
const GHOST_RADIUS = 0.28;

const COLORS = {
  bg: "#000000",
  wall: "#1a4cff",
  pellet: "#ffffff",
  pac: "#ffeb3b",
  ghost: "#ff3b3b",
  text: "#ffffff",
};

// A compact Pac-Man-ish maze.
// Legend:
//   # = wall
//   . = pellet
//   P = pacman spawn
//   G = ghost spawn
//   (space) = empty
const MAP = [
  "###############",
  "#P....#.......#",
  "#.###.#.#####.#",
  "#.#...#.....#.#",
  "#.#.#####.#.#.#",
  "#.#.......#.#.#",
  "#.#######.#.#.#",
  "#.....#...#...#",
  "###.#.#.#####.#",
  "#...#.#.....#.#",
  "#.###.#####.#.#",
  "#.#.....#...#.#",
  "#.#.###.#.###.#",
  "#...#...#...G.#",
  "###############",
];

// Directions in world coordinates (x right, y up).
const DIRS = {
  up:    { dx: 0, dy: 1 },
  down:  { dx: 0, dy: -1 },
  left:  { dx: -1, dy: 0 },
  right: { dx: 1, dy: 0 },
  none:  { dx: 0, dy: 0 },
};

// -----------------------------
// DOM
// -----------------------------
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const elScore = document.getElementById("score");
const elTotal = document.getElementById("total");
const elStatus = document.getElementById("status");

// -----------------------------
// Map parsing helpers
// -----------------------------
function mapDims() {
  const H = MAP.length;
  const W = MAP[0].length;
  return { W, H };
}

function assertRectangular() {
  const { W } = mapDims();
  for (let r = 0; r < MAP.length; r++) {
    if (MAP[r].length !== W) {
      throw new Error("MAP must be rectangular: row " + r + " has different length.");
    }
  }
}

function inBounds(col, row, W, H) {
  return col >= 0 && col < W && row >= 0 && row < H;
}

// Convert map cell (col,rowFromTop) to world center coordinate (x,y).
function cellToWorldCenter(col, rowTop, W, H) {
  const x = (col + 0.5) * CELL;
  const yFromBottom = (H - 1 - rowTop);
  const y = (yFromBottom + 0.5) * CELL;
  return { x, y };
}

// Convert world (x,y) to map cell (col,rowFromTop).
function worldToCell(x, y, W, H) {
  const col = Math.floor(x / CELL);
  const rowFromBottom = Math.floor(y / CELL);
  const rowTop = (H - 1) - rowFromBottom;
  return { col, rowTop };
}

function cellKey(col, rowTop) {
  return `${col},${rowTop}`;
}

// "Near center" check: helps with grid-accurate turning and snapping.
function nearCellCenter(x, y, col, rowTop, W, H) {
  const c = cellToWorldCenter(col, rowTop, W, H);
  const dx = x - c.x;
  const dy = y - c.y;
  return (dx * dx + dy * dy) <= (SNAP_EPS * SNAP_EPS);
}

// Given a cell in rowTop indexing and a dir in world coordinates, compute next cell.
function nextCell(cell, dir) {
  // Moving up in world decreases rowTop.
  return { col: cell.col + dir.dx, rowTop: cell.rowTop - dir.dy };
}

// -----------------------------
// Simple BFS pathfinding (grid)
// -----------------------------
function bfsNextStep(start, goal, walls, W, H) {
  const sKey = cellKey(start.col, start.rowTop);
  const gKey = cellKey(goal.col, goal.rowTop);

  if (sKey === gKey) return null;

  const q = [];
  const parent = new Map();

  q.push(start);
  parent.set(sKey, null);

  const neighbors = [DIRS.up, DIRS.down, DIRS.left, DIRS.right];

  while (q.length) {
    const cur = q.shift();
    const curKey = cellKey(cur.col, cur.rowTop);

    if (curKey === gKey) break;

    for (const d of neighbors) {
      const n = nextCell(cur, d);
      if (!inBounds(n.col, n.rowTop, W, H)) continue;
      if (walls[n.rowTop][n.col]) continue;

      const nKey = cellKey(n.col, n.rowTop);
      if (parent.has(nKey)) continue;

      parent.set(nKey, cur);
      q.push(n);
    }
  }

  if (!parent.has(gKey)) return null; // No path found

  // Reconstruct path from goal back to start.
  const path = [];
  let cur = goal;
  while (cur) {
    path.push(cur);
    const p = parent.get(cellKey(cur.col, cur.rowTop));
    cur = p;
  }
  path.reverse();

  // path[0] is start, path[1] is the immediate next step.
  if (path.length < 2) return null;
  return path[1];
}

// Convert delta between two cells into a DIRS direction.
function dirFromCells(a, b) {
  const dx = b.col - a.col;
  const dyRowTop = b.rowTop - a.rowTop;

  // Recall: moving up in world means rowTop decreases.
  // So world dy is -dyRowTop.
  const dy = -dyRowTop;

  if (dx === 1 && dy === 0) return DIRS.right;
  if (dx === -1 && dy === 0) return DIRS.left;
  if (dx === 0 && dy === 1) return DIRS.up;
  if (dx === 0 && dy === -1) return DIRS.down;
  return DIRS.none;
}

// -----------------------------
// MJCF generation (maze + 2 bodies)
// -----------------------------
function buildMJCF({ walls, W, H }) {
  // Wall geometry: one box per wall cell.
  // For a more optimized model you would merge wall segments, but this is fine for small maps.
  const wallHalf = 0.50 * CELL;
  const wallHalfH = 0.35;
  const wallZ = wallHalfH;

  const xMin = 0.5 * CELL;
  const xMax = (W - 0.5) * CELL;
  const yMin = 0.5 * CELL;
  const yMax = (H - 0.5) * CELL;

  const geoms = [];
  for (let rowTop = 0; rowTop < H; rowTop++) {
    for (let col = 0; col < W; col++) {
      if (!walls[rowTop][col]) continue;
      const p = cellToWorldCenter(col, rowTop, W, H);
      geoms.push(
        `<geom name="wall_${col}_${rowTop}" type="box" size="${wallHalf} ${wallHalf} ${wallHalfH}" ` +
        `pos="${p.x} ${p.y} ${wallZ}" rgba="0.10 0.30 1.00 1" />`
      );
    }
  }

  // We keep gravity = 0 and constrain motion to x/y using slide joints.
  // qpos coordinates directly correspond to world x/y, because parent body is at (0,0,z).
  return `<?xml version="1.0" encoding="utf-8"?>
<mujoco model="pacman_mini">
  <option timestep="${SIM_TIMESTEP}" gravity="0 0 0" integrator="Euler" />
  <size njmax="4000" nconmax="500" />

  <default>
    <geom contype="1" conaffinity="1"
          friction="1 0.01 0.001"
          solref="0.02 1"
          solimp="0.95 0.95 0.01" />
    <joint damping="2" />
  </default>

  <worldbody>
    ${geoms.join("\n    ")}

    <body name="pacman" pos="0 0 0.30">
      <joint name="pacman_x" type="slide" axis="1 0 0" limited="true" range="${xMin} ${xMax}" />
      <joint name="pacman_y" type="slide" axis="0 1 0" limited="true" range="${yMin} ${yMax}" />
      <geom name="pacman_geom" type="sphere" size="${PAC_RADIUS}" rgba="1 0.95 0.10 1" />
    </body>

    <body name="ghost" pos="0 0 0.30">
      <joint name="ghost_x" type="slide" axis="1 0 0" limited="true" range="${xMin} ${xMax}" />
      <joint name="ghost_y" type="slide" axis="0 1 0" limited="true" range="${yMin} ${yMax}" />
      <geom name="ghost_geom" type="sphere" size="${GHOST_RADIUS}" rgba="1 0.20 0.20 1" />
    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servos: ctrl is the desired joint velocity. -->
    <velocity joint="pacman_x" kv="25" ctrlrange="-${PAC_SPEED} ${PAC_SPEED}" />
    <velocity joint="pacman_y" kv="25" ctrlrange="-${PAC_SPEED} ${PAC_SPEED}" />
    <velocity joint="ghost_x"  kv="18" ctrlrange="-${GHOST_SPEED} ${GHOST_SPEED}" />
    <velocity joint="ghost_y"  kv="18" ctrlrange="-${GHOST_SPEED} ${GHOST_SPEED}" />
  </actuator>
</mujoco>`;
}

// -----------------------------
// Rendering helpers
// -----------------------------
function resizeCanvas() {
  const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawCircle(x, y, r, fillStyle) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

function drawPacman(x, y, r, t, moving, dir) {
  // Simple mouth animation: open/close over time when moving.
  const baseOpen = moving ? (0.15 + 0.25 * (0.5 + 0.5 * Math.sin(t * 12))) : 0.08;
  const mouth = baseOpen * Math.PI;

  let ang = 0;
  if (dir === DIRS.right) ang = 0;
  else if (dir === DIRS.left) ang = Math.PI;
  else if (dir === DIRS.up) ang = -Math.PI / 2;
  else if (dir === DIRS.down) ang = Math.PI / 2;

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.arc(x, y, r, ang + mouth, ang - mouth, false);
  ctx.closePath();
  ctx.fillStyle = COLORS.pac;
  ctx.fill();
}

function setupWorldToCanvas(W, H) {
  const worldW = W * CELL;
  const worldH = H * CELL;

  // Fit the whole maze into the canvas while keeping aspect ratio.
  const scale = Math.min(
    (window.innerWidth * 0.92) / worldW,
    (window.innerHeight * 0.92) / worldH
  );
  const offsetX = (window.innerWidth - worldW * scale) * 0.5;
  const offsetY = (window.innerHeight - worldH * scale) * 0.5;

  // World y is "up", canvas y is "down", so we invert y.
  function worldToCanvas(wx, wy) {
    const cx = offsetX + wx * scale;
    const cy = offsetY + (worldH - wy) * scale;
    return { x: cx, y: cy };
  }

  return { scale, offsetX, offsetY, worldW, worldH, worldToCanvas };
}

// -----------------------------
// Main
// -----------------------------
async function main() {
  assertRectangular();
  resizeCanvas();
  window.addEventListener("resize", () => {
    resizeCanvas();
  });

  // Parse map
  const { W, H } = mapDims();
  const walls = Array.from({ length: H }, () => Array(W).fill(false));
  const pellets = new Map(); // key: "col,rowTop" => {col,rowTop, eaten}
  let pacSpawn = null;
  let ghostSpawn = null;

  for (let rowTop = 0; rowTop < H; rowTop++) {
    for (let col = 0; col < W; col++) {
      const ch = MAP[rowTop][col];
      if (ch === "#") {
        walls[rowTop][col] = true;
      } else if (ch === ".") {
        pellets.set(cellKey(col, rowTop), { col, rowTop, eaten: false });
      } else if (ch === "P") {
        pacSpawn = { col, rowTop };
      } else if (ch === "G") {
        ghostSpawn = { col, rowTop };
      }
    }
  }

  if (!pacSpawn || !ghostSpawn) {
    throw new Error("MAP must include 'P' (pacman spawn) and 'G' (ghost spawn).");
  }

  const totalPellets = pellets.size;
  elTotal.textContent = String(totalPellets);

  // Load MuJoCo WASM
  elStatus.textContent = "Loading MuJoCo module…";
  const mujoco = await loadMujocoFromGlobal();

  // Virtual file system setup + model load
  // (This is the standard workflow: create a working dir, write XML, then load model from that path.)
  mujoco.FS.mkdir("/working");
  mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

  const mjcf = buildMJCF({ walls, W, H });
  mujoco.FS.writeFile("/working/pacman.xml", mjcf);

  const model = mujoco.MjModel.loadFromXML("/working/pacman.xml");
  const data = new mujoco.MjData(model);

  // Joint / actuator indices:
  // We rely on the XML order:
  //   qpos[0]=pacman_x, qpos[1]=pacman_y, qpos[2]=ghost_x, qpos[3]=ghost_y
  //   ctrl[0..3] correspond to the 4 velocity actuators in that order.
  const PAC_X = 0, PAC_Y = 1, GHOST_X = 2, GHOST_Y = 3;
  const PAC_CTRL_X = 0, PAC_CTRL_Y = 1, GHOST_CTRL_X = 2, GHOST_CTRL_Y = 3;

  function resetGameState() {
    // Reset physics state
    mujoco.mj_resetData(model, data);

    const p0 = cellToWorldCenter(pacSpawn.col, pacSpawn.rowTop, W, H);
    const g0 = cellToWorldCenter(ghostSpawn.col, ghostSpawn.rowTop, W, H);

    data.qpos[PAC_X] = p0.x;
    data.qpos[PAC_Y] = p0.y;
    data.qpos[GHOST_X] = g0.x;
    data.qpos[GHOST_Y] = g0.y;

    data.qvel[PAC_X] = 0;
    data.qvel[PAC_Y] = 0;
    data.qvel[GHOST_X] = 0;
    data.qvel[GHOST_Y] = 0;

    data.ctrl[PAC_CTRL_X] = 0;
    data.ctrl[PAC_CTRL_Y] = 0;
    data.ctrl[GHOST_CTRL_X] = 0;
    data.ctrl[GHOST_CTRL_Y] = 0;

    // Recompute derived quantities
    mujoco.mj_forward(model, data);

    // Reset pellets
    for (const p of pellets.values()) {
      p.eaten = false;
    }

    score = 0;
    elScore.textContent = "0";

    requestedDir = DIRS.none;
    pacDir = DIRS.none;
    ghostDir = DIRS.none;

    gameOver = false;
    win = false;
    paused = false;
    aiTimer = 0;

    elStatus.textContent = "Ready! Move with arrows/WASD.";
  }

  // Input state
  let requestedDir = DIRS.none;
  let pacDir = DIRS.none;
  let ghostDir = DIRS.none;

  // Game state
  let score = 0;
  let gameOver = false;
  let win = false;
  let paused = false;
  let aiTimer = 0;

  // Keyboard controls
  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    # debugTH
    elStatus.textContent = `Key: ${k}`;

    if (k === "r") {
      resetGameState();
      return;
    }
    if (k === " ") {
      paused = !paused;
      elStatus.textContent = paused ? "Paused (Space to resume)" : "Resumed";
      return;
    }

    // Movement keys
    if (k === "arrowup" || k === "w") requestedDir = DIRS.up;
    else if (k === "arrowdown" || k === "s") requestedDir = DIRS.down;
    else if (k === "arrowleft" || k === "a") requestedDir = DIRS.left;
    else if (k === "arrowright" || k === "d") requestedDir = DIRS.right;
  });

  // Initialize
  resetGameState();

  // Simulation loop with fixed-step accumulator
  let last = performance.now();
  let acc = 0;

  function isWallCell(col, rowTop) {
    if (!inBounds(col, rowTop, W, H)) return true;
    return walls[rowTop][col];
  }

  function canMoveFromCell(cell, dir) {
    if (dir === DIRS.none) return true;
    const n = nextCell(cell, dir);
    if (!inBounds(n.col, n.rowTop, W, H)) return false;
    return !walls[n.rowTop][n.col];
  }


  function snapBodyToCenterIfClose(bodyXIdx, bodyYIdx) {
    // Don't snap while moving fast; otherwise the character gets "stuck"
    // because each simulation step moves less than SNAP_EPS.
    const vx = data.qvel[bodyXIdx];
    const vy = data.qvel[bodyYIdx];
    if (Math.hypot(vx, vy) > 0.5) return;

    const x = data.qpos[bodyXIdx];
    const y = data.qpos[bodyYIdx];
    const cell = worldToCell(x, y, W, H);
    if (!inBounds(cell.col, cell.rowTop, W, H)) return;

    if (nearCellCenter(x, y, cell.col, cell.rowTop, W, H)) {
      const c = cellToWorldCenter(cell.col, cell.rowTop, W, H);
      data.qpos[bodyXIdx] = c.x;
      data.qpos[bodyYIdx] = c.y;
      data.qvel[bodyXIdx] = 0;
      data.qvel[bodyYIdx] = 0;
    }
  }



  function updatePacmanControls() {
    const x = data.qpos[PAC_X];
    const y = data.qpos[PAC_Y];
    const cell = worldToCell(x, y, W, H);

    if (!inBounds(cell.col, cell.rowTop, W, H)) {
      pacDir = DIRS.none;
      data.ctrl[PAC_CTRL_X] = 0;
      data.ctrl[PAC_CTRL_Y] = 0;
      return;
    }

    const atCenter = nearCellCenter(x, y, cell.col, cell.rowTop, W, H);

    if (atCenter) {
      // Snap to center to keep motion grid-accurate.
      snapBodyToCenterIfClose(PAC_X, PAC_Y);

      // Turn if requested direction is possible.
      if (requestedDir !== DIRS.none && canMoveFromCell(cell, requestedDir)) {
        pacDir = requestedDir;
      }

      // Stop if current direction is blocked.
      if (!canMoveFromCell(cell, pacDir)) {
        pacDir = DIRS.none;
      }
    }

    // Apply desired velocities via MuJoCo velocity actuators.
    data.ctrl[PAC_CTRL_X] = pacDir.dx * PAC_SPEED;
    data.ctrl[PAC_CTRL_Y] = pacDir.dy * PAC_SPEED;
  }

  function updateGhostAI(dtSec) {
    aiTimer += dtSec;
    if (aiTimer < AI_UPDATE_SEC) return;
    aiTimer = 0;

    const px = data.qpos[PAC_X];
    const py = data.qpos[PAC_Y];
    const gx = data.qpos[GHOST_X];
    const gy = data.qpos[GHOST_Y];

    const pacCell = worldToCell(px, py, W, H);
    const ghostCell = worldToCell(gx, gy, W, H);

    if (!inBounds(pacCell.col, pacCell.rowTop, W, H)) return;
    if (!inBounds(ghostCell.col, ghostCell.rowTop, W, H)) return;

    const next = bfsNextStep(ghostCell, pacCell, walls, W, H);
    if (!next) return;

    ghostDir = dirFromCells(ghostCell, next);
  }

  function updateGhostControls() {
    const x = data.qpos[GHOST_X];
    const y = data.qpos[GHOST_Y];
    const cell = worldToCell(x, y, W, H);

    if (!inBounds(cell.col, cell.rowTop, W, H)) {
      ghostDir = DIRS.none;
      data.ctrl[GHOST_CTRL_X] = 0;
      data.ctrl[GHOST_CTRL_Y] = 0;
      return;
    }

    const atCenter = nearCellCenter(x, y, cell.col, cell.rowTop, W, H);
    if (atCenter) {
      snapBodyToCenterIfClose(GHOST_X, GHOST_Y);

      if (!canMoveFromCell(cell, ghostDir)) {
        ghostDir = DIRS.none;
      }
    }

    data.ctrl[GHOST_CTRL_X] = ghostDir.dx * GHOST_SPEED;
    data.ctrl[GHOST_CTRL_Y] = ghostDir.dy * GHOST_SPEED;
  }

  function handlePelletsAndWin() {
    const x = data.qpos[PAC_X];
    const y = data.qpos[PAC_Y];
    const cell = worldToCell(x, y, W, H);
    const key = cellKey(cell.col, cell.rowTop);

    const pellet = pellets.get(key);
    if (pellet && !pellet.eaten) {
      pellet.eaten = true;
      score += 1;
      elScore.textContent = String(score);

      if (score >= totalPellets) {
        win = true;
        elStatus.textContent = "You win! (R to restart)";
      }
    }
  }

  function handleGhostCollision() {
    const px = data.qpos[PAC_X];
    const py = data.qpos[PAC_Y];
    const gx = data.qpos[GHOST_X];
    const gy = data.qpos[GHOST_Y];

    const dx = px - gx;
    const dy = py - gy;
    const r = PAC_RADIUS + GHOST_RADIUS;
    if ((dx * dx + dy * dy) < (r * r * 0.75)) {
      gameOver = true;
      elStatus.textContent = "Game Over! (R to restart)";
    }
  }

  function render(tSec) {
    // Canvas uses CSS pixels, not device pixels (we setTransform with dpr).
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

    const { scale, worldToCanvas, worldW, worldH, offsetX, offsetY } = setupWorldToCanvas(W, H);

    // Draw walls
    ctx.fillStyle = COLORS.wall;
    for (let rowTop = 0; rowTop < H; rowTop++) {
      for (let col = 0; col < W; col++) {
        if (!walls[rowTop][col]) continue;

        // Cell bounds in world coordinates:
        const left = col * CELL;
        const topWorldY = (H - rowTop) * CELL; // top edge in world (y up)
        const right = left + CELL;
        const bottomWorldY = topWorldY - CELL;

        // Convert (left, top) and (right, bottom) to canvas.
        const a = worldToCanvas(left, topWorldY);
        const b = worldToCanvas(right, bottomWorldY);

        const x = a.x;
        const y = a.y;
        const w = (b.x - a.x);
        const h = (b.y - a.y);

        ctx.fillRect(x, y, w, h);
      }
    }

    // Draw pellets
    for (const p of pellets.values()) {
      if (p.eaten) continue;
      const wp = cellToWorldCenter(p.col, p.rowTop, W, H);
      const cp = worldToCanvas(wp.x, wp.y);
      drawCircle(cp.x, cp.y, Math.max(2, 0.08 * CELL * scale), COLORS.pellet);
    }

    // Draw pacman and ghost
    const pacCanvas = worldToCanvas(data.qpos[0], data.qpos[1]);
    const ghostCanvas = worldToCanvas(data.qpos[2], data.qpos[3]);

    drawPacman(
      pacCanvas.x,
      pacCanvas.y,
      PAC_RADIUS * scale,
      tSec,
      pacDir !== DIRS.none && !paused && !gameOver && !win,
      pacDir
    );

    drawCircle(ghostCanvas.x, ghostCanvas.y, GHOST_RADIUS * scale, COLORS.ghost);

    // Optional: draw a thin border around the maze
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(offsetX, offsetY, worldW * scale, worldH * scale);
  }

  function frame(now) {
    const dt = (now - last) / 1000;
    last = now;
    acc += dt;

    if (!paused && !gameOver && !win) {
      // Fixed-step simulation
      while (acc >= SIM_TIMESTEP) {
        // Update controls and AI before each physics step
        updateGhostAI(SIM_TIMESTEP);
        updatePacmanControls();
        updateGhostControls();

        mujoco.mj_step(model, data);

        // Game logic
        handlePelletsAndWin();
        handleGhostCollision();

        // Keep bodies aligned to cell centers when close (reduces drift)
        snapBodyToCenterIfClose(PAC_X, PAC_Y);
        snapBodyToCenterIfClose(GHOST_X, GHOST_Y);

        acc -= SIM_TIMESTEP;

        if (gameOver || win) {
          // Stop motion immediately after terminal state.
          data.ctrl[PAC_CTRL_X] = 0;
          data.ctrl[PAC_CTRL_Y] = 0;
          data.ctrl[GHOST_CTRL_X] = 0;
          data.ctrl[GHOST_CTRL_Y] = 0;
          break;
        }
      }
    }

    render(now / 1000);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  elStatus.textContent = "Error: " + (err?.message || String(err));
});
