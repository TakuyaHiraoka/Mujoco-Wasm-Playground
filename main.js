import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";



const canvas = document.getElementById("c");
const statusEl = document.getElementById("status");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200);
camera.position.set(2.8, 2.0, 1.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.4);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(5, 3, 8);
scene.add(dir);

scene.add(new THREE.GridHelper(20, 40));

const keys = new Set();
addEventListener("keydown", (e) => keys.add(e.key));
addEventListener("keyup", (e) => keys.delete(e.key));

let paused = false;

// --- MuJoCo init (VFS→loadFromXML→step の流れ) :contentReference[oaicite:3]{index=3}
const mujoco = await load_mujoco();

mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
mujoco.FS.writeFile(
  "/working/boxbot.xml",
  await (await fetch("./assets/boxbot.xml")).text()
);

const model = mujoco.MjModel.loadFromXML("/working/boxbot.xml");
const data = new mujoco.MjData(model);

const timestep = model.opt.timestep;
const ngeom = model.ngeom;

// geom arrays
const geom_type = model.geom_type;
const geom_size = model.geom_size;
const geom_rgba = model.geom_rgba;

const mjGEOM_PLANE = 0;
const mjGEOM_BOX = 6;

// build meshes
const meshes = new Array(ngeom);
for (let i = 0; i < ngeom; i++) {
  const t = geom_type[i];
  const r = geom_rgba[4 * i + 0], g = geom_rgba[4 * i + 1], b = geom_rgba[4 * i + 2], a = geom_rgba[4 * i + 3];
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, g, b),
    transparent: a < 0.999,
    opacity: a,
    roughness: 0.9,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });

  let mesh = null;
  if (t === mjGEOM_PLANE) {
    mesh = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), mat);
  } else if (t === mjGEOM_BOX) {
    const sx = geom_size[3 * i + 0] * 2;
    const sy = geom_size[3 * i + 1] * 2;
    const sz = geom_size[3 * i + 2] * 2;
    mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  }
  if (!mesh) continue;

  meshes[i] = mesh;
  scene.add(mesh);
}

function updateMeshes() {
  const gx = data.geom_xpos;
  const gm = data.geom_xmat;

  for (let i = 0; i < ngeom; i++) {
    const m = meshes[i];
    if (!m) continue;

    m.position.set(gx[3*i+0], gx[3*i+1], gx[3*i+2]);

    const r00 = gm[9*i+0], r01 = gm[9*i+1], r02 = gm[9*i+2];
    const r10 = gm[9*i+3], r11 = gm[9*i+4], r12 = gm[9*i+5];
    const r20 = gm[9*i+6], r21 = gm[9*i+7], r22 = gm[9*i+8];

    const mat4 = new THREE.Matrix4().set(
      r00, r01, r02, 0,
      r10, r11, r12, 0,
      r20, r21, r22, 0,
      0,   0,   0,   1
    );
    m.setRotationFromMatrix(mat4);
  }
}

// bodyId: 0=world, 1=box (このXMLの順)
const boxBodyId = 1;

function applyKeyboardForce() {
  // 毎ステップ clear（押しっぱなしじゃない時も力が残らないように）
  data.qfrc_applied.fill(0);

  const push = 60;
  const fx = (keys.has("w")||keys.has("W") ?  push : 0) + (keys.has("s")||keys.has("S") ? -push : 0);
  const fy = (keys.has("a")||keys.has("A") ?  push : 0) + (keys.has("d")||keys.has("D") ? -push : 0);
  const fz = (keys.has(" ") ? push*1.5 : 0);

  const px = data.xpos[3*boxBodyId + 0];
  const py = data.xpos[3*boxBodyId + 1];
  const pz = data.xpos[3*boxBodyId + 2];

  mujoco.mj_applyFT(
    model, data,
    [fx, fy, fz],
    [0, 0, 0],
    [px, py, pz],
    boxBodyId,
    data.qfrc_applied
  );
}

let acc = 0;
let last = performance.now();

function loop(now) {
  requestAnimationFrame(loop);

  // one-shot keys
  if (keys.has("p") || keys.has("P")) { paused = !paused; keys.delete("p"); keys.delete("P"); }
  if (keys.has("r") || keys.has("R")) { mujoco.mj_resetData(model, data); keys.delete("r"); keys.delete("R"); }

  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  if (!paused) {
    acc += dt;
    while (acc >= timestep) {
      applyKeyboardForce();
      mujoco.mj_step(model, data); // step :contentReference[oaicite:4]{index=4}
      acc -= timestep;
    }
  }

  updateMeshes();
  renderer.render(scene, camera);

  statusEl.textContent =
`paused: ${paused}
timestep: ${timestep}
box pos: (${data.xpos[3*boxBodyId+0].toFixed(2)}, ${data.xpos[3*boxBodyId+1].toFixed(2)}, ${data.xpos[3*boxBodyId+2].toFixed(2)})`;
}

loop(performance.now());

addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});
