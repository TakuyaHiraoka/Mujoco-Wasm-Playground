import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// MuJoCo公式WASMバインディング（CDN）
// dist に mujoco_wasm.js がある 
import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";

const UI = {
  status: document.getElementById("status"),
  midiFile: document.getElementById("midiFile"),
  playPause: document.getElementById("playPause"),
  stop: document.getElementById("stop"),
  speed: document.getElementById("speed"),
  speedValue: document.getElementById("speedValue"),
  loop: document.getElementById("loop"),
  audio: document.getElementById("audio"),
};

const FIRST_MIDI = 21; // A0
const N_KEYS = 88;     // A0..C8

// ---------------------------
// 1) Three.js setup
// ---------------------------
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, -1.2, 0.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.15, 0.03);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x222222, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(1.2, -0.6, 2.0);
scene.add(dir);

// ---------------------------
// 2) Piano layout (88 keys)
// ---------------------------
function isBlackMidi(midiNote) {
  const m = midiNote % 12;
  return m === 1 || m === 3 || m === 6 || m === 8 || m === 10;
}

// 物理っぽく、白鍵は等間隔、黒鍵は白鍵の間に配置（簡易）
function buildKeyInfos() {
  const whitePitch = 0.023; // 白鍵中心間隔（m）
  let whiteIndex = 0;

  const infos = [];
  for (let i = 0; i < N_KEYS; i++) {
    const midi = FIRST_MIDI + i;
    const black = isBlackMidi(midi);

    let x;
    if (black) {
      // 直前の白鍵と次の白鍵の間に置く → whiteIndex-0.5
      x = (whiteIndex - 0.5) * whitePitch;
    } else {
      x = whiteIndex * whitePitch;
      whiteIndex += 1;
    }

    // サイズ（half extents は後で作る）
    const width  = black ? 0.014 : 0.022;
    const length = black ? 0.090 : 0.140;
    const height = 0.020;

    infos.push({
      midi,
      black,
      x,
      pivotY: 0.0,
      pivotZ: black ? 0.030 : 0.022,
      width,
      length,
      height,
      // hinge は x軸回り。押し込み角度（負方向に倒す）
      pressAngle: black ? -0.045 : -0.040,
    });
  }

  // 全体を0中心に寄せる（白鍵52本：インデックス0..51）
  const lastWhiteCenterX = 51 * whitePitch;
  const centerX = lastWhiteCenterX / 2;
  for (const k of infos) k.x -= centerX;

  return infos;
}

const keyInfos = buildKeyInfos();

// ---------------------------
// 3) Generate MJCF in JS
// ---------------------------
function mjcfPiano(keys) {
  // base（鍵盤台）
  const xs = keys.map(k => k.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const baseHalfX = (maxX - minX) / 2 + 0.06;
  const baseHalfY = 0.12;
  const baseHalfZ = 0.02;
  const baseZ = 0.0;

  // 衝突計算は不要なので contype/conaffinity=0 で軽量化
  // position actuator を各鍵に付け、ctrl[i] に目標角度を入れる形にする
  let xml = "";
  xml += `<?xml version="1.0"?>\n`;
  xml += `<mujoco model="midi_piano">\n`;
  xml += `  <compiler angle="radian" coordinate="local"/>\n`;
  xml += `  <option timestep="0.005" gravity="0 0 -9.81"/>\n`;
  xml += `  <default>\n`;
  xml += `    <joint type="hinge" damping="1.0" limited="true"/>\n`;
  xml += `    <geom type="box" density="300" contype="0" conaffinity="0"/>\n`;
  xml += `  </default>\n`;
  xml += `  <worldbody>\n`;
  xml += `    <body name="keyboard" pos="0 0 0">\n`;
  xml += `      <geom name="base" type="box" size="${baseHalfX.toFixed(4)} ${baseHalfY.toFixed(4)} ${baseHalfZ.toFixed(4)}" pos="0 ${baseHalfY.toFixed(4)} ${baseZ.toFixed(4)}" rgba="0.25 0.25 0.25 1"/>\n`;

  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const hx = (k.width / 2).toFixed(4);
    const hy = (k.length / 2).toFixed(4);
    const hz = (k.height / 2).toFixed(4);

    const bodyName = `key_${i}`;
    const jointName = `key_joint_${i}`;
    const geomName = `key_geom_${i}`;

    // 白鍵は薄い白、黒鍵は黒
    const rgba = k.black ? "0.1 0.1 0.1 1" : "0.95 0.95 0.95 1";

    xml += `      <body name="${bodyName}" pos="${k.x.toFixed(4)} ${k.pivotY.toFixed(4)} ${k.pivotZ.toFixed(4)}">\n`;
    xml += `        <joint name="${jointName}" axis="1 0 0" range="${k.pressAngle.toFixed(4)} 0"/>\n`;
    // pivot が後端。geom 中心を前方へ hy だけ移動
    xml += `        <geom name="${geomName}" type="box" size="${hx} ${hy} ${hz}" pos="0 ${hy} 0" rgba="${rgba}"/>\n`;
    xml += `      </body>\n`;
  }

  xml += `    </body>\n`;
  xml += `  </worldbody>\n`;

  xml += `  <actuator>\n`;
  for (let i = 0; i < keys.length; i++) {
    const jointName = `key_joint_${i}`;
    // kp/kv は適当な範囲。ctrl に目標角度[radian]を入れる
    xml += `    <position name="act_${i}" joint="${jointName}" kp="120" kv="6"/>\n`;
  }
  xml += `  </actuator>\n`;

  xml += `</mujoco>\n`;
  return xml;
}

// ---------------------------
// 4) MuJoCo init
// ---------------------------
UI.status.textContent = "Loading MuJoCo WASM…";

const MUJOCO_CDN_BASE = "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/";

const mujoco = await load_mujoco({
  // wasm等の補助ファイルを探すときに CDN を指す（安全策）
  locateFile: (path) => `${MUJOCO_CDN_BASE}${path}`,
});

// FS セットアップ（zalo/mujoco_wasm README の典型パターン） 
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

// MJCF を書き込んでロード
const xml = mjcfPiano(keyInfos);
mujoco.FS.writeFile("/working/piano.xml", xml);

const model = mujoco.MjModel.loadFromXML("/working/piano.xml");
const data = new mujoco.MjData(model);
mujoco.mj_forward(model, data);

UI.status.textContent = "Ready (load a MIDI)";

// ---------------------------
// 5) Build Three meshes & mapping
// ---------------------------
const keyMeshes = new Array(N_KEYS);
const keyBodyIds = new Int32Array(N_KEYS);

const matWhite = new THREE.MeshStandardMaterial({ color: 0xf0f0f0 });
const matWhiteOn = new THREE.MeshStandardMaterial({ color: 0x40ff40 });
const matBlack = new THREE.MeshStandardMaterial({ color: 0x111111 });
const matBlackOn = new THREE.MeshStandardMaterial({ color: 0x40ff40 });

const group = new THREE.Group();
scene.add(group);

// base mesh
{
  const xs = keyInfos.map(k => k.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const baseW = (maxX - minX) + 0.12;
  const baseL = 0.24;
  const baseH = 0.04;

  const baseGeo = new THREE.BoxGeometry(baseW, baseL, baseH);
  const baseMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
  const base = new THREE.Mesh(baseGeo, baseMat);
  // MuJoCo側 base は pos=(0, baseHalfY, 0) なのでそれに合わせる
  base.position.set(0, baseL / 2, 0.0);
  group.add(base);

  // カメラの見えを軽く調整
  controls.target.set(0, baseL * 0.6, 0.02);
  controls.update();
}

// name2id の enum
const OBJ_BODY = mujoco.mjtObj.mjOBJ_BODY;

for (let i = 0; i < N_KEYS; i++) {
  const k = keyInfos[i];
  const geo = new THREE.BoxGeometry(k.width, k.length, k.height);
  const mat = k.black ? matBlack : matWhite;
  const mesh = new THREE.Mesh(geo, mat);
  group.add(mesh);
  keyMeshes[i] = mesh;

  const bodyName = `key_${i}`;
  const bodyId = mujoco.mj_name2id(model, OBJ_BODY, bodyName);
  if (bodyId < 0) {
    throw new Error(`Body not found: ${bodyName}`);
  }
  keyBodyIds[i] = bodyId;
}

// 押下状態（色更新最適化）
const keyOn = new Uint8Array(N_KEYS);

// ---------------------------
// 6) MIDI parsing (no npm, no external lib)
// ---------------------------

function readStr(u8, off, len) {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(u8[off + i]);
  return s;
}
function readU16(u8, off) {
  return (u8[off] << 8) | u8[off + 1];
}
function readU32(u8, off) {
  return (u8[off] << 24) | (u8[off + 1] << 16) | (u8[off + 2] << 8) | u8[off + 3];
}
function readVar(u8, off) {
  let v = 0;
  while (true) {
    const b = u8[off++];
    v = (v << 7) | (b & 0x7f);
    if ((b & 0x80) === 0) break;
  }
  return { value: v, off };
}

function parseMidiToEvents(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  let off = 0;

  if (readStr(u8, off, 4) !== "MThd") throw new Error("Invalid MIDI (missing MThd)");
  off += 4;
  const headerLen = readU32(u8, off); off += 4;
  const format = readU16(u8, off); off += 2;
  const nTracks = readU16(u8, off); off += 2;
  const division = readU16(u8, off); off += 2;
  off += (headerLen - 6);

  if (division & 0x8000) {
    throw new Error("SMPTE time division MIDI is not supported in this demo");
  }
  const ticksPerBeat = division;

  // tempo map (tick -> usPerBeat). default 120bpm = 500000 us/beat
  const tempoEvents = [{ tick: 0, usPerBeat: 500000 }];

  // note events collected from all tracks
  const noteEvents = [];
  let globalOrder = 0;

  for (let t = 0; t < nTracks; t++) {
    if (readStr(u8, off, 4) !== "MTrk") throw new Error("Invalid MIDI (missing MTrk)");
    off += 4;
    const trkLen = readU32(u8, off); off += 4;
    const end = off + trkLen;

    let tick = 0;
    let runningStatus = null;

    while (off < end) {
      const dv = readVar(u8, off); tick += dv.value; off = dv.off;

      let statusByte = u8[off++];
      let data1FromRunning = null;

      if (statusByte < 0x80) {
        // running status
        if (runningStatus === null) throw new Error("Running status without previous status");
        data1FromRunning = statusByte;
        statusByte = runningStatus;
      } else {
        // channel status can be used for running
        runningStatus = statusByte;
      }

      if (statusByte === 0xFF) {
        // meta
        const metaType = u8[off++];
        const lv = readVar(u8, off); const len = lv.value; off = lv.off;

        if (metaType === 0x2F) {
          // end of track
          off += len;
          runningStatus = null;
          break;
        }

        if (metaType === 0x51 && len === 3) {
          const usPerBeat = (u8[off] << 16) | (u8[off + 1] << 8) | u8[off + 2];
          tempoEvents.push({ tick, usPerBeat });
        }

        off += len;
        runningStatus = null; // meta後は running status リセット
        continue;
      }

      if (statusByte === 0xF0 || statusByte === 0xF7) {
        // sysex
        const lv = readVar(u8, off); const len = lv.value; off = lv.off;
        off += len;
        runningStatus = null;
        continue;
      }

      const type = statusByte & 0xF0;
      const needs1 = (type === 0xC0 || type === 0xD0) ? 1 : 2;

      let data1 = data1FromRunning;
      if (data1 === null) data1 = u8[off++];

      let data2 = null;
      if (needs1 === 2) data2 = u8[off++];

      // note on/off
      if (type === 0x90 || type === 0x80) {
        const note = data1;
        const vel = data2 ?? 0;
        const on = (type === 0x90) && vel > 0;
        noteEvents.push({
          tick,
          note,
          on,
          order: globalOrder++,
        });
      }
    }

    off = end;
  }

  // tempo sort & normalize
  tempoEvents.sort((a, b) => a.tick - b.tick);

  // build segments: each segment has start tick/time/tempo
  const segments = [];
  let curTempo = tempoEvents[0].usPerBeat;
  let lastTick = tempoEvents[0].tick;
  let curTime = 0;
  segments.push({ tick: lastTick, time: 0, usPerBeat: curTempo });

  for (let i = 1; i < tempoEvents.length; i++) {
    const te = tempoEvents[i];
    if (te.tick < lastTick) continue;
    curTime += ((te.tick - lastTick) * curTempo) / ticksPerBeat / 1e6;
    curTempo = te.usPerBeat;
    lastTick = te.tick;
    segments.push({ tick: lastTick, time: curTime, usPerBeat: curTempo });
  }

  function tickToSec(tick) {
    // linear scan is fine (tempo changes are few)
    let seg = segments[0];
    for (let i = 1; i < segments.length; i++) {
      if (segments[i].tick <= tick) seg = segments[i];
      else break;
    }
    const dtick = tick - seg.tick;
    return seg.time + (dtick * seg.usPerBeat) / ticksPerBeat / 1e6;
  }

  // convert to seconds and sort
  const events = noteEvents.map(e => ({
    time: tickToSec(e.tick),
    note: e.note,
    on: e.on,
    order: e.order,
  }));

  // 先頭の無音を詰める
  const minTime = events.length ? Math.min(...events.map(e => e.time)) : 0;
  for (const e of events) e.time -= minTime;

  events.sort((a, b) => (a.time - b.time) || (a.order - b.order));
  return events;
}

// ---------------------------
// 7) Playback state & (optional) simple audio
// ---------------------------
let events = makeDemoScale(); // 初期はデモ（Cメジャースケール）
let eventIdx = 0;

const noteCounts = new Int16Array(N_KEYS);

// audio
let audioCtx = null;
const activeOsc = new Map(); // midiNote -> {osc, gain}

function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}
function ensureAudio() {
  if (!UI.audio.checked) return null;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function audioNoteOn(midi) {
  const ctx = ensureAudio();
  if (!ctx) return;
  if (activeOsc.has(midi)) return;

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = midiToFreq(midi);

  // 軽いエンベロープ
  const now = ctx.currentTime;
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.06, now + 0.01);

  osc.connect(gain).connect(ctx.destination);
  osc.start();

  activeOsc.set(midi, { osc, gain });
}
function audioNoteOff(midi) {
  const ctx = audioCtx;
  const node = activeOsc.get(midi);
  if (!ctx || !node) return;

  const now = ctx.currentTime;
  node.gain.gain.cancelScheduledValues(now);
  node.gain.gain.setValueAtTime(Math.max(node.gain.gain.value, 0.0001), now);
  node.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.03);
  node.osc.stop(now + 0.04);

  activeOsc.delete(midi);
}

function stopAllAudio() {
  for (const [midi] of activeOsc) audioNoteOff(midi);
}

// ---------------------------
// 8) Controls
// ---------------------------
let playing = false;

UI.speed.addEventListener("input", () => {
  UI.speedValue.textContent = `${Number(UI.speed.value).toFixed(2)}x`;
});

UI.playPause.addEventListener("click", async () => {
  // ユーザージェスチャーで audio context を resume できる
  if (UI.audio.checked) {
    const ctx = ensureAudio();
    if (ctx && ctx.state === "suspended") await ctx.resume();
  }

  playing = !playing;
  UI.playPause.textContent = playing ? "⏸ Pause" : "▶︎ Play";
});

UI.stop.addEventListener("click", () => {
  playing = false;
  UI.playPause.textContent = "▶︎ Play";
  resetPlayback(true);
});

UI.midiFile.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  try {
    UI.status.textContent = `Parsing MIDI… (${file.name})`;
    const buf = await file.arrayBuffer();
    events = parseMidiToEvents(buf);
    UI.status.textContent = `Loaded: ${file.name} / events=${events.length}`;
    playing = true;
    UI.playPause.textContent = "⏸ Pause";
    resetPlayback(true);
  } catch (err) {
    console.error(err);
    UI.status.textContent = `MIDI parse error: ${err?.message ?? err}`;
  }
});

// ---------------------------
// 9) Demo song (no MIDI file needed)
// ---------------------------
function makeDemoScale() {
  // C major scale: C4..C5
  const notes = [60, 62, 64, 65, 67, 69, 71, 72];
  const dur = 0.22;
  const gap = 0.02;
  let t = 0;
  const ev = [];
  let order = 0;
  for (const n of notes) {
    ev.push({ time: t, note: n, on: true, order: order++ });
    ev.push({ time: t + dur, note: n, on: false, order: order++ });
    t += dur + gap;
  }
  return ev;
}

// ---------------------------
// 10) Simulation loop
// ---------------------------
function resetPlayback(resetSim) {
  eventIdx = 0;
  noteCounts.fill(0);
  keyOn.fill(0);
  stopAllAudio();

  if (resetSim) {
    mujoco.mj_resetData(model, data);
    mujoco.mj_forward(model, data);
  }

  // ctrl を初期化
  for (let i = 0; i < N_KEYS; i++) data.ctrl[i] = 0;
}

resetPlayback(true);

function stepSimulation(realDt) {
  const speed = Number(UI.speed.value);
  const dt = realDt * speed;

  const simDt = model.opt.timestep; // 例: 0.005 
  let steps = Math.floor(dt / simDt);
  steps = Math.min(steps, 2000); // タブ復帰などで暴走しないように

  for (let s = 0; s < steps; s++) {
    const t = data.time;

    // MIDIイベントを時間順に適用
    while (eventIdx < events.length && events[eventIdx].time <= t) {
      const ev = events[eventIdx++];

      const keyIndex = ev.note - FIRST_MIDI;
      if (keyIndex >= 0 && keyIndex < N_KEYS) {
        const prev = noteCounts[keyIndex];

        if (ev.on) noteCounts[keyIndex] = prev + 1;
        else noteCounts[keyIndex] = Math.max(0, prev - 1);

        // audio: 0→1 で on、1→0 で off
        const nowc = noteCounts[keyIndex];
        if (prev === 0 && nowc === 1) audioNoteOn(ev.note);
        if (prev === 1 && nowc === 0) audioNoteOff(ev.note);
      }
    }

    // ctrl を更新（position actuator: ctrl=目標角度）
    for (let i = 0; i < N_KEYS; i++) {
      const on = noteCounts[i] > 0;
      data.ctrl[i] = on ? keyInfos[i].pressAngle : 0;
    }

    mujoco.mj_step(model, data);

    // 曲終端
    if (eventIdx >= events.length) {
      if (UI.loop.checked) {
        // ループ
        resetPlayback(true);
      } else {
        playing = false;
        UI.playPause.textContent = "▶︎ Play";
        break;
      }
    }
  }
}

function syncMeshesFromPhysics() {
  // data.xpos: nbody*3, data.xquat: nbody*4 (quat は w,x,y,z) 
  const xpos = data.xpos;
  const xquat = data.xquat;

  for (let i = 0; i < N_KEYS; i++) {
    const b = keyBodyIds[i];
    const mesh = keyMeshes[i];

    const px = xpos[b * 3 + 0];
    const py = xpos[b * 3 + 1];
    const pz = xpos[b * 3 + 2];

    const qw = xquat[b * 4 + 0];
    const qx = xquat[b * 4 + 1];
    const qy = xquat[b * 4 + 2];
    const qz = xquat[b * 4 + 3];

    mesh.position.set(px, py, pz);
    mesh.quaternion.set(qx, qy, qz, qw);
  }

  // 押下状態で色更新（変化があったキーだけ）
  for (let i = 0; i < N_KEYS; i++) {
    const on = noteCounts[i] > 0 ? 1 : 0;
    if (on === keyOn[i]) continue;
    keyOn[i] = on;

    const black = keyInfos[i].black;
    keyMeshes[i].material = black ? (on ? matBlackOn : matBlack) : (on ? matWhiteOn : matWhite);
  }
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let last = performance.now();

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const realDt = (now - last) / 1000;
  last = now;

  if (playing) {
    stepSimulation(realDt);
  }

  syncMeshesFromPhysics();
  controls.update();
  renderer.render(scene, camera);
}

animate();
