import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { LUT, lutIndex, putColor } from './colormap.js';
import { prettyLabel } from './recognizer.js';

const BG = 0x0f1626;
const BLUE = new THREE.Color('#8db4ff');
const ORANGE = new THREE.Color('#ff8f52');
const STEP = 0.17; // seconds for the signal wave to travel one layer
const DIM = 0.14; // brightness of non-focused layers in presenter mode
const GAP = 1.5; // world units between layers
const LINK_BASE = 0.1; // resting opacity of connection lines

const SHORT_TYPE = { Conv2D: 'Conv', MaxPooling2D: 'Pool', Dense: 'Dense', Flatten: 'Flatten', Dropout: 'Dropout', Input: 'Input' };

export const formatShape = (shape) => shape.slice(1).join(' × ');

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function rng(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Linear-space color from a LUT entry (LUT holds sRGB bytes). */
function lutColor(t, out) {
  const i = lutIndex(t);
  return out.setRGB(LUT[i] / 255, LUT[i + 1] / 255, LUT[i + 2] / 255, THREE.SRGBColorSpace);
}

/**
 * 3D view of the network, generated from the model's actual layers:
 * 4D outputs → grids of feature-map tiles, Dropout → pass-through gate, other → spheres.
 */
export class Network3D {
  constructor(container, layers, words, tooltip) {
    this.container = container;
    this.layers = layers;
    this.words = words;
    this.tooltip = tooltip;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor(BG);
    container.appendChild(this.renderer.domElement);
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = 'labels3d';
    container.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 500);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener('start', () => {
      this.tween = null;
      this.userMoved = true;
    });

    this.glowTex = glowTexture();
    this.tmpColor = new THREE.Color();
    this.tmpMatrix = new THREE.Matrix4();
    this.build();

    this.lastFrame = performance.now();
    this.wave = null;
    this.focusIndex = null;
    this.hover = -1;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    // Mouse: hover shows layer details. Touch: a tap shows them (a drag rotates instead), and
    // touch pointers "leave" as soon as the finger lifts, so leave only counts for the mouse.
    const el = this.renderer.domElement;
    el.addEventListener('pointermove', (e) => e.pointerType === 'mouse' && this.onPointer(e));
    el.addEventListener('pointerleave', (e) => e.pointerType === 'mouse' && this.setHover(-1));
    el.addEventListener('pointerdown', (e) => (this.tapStart = { x: e.clientX, y: e.clientY }));
    el.addEventListener('pointerup', (e) => {
      if (e.pointerType === 'mouse' || !this.tapStart) return;
      if (Math.hypot(e.clientX - this.tapStart.x, e.clientY - this.tapStart.y) < 10) this.onPointer(e);
    });
    this.renderer.domElement.addEventListener('dblclick', () => this.goHome());

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.goHome(true);
    this.renderer.setAnimationLoop(() => this.frame());
  }

  // ---------------------------------------------------------------- building

  build() {
    const inputW = this.layers[0].plan.W;
    this.vis = [];
    this.hitBoxes = [];
    let x = 0;
    this.layers.forEach((L, i) => {
      const v = { L, i, group: new THREE.Group(), flash: 0, dim: 1, dimTarget: 1, labelOpacity: 1 };
      if (L.plan.rank === 4) this.buildMaps(v, inputW);
      else if (L.type === 'Dropout') this.buildGate(v, this.vis[i - 1]);
      else this.buildUnits(v);
      v.x = x + v.width / 2;
      x += v.width + GAP;
      this.vis.push(v);
    });
    this.totalWidth = x - GAP;
    this.maxHeight = Math.max(...this.vis.map((v) => v.height));
    for (const v of this.vis) {
      v.group.position.x = v.x - this.totalWidth / 2;
      this.scene.add(v.group);
      this.addDecorations(v);
    }
    for (let i = 0; i < this.vis.length - 1; i++) this.buildLink(this.vis[i], this.vis[i + 1], i);

    const floorY = -this.maxHeight / 2 - 1.6;
    const grid = new THREE.GridHelper(Math.ceil(this.totalWidth + 12), 40, 0x8db4ff, 0x8db4ff);
    grid.material.transparent = true;
    grid.material.opacity = 0.07;
    grid.position.y = floorY;
    this.scene.add(grid);
  }

  buildMaps(v, inputW) {
    const { plan, isInput } = v.L;
    const k = isInput ? 1 : plan.k;
    // One column of channels per layer (two if there are more than 8), slightly fanned in depth.
    const cols = k > 8 ? 2 : 1;
    const rows = Math.ceil(k / cols);
    const gap = 0.14;
    const zStep = 0.45;
    // Taller layers for more frequency bins, so you can watch pooling shrink the maps.
    const height = isInput ? 11 : 6 + 7 * Math.sqrt(plan.W / inputW);
    const tileH = (height - gap * (rows - 1)) / rows;
    const tileW = tileH * THREE.MathUtils.clamp(plan.H / plan.W, isInput ? 0.32 : 0.55, 1.4);

    v.kind = 'maps';
    v.tiles = [];
    v.frameMat = new THREE.LineBasicMaterial({ color: BLUE, transparent: true, opacity: 0.5 });
    const plane = new THREE.PlaneGeometry(tileW, tileH);
    const edges = new THREE.EdgesGeometry(new THREE.PlaneGeometry(tileW + 0.05, tileH + 0.05));
    v.anchorsIn = [];
    v.anchorsOut = [];
    for (let i = 0; i < k; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      const data = new Uint8Array(plan.h * plan.w * 4);
      for (let p = 0; p < data.length; p += 4) putColor(data, p, 0);
      // Texture x = time (frames), y = frequency (bins, low at the bottom).
      const tex = new THREE.DataTexture(data, plan.h, plan.w, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.LinearFilter;
      tex.needsUpdate = true;
      const mat = new THREE.MeshBasicMaterial({ map: tex });
      const mesh = new THREE.Mesh(plane, mat);
      mesh.position.set((c - (cols - 1) / 2) * (tileW + gap), ((rows - 1) / 2 - r) * (tileH + gap), -c * zStep - (r % 2) * 0.12);
      mesh.add(new THREE.LineSegments(edges, v.frameMat));
      v.group.add(mesh);
      v.tiles.push({ mesh, tex, mat });
      v.anchorsIn.push(mesh.position.clone().add(new THREE.Vector3(-tileW / 2, 0, 0)));
      v.anchorsOut.push(mesh.position.clone().add(new THREE.Vector3(tileW / 2, 0, 0)));
    }
    v.width = cols * tileW + (cols - 1) * gap;
    v.height = height;
    v.depth = (cols - 1) * zStep + 0.12;
  }

  buildGate(v, prev) {
    const h = prev.height * 0.85;
    const geo = new THREE.PlaneGeometry(0.5, h);
    v.kind = 'gate';
    v.gateMat = new THREE.MeshBasicMaterial({ color: BLUE, transparent: true, opacity: 0.06, depthWrite: false, side: THREE.DoubleSide });
    v.group.add(new THREE.Mesh(geo, v.gateMat));
    v.frameMat = new THREE.LineDashedMaterial({ color: BLUE, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.5 });
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(geo), v.frameMat);
    frame.computeLineDistances();
    v.group.add(frame);
    v.anchorsIn = [];
    for (let i = 0; i < 6; i++) v.anchorsIn.push(new THREE.Vector3(0, (i / 5 - 0.5) * h * 0.8, 0));
    v.anchorsOut = v.anchorsIn;
    v.width = 0.5;
    v.height = h;
    v.depth = 0;
  }

  buildUnits(v) {
    const { plan, isOutput, type } = v.L;
    const n = plan.count;
    const radius = isOutput ? 0.2 : type === 'Flatten' ? 0.08 : 0.11;
    const pos = [];
    if (n <= 24) {
      const sp = Math.min(0.55, 11 / n);
      for (let i = 0; i < n; i++) pos.push(new THREE.Vector3(0, ((n - 1) / 2 - i) * sp, 0));
      v.width = 0.5;
      v.height = n * sp;
      v.depth = 0;
    } else {
      // A 3D block of neurons: rows (y) × depth layers (z) × columns (x).
      const sp = type === 'Flatten' ? 0.3 : 0.36;
      const rows = Math.min(n, 30);
      const zl = Math.min(Math.ceil(n / rows), 5);
      const xl = Math.ceil(n / (rows * zl));
      for (let i = 0; i < n; i++) {
        const r = i % rows, rest = Math.floor(i / rows), z = rest % zl, xc = Math.floor(rest / zl);
        pos.push(new THREE.Vector3((xc - (xl - 1) / 2) * sp, ((rows - 1) / 2 - r) * sp, -(z - (zl - 1) / 2) * sp));
      }
      v.width = xl * sp;
      v.height = rows * sp;
      v.depth = zl * sp;
    }
    v.kind = 'units';
    v.radius = radius;
    v.unitPos = pos;
    v.norm = new Float32Array(n);
    v.base = new Float32Array(n * 3); // linear RGB per unit, for the glow
    v.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }), n);
    v.group.add(v.mesh);

    const gGeo = new THREE.BufferGeometry();
    gGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos.flatMap((p) => [p.x, p.y, p.z]), 3));
    gGeo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    v.glow = new THREE.Points(
      gGeo,
      new THREE.PointsMaterial({ size: radius * 8, map: this.glowTex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    v.group.add(v.glow);
    this.setUnits(v, new Float32Array(n), 1);

    if (isOutput) {
      v.unitLabels = pos.map((p, i) => {
        const el = document.createElement('div');
        el.className = 'unit-label';
        el.style.display = 'none';
        const obj = new CSS2DObject(el);
        obj.position.set(p.x + radius + 0.12, p.y, p.z);
        v.group.add(obj);
        return el;
      });
    }
    v.anchorsIn = pos;
    v.anchorsOut = pos;
  }

  addDecorations(v) {
    const { L } = v;
    const el = document.createElement('div');
    el.className = 'layer-label';
    const sub = L.isOutput && L.cfg.activation === 'softmax' ? 'softmax' : L.type === 'Dropout' ? '' : formatShape(L.shape);
    el.innerHTML = `<b>${SHORT_TYPE[L.type] ?? L.type}</b><span>${sub}</span>`;
    const label = new CSS2DObject(el);
    // Dropout gates sit between narrow layers, so their label goes on top to avoid collisions.
    label.position.set(0, v.kind === 'gate' ? v.height / 2 + 0.45 : -v.height / 2 - 0.55, 0);
    v.group.add(label);
    v.labelEl = el;

    v.halo = new THREE.Mesh(
      new THREE.PlaneGeometry(v.width + 2.4, v.height + 2.4),
      new THREE.MeshBasicMaterial({ map: this.glowTex, color: ORANGE, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    v.halo.position.z = -v.depth - 0.7;
    v.group.add(v.halo);

    const hit = new THREE.Mesh(
      new THREE.BoxGeometry(v.width + 0.5, v.height + 1.2, v.depth + 0.8),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false })
    );
    hit.position.set(0, -0.3, -v.depth / 2);
    hit.userData.i = v.i;
    v.group.add(hit);
    this.hitBoxes.push(hit);
  }

  /** Faint lines from layer a to b, plus pulse particles that ride them during the wave. */
  buildLink(a, b, seed) {
    const rand = rng(seed + 7);
    const pa = a.anchorsOut.map((p) => p.clone().add(a.group.position));
    const pb = b.anchorsIn.map((p) => p.clone().add(b.group.position));
    const pairs = [];
    if (pa.length * pb.length <= 64) {
      for (const p of pa) for (const q of pb) pairs.push([p, q]);
    } else {
      for (let i = 0; i < 56; i++) pairs.push([pa[(rand() * pa.length) | 0], pb[(rand() * pb.length) | 0]]);
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pairs.flat());
    const mat = new THREE.LineBasicMaterial({ color: BLUE, transparent: true, opacity: LINK_BASE, blending: THREE.AdditiveBlending, depthWrite: false });
    const lines = new THREE.LineSegments(geo, mat);
    this.scene.add(lines);

    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(pairs.length * 3), 3));
    const pulse = new THREE.Points(
      pGeo,
      new THREE.PointsMaterial({ size: 0.42, map: this.glowTex, color: ORANGE, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    pulse.visible = false;
    this.scene.add(pulse);
    a.link = { pairs, lines, pulse, active: 0 };
  }

  // ---------------------------------------------------------------- activations

  applyLayer(i, act) {
    const v = this.vis[i];
    if (!act) return;
    v.channels = act.channels;
    if (v.kind === 'maps') {
      const { h, w } = v.L.plan;
      const isInput = v.L.isInput;
      const scale = act.max > 0 ? 1 / act.max : 0;
      v.tiles.forEach((tile, c) => {
        const buf = tile.tex.image.data;
        const base = c * h * w;
        for (let f = 0; f < h; f++) {
          for (let b = 0; b < w; b++) {
            const val = act.data[base + f * w + b];
            putColor(buf, (b * h + f) * 4, isInput ? (val + 1.5) / 4.5 : Math.sqrt(val * scale));
          }
        }
        tile.tex.needsUpdate = true;
      });
    } else if (v.kind === 'units') {
      this.setUnits(v, act.data, v.L.isOutput ? 1 : act.max);
      if (v.unitLabels) {
        // Label only the three most likely words; the Output panel lists them all.
        const top = [...act.data.keys()].sort((a, b) => act.data[b] - act.data[a]).slice(0, 3);
        v.unitLabels.forEach((el, k) => {
          const rank = top.indexOf(k);
          el.style.display = rank < 0 ? 'none' : '';
          el.classList.toggle('top', rank === 0);
          if (rank >= 0) el.textContent = `${prettyLabel(this.words[k] ?? String(k))} ${Math.round(act.data[k] * 100)}%`;
        });
      }
    }
  }

  setUnits(v, values, max) {
    const big = v.L.isOutput ? 1.7 : 0.9;
    for (let k = 0; k < v.unitPos.length; k++) {
      const a = max > 0 ? (v.L.isOutput ? values[k] : Math.sqrt(Math.max(0, values[k]) / max)) : 0;
      v.norm[k] = a;
      const s = v.radius * (0.45 + big * a);
      this.tmpMatrix.makeScale(s, s, s).setPosition(v.unitPos[k]);
      v.mesh.setMatrixAt(k, this.tmpMatrix);
      lutColor(0.3 + 0.7 * a, this.tmpColor);
      v.mesh.setColorAt(k, this.tmpColor);
      v.base[k * 3] = this.tmpColor.r;
      v.base[k * 3 + 1] = this.tmpColor.g;
      v.base[k * 3 + 2] = this.tmpColor.b;
    }
    v.mesh.instanceMatrix.needsUpdate = true;
    v.mesh.instanceColor.needsUpdate = true;
  }

  /** Signal wave from input to output; each layer takes its new activations when the wave arrives. */
  play(acts) {
    this.wave = { start: performance.now(), from: 0, to: this.vis.length - 1, acts, applied: [] };
  }

  /** Presenter mode: spotlight one layer (null = everything back to normal). */
  focus(index) {
    this.focusIndex = index;
    for (const v of this.vis) v.dimTarget = index == null || v.i === index ? 1 : DIM;
    if (index == null) {
      this.goHome();
      return;
    }
    this.wave = { start: performance.now(), from: Math.max(0, index - 1), to: index, acts: null, applied: [] };
    // Frame the focused layer together with its neighbours.
    const near = this.vis.slice(Math.max(0, index - 1), index + 2);
    const box = this.boxOf(near.length > 2 ? near : [this.vis[index]], 0.6);
    box.union(this.boxOf([this.vis[index]], 2.5));
    const { pos, target } = this.fitView(box, new THREE.Vector3(-0.42, 0.2, 1), 0.82);
    this.tweenTo(pos, target);
  }

  /** Bounding box of some layers (plus room for their labels underneath). */
  boxOf(list, pad = 0) {
    const box = new THREE.Box3();
    for (const v of list) {
      const x = v.group.position.x;
      box.expandByPoint(new THREE.Vector3(x - v.width / 2 - pad, -v.height / 2 - 1.3, -v.depth - pad));
      box.expandByPoint(new THREE.Vector3(x + v.width / 2 + pad, v.height / 2 + pad * 0.5, pad));
    }
    return box;
  }

  /** Camera distance along `dir` at which every corner of `box` lands inside the viewport. */
  fitView(box, dir, margin) {
    dir = dir.clone().normalize();
    const target = box.getCenter(new THREE.Vector3());
    const cam = this.camera.clone();
    const corners = [];
    for (let i = 0; i < 8; i++) {
      corners.push(new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
    }
    let d = box.getSize(new THREE.Vector3()).length() * 1.5;
    for (let it = 0; it < 8; it++) {
      cam.position.copy(target).addScaledVector(dir, d);
      cam.lookAt(target);
      cam.updateMatrixWorld();
      let m = 0;
      for (const c of corners) {
        const p = c.clone().project(cam);
        m = Math.max(m, Math.abs(p.x), Math.abs(p.y));
      }
      d *= m / margin;
    }
    return { pos: cam.position.clone(), target };
  }

  homeView() {
    return this.fitView(this.boxOf(this.vis), new THREE.Vector3(-0.5, 0.2, 1), 0.96);
  }

  goHome(instant = false) {
    const { pos, target } = this.homeView();
    this.userMoved = false;
    if (instant) {
      this.tween = null;
      this.camera.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
    } else {
      this.tweenTo(pos, target);
    }
  }

  tweenTo(pos, target) {
    this.tween = {
      start: performance.now(),
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      pos,
      target,
    };
  }

  // ---------------------------------------------------------------- interaction

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.userMoved && this.focusIndex == null) this.goHome(true);
  }

  onPointer(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.hitBoxes, false)[0];
    this.setHover(hit ? hit.object.userData.i : -1, e.clientX - rect.left, e.clientY - rect.top, rect);
  }

  setHover(i, x, y, rect) {
    this.hover = i;
    const tt = this.tooltip;
    if (i < 0) {
      tt.classList.add('hidden');
      return;
    }
    tt.innerHTML = this.tooltipHTML(this.vis[i]);
    tt.classList.remove('hidden');
    const left = Math.min(x + 18, rect.width - tt.offsetWidth - 8);
    const top = Math.min(y + 18, rect.height - tt.offsetHeight - 8);
    tt.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
  }

  tooltipHTML(v) {
    const { L } = v;
    const c = L.cfg;
    const rows = [];
    const act = c.activation && c.activation !== 'linear' ? ` · ${c.activation}` : '';
    rows.push(`<div class="tt-title">${L.name}</div><div class="tt-type">${L.type}${act}</div>`);
    rows.push(`<div>output shape <b>${formatShape(L.shape)}</b></div>`);
    if (L.isInput) rows.push('<div>time frames × frequency bins × 1<br>the spectrogram the network sees</div>');
    if (L.type === 'Conv2D') rows.push(`<div>${c.filters} filters · kernel ${c.kernelSize.join('×')} · stride ${c.strides.join('×')}</div>`);
    if (L.type === 'MaxPooling2D') rows.push(`<div>pool ${c.poolSize.join('×')} · stride ${c.strides.join('×')}</div>`);
    if (L.type === 'Dense') rows.push(`<div>${c.units} neurons, each connected to every input</div>`);
    if (L.type === 'Dropout') rows.push(`<div>rate ${c.rate} · only active while training</div>`);
    if (L.params) rows.push(`<div>${L.params.toLocaleString()} parameters (learned weights)</div>`);
    const p = L.plan;
    if (p.rank === 4 && p.C > p.k) rows.push(`<div class="tt-note">showing the ${p.k} most active of ${p.C} channels${v.channels ? ` (#${[...v.channels].join(', #')})` : ''}</div>`);
    if (p.rank === 2 && p.group > 1 && v.kind === 'units') rows.push(`<div class="tt-note">${p.count} spheres · each = max of ${p.group} neurons</div>`);
    return rows.join('');
  }

  // ---------------------------------------------------------------- animation loop

  frame() {
    const now = performance.now();
    const dt = Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;

    for (const v of this.vis) if (v.link) v.link.active = 0;
    if (this.wave) {
      const w = this.wave;
      const t = (now - w.start) / 1000;
      for (let i = w.from; i <= w.to; i++) {
        if (!w.applied[i] && t >= (i - w.from) * STEP) {
          w.applied[i] = true;
          this.applyLayer(i, w.acts?.[i]);
          this.vis[i].flash = 1;
        }
      }
      for (let g = w.from; g < w.to; g++) this.updatePulse(this.vis[g].link, (t - (g - w.from) * STEP) / STEP);
      if (t > (w.to - w.from) * STEP + 0.4) this.wave = null;
    }

    const k = Math.min(1, dt * 6);
    for (const v of this.vis) {
      v.flash *= Math.exp(-dt * 2.2);
      v.dim += (v.dimTarget - v.dim) * k;
      this.updateLayerLook(v);
      if (v.link) {
        const next = this.vis[v.i + 1];
        const d = Math.min(v.dim, next.dim);
        v.link.lines.material.opacity = (LINK_BASE + 0.45 * v.link.active) * d;
        v.link.pulse.visible = v.link.active > 0;
        v.link.pulse.material.opacity = d;
      }
    }

    if (this.tween) {
      const p = Math.min(1, (now - this.tween.start) / 900);
      const e = easeInOut(p);
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.pos, e);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.target, e);
      if (p >= 1) this.tween = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }

  updatePulse(link, p) {
    if (p < 0 || p > 1) return;
    link.active = 1 - Math.abs(p - 0.5) * 2;
    const arr = link.pulse.geometry.attributes.position.array;
    link.pairs.forEach(([a, b], j) => {
      arr[j * 3] = a.x + (b.x - a.x) * p;
      arr[j * 3 + 1] = a.y + (b.y - a.y) * p;
      arr[j * 3 + 2] = a.z + (b.z - a.z) * p;
    });
    link.pulse.geometry.attributes.position.needsUpdate = true;
  }

  updateLayerLook(v) {
    const { flash, dim } = v;
    const hovered = this.hover === v.i;
    const focused = this.focusIndex === v.i;
    v.halo.material.opacity = (0.55 * flash + (focused ? 0.3 : 0) + (hovered ? 0.12 : 0)) * dim;
    if (v.frameMat) {
      v.frameMat.color.copy(BLUE).lerp(ORANGE, Math.min(1, flash + (focused ? 0.6 : 0)));
      v.frameMat.opacity = Math.min(1, 0.35 + 0.65 * flash + (hovered ? 0.4 : 0)) * dim;
    }
    if (v.kind === 'maps') {
      for (const t of v.tiles) t.mat.color.setScalar(dim * (1 + 0.6 * flash));
    } else if (v.kind === 'gate') {
      v.gateMat.opacity = (0.05 + 0.3 * flash) * dim;
    } else {
      v.mesh.material.color.setScalar(dim * (1 + 0.5 * flash));
      const col = v.glow.geometry.attributes.color.array;
      for (let k = 0; k < v.norm.length; k++) {
        const s = Math.pow(v.norm[k], 1.5) * (0.35 + 1.4 * flash) * dim;
        col[k * 3] = v.base[k * 3] * s;
        col[k * 3 + 1] = v.base[k * 3 + 1] * s;
        col[k * 3 + 2] = v.base[k * 3 + 2] * s;
      }
      v.glow.geometry.attributes.color.needsUpdate = true;
    }
    const op = 0.3 + 0.7 * dim;
    if (Math.abs(op - v.labelOpacity) > 0.01) {
      v.labelOpacity = op;
      v.labelEl.style.opacity = op;
      v.unitLabels?.forEach((el) => (el.style.opacity = op));
    }
  }
}
