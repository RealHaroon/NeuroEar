import { putColor } from './colormap.js';
import { formatShape } from './network3d.js';

/**
 * Plain 2D view of every layer's activations (press D). Useful as a sanity check that the
 * activation model really sees the same thing as the recognizer.
 */
export class Layers2D {
  constructor(root, layers) {
    this.root = root;
    root.innerHTML = '<div class="l2d-head"><b>All layers · 2D debug view</b> <span id="l2d-verify"></span> <span class="dim">press D to close</span></div><div class="l2d-strip"></div>';
    this.verifyEl = root.querySelector('#l2d-verify');
    const strip = root.querySelector('.l2d-strip');
    this.cells = layers.map((L) => {
      const cell = document.createElement('div');
      cell.className = 'l2d-cell';
      cell.innerHTML = `<canvas></canvas><div class="l2d-name">${L.name}</div><div class="l2d-shape">${L.type} · ${formatShape(L.shape)}</div>`;
      strip.appendChild(cell);
      return { L, canvas: cell.querySelector('canvas') };
    });
  }

  toggle() {
    this.root.classList.toggle('hidden');
  }

  update(acts, maxDiff) {
    this.verifyEl.textContent = `final layer vs recognizer scores: max |Δ| = ${maxDiff.toExponential(2)} ${maxDiff < 1e-4 ? '✓' : '✗'}`;
    this.cells.forEach(({ L, canvas }, i) => {
      const act = acts[i];
      const p = L.plan;
      const g = canvas.getContext('2d');
      if (p.rank === 4) {
        const k = L.isInput ? 1 : p.k;
        canvas.width = k * (p.h + 1);
        canvas.height = p.w;
        const img = g.createImageData(canvas.width, canvas.height);
        const scale = act.max > 0 ? 1 / act.max : 0;
        for (let c = 0; c < k; c++) {
          for (let f = 0; f < p.h; f++) {
            for (let b = 0; b < p.w; b++) {
              const v = act.data[(c * p.h + f) * p.w + b];
              putColor(img.data, ((p.w - 1 - b) * canvas.width + c * (p.h + 1) + f) * 4, L.isInput ? (v + 1.5) / 4.5 : Math.sqrt(v * scale));
            }
          }
        }
        g.putImageData(img, 0, 0);
      } else {
        const n = act.data.length;
        canvas.width = Math.max(n, 40);
        canvas.height = 60;
        g.clearRect(0, 0, canvas.width, canvas.height);
        const max = L.isOutput ? 1 : act.max || 1;
        g.fillStyle = L.isOutput ? '#ff8f52' : '#8db4ff';
        const bw = canvas.width / n;
        for (let j = 0; j < n; j++) {
          const hgt = (act.data[j] / max) * canvas.height;
          g.fillRect(j * bw, canvas.height - hgt, Math.max(1, bw - 0.5), hgt);
        }
      }
    });
  }
}
