import { decodeClip, encodeWav } from './audio.js';

const CAN_SAVE = import.meta.env.DEV; // saving needs the dev server (npm run dev)

/** Backup demo clips in public/demo: play them through the pipeline (keys 1–9), save new ones. */
export class DemoClips {
  constructor(root, { onPlay }) {
    this.root = root;
    this.onPlay = onPlay;
    this.files = [];
    root.addEventListener('click', (e) => {
      const play = e.target.closest('[data-play]');
      const del = e.target.closest('[data-delete]');
      if (del) this.remove(del.dataset.delete);
      else if (play) this.play(Number(play.dataset.play));
    });
  }

  async load() {
    try {
      const res = await fetch(`demo/index.json?t=${Date.now()}`);
      this.files = res.ok ? await res.json() : [];
    } catch {
      this.files = [];
    }
    this.render();
  }

  render() {
    if (!this.files.length) {
      this.root.innerHTML = `<div class="demo-empty">${CAN_SAVE ? 'No backup clips yet: record a word, then 💾 save it.' : 'No backup clips in public/demo.'}</div>`;
      return;
    }
    this.root.innerHTML = this.files
      .map(
        (f, i) =>
          `<span class="demo-chip"><button data-play="${i}" title="play ${f}">${i < 9 ? `<kbd>${i + 1}</kbd>` : ''}${f.replace(/\.wav$/, '')}</button>` +
          (CAN_SAVE ? `<button class="demo-del" data-delete="${f}" title="delete ${f}">×</button>` : '') +
          '</span>'
      )
      .join('');
  }

  async play(i) {
    const file = this.files[i];
    if (!file) return;
    try {
      const res = await fetch(`demo/${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.onPlay(await decodeClip(await res.arrayBuffer()), file);
    } catch (err) {
      console.error(err);
      this.root.insertAdjacentHTML('beforeend', `<div class="demo-empty">Could not play ${file}: ${err.message}</div>`);
    }
  }

  /** Save raw 44.1 kHz samples as public/demo/<name>.wav (dev server only). */
  async save(samples, name) {
    if (!CAN_SAVE) throw new Error('Saving clips needs the dev server (npm run dev).');
    const res = await fetch(`/__neuroear/save-clip?name=${encodeURIComponent(name)}`, { method: 'POST', body: encodeWav(samples) });
    if (!res.ok) throw new Error(`save failed: HTTP ${res.status}`);
    const { file, files } = await res.json();
    this.files = files;
    this.render();
    return file;
  }

  async remove(file) {
    if (!confirm(`Delete backup clip ${file}?`)) return;
    const res = await fetch(`/__neuroear/delete-clip?file=${encodeURIComponent(file)}`, { method: 'POST' });
    if (res.ok) this.files = (await res.json()).files;
    this.render();
  }
}

export const canSaveClips = CAN_SAVE;
