import { LUT, lutIndex } from './colormap.js';
import { createSpectrum } from './spectrum.js';
import workletUrl from './recorder-worklet.js?url';

// Same features as speech-commands' BrowserFftFeatureExtractor: 44.1 kHz audio, an analyser-style
// spectrum over 2048 samples (no smoothing), one frame every 1024 samples, first 232 bins (≈ 0–5 kHz).
// Frames are computed from raw samples captured on the audio thread, so none are ever dropped.
export const SAMPLE_RATE = 44100;
const FRAME_SIZE = 1024;
const FFT_SIZE = FRAME_SIZE * 2;
const DB_FLOOR = -160; // digital silence would be -Infinity

/** Spectrogram frame i: the 2048 samples ending at 2048 + i·1024. */
function frameAt(spectrum, samples, i, bins) {
  const frame = spectrum(samples, FFT_SIZE + i * FRAME_SIZE, bins);
  for (let b = 0; b < frame.length; b++) if (!(frame[b] > DB_FLOOR)) frame[b] = DB_FLOOR;
  return frame;
}

/**
 * Turn a whole clip into what a live recording of it would have produced:
 * { frames, samples } with samples aligned to the frames (see Recorder.stop()).
 */
export function framesFromSamples(samples, bins, maxSeconds) {
  const spectrum = createSpectrum(FFT_SIZE);
  const maxFrames = Math.round((maxSeconds * SAMPLE_RATE) / FRAME_SIZE);
  const frames = [];
  while (frames.length < maxFrames && FFT_SIZE + frames.length * FRAME_SIZE <= samples.length) {
    frames.push(frameAt(spectrum, samples, frames.length, bins));
  }
  return { frames, samples: samples.slice(FRAME_SIZE, FRAME_SIZE + frames.length * FRAME_SIZE) };
}

/** Decode any audio file and resample it to mono 44.1 kHz. */
export async function decodeClip(arrayBuffer) {
  const tmp = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await tmp.decodeAudioData(arrayBuffer);
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

/** 16-bit mono WAV file from 44.1 kHz float samples. */
export function encodeWav(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, SAMPLE_RATE, true);
  v.setUint32(28, SAMPLE_RATE * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
  return new Blob([buf], { type: 'audio/wav' });
}

/** Keeps a canvas' backing store matched to its CSS size (crisp on HiDPI / projector). */
function fitCanvas(canvas, onResize) {
  const ro = new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    onResize?.();
  });
  ro.observe(canvas);
}

/**
 * Streaming linear-interpolation resampler (e.g. 48 kHz → 44.1 kHz). Only used when the browser
 * won't run a 44.1 kHz audio context with the microphone; the model only looks at 0–5 kHz, far
 * below where linear interpolation loses accuracy.
 */
export class LinearResampler {
  constructor(fromRate, toRate) {
    this.step = fromRate / toRate;
    this.reset();
  }

  reset() {
    this.pos = 0; // read position relative to the start of the next chunk (-1 = previous sample)
    this.prev = 0;
  }

  process(chunk) {
    const out = [];
    const last = chunk.length - 1;
    let pos = this.pos;
    while (pos <= last) {
      const i = Math.floor(pos), f = pos - i;
      const a = i < 0 ? this.prev : chunk[i];
      const b = i + 1 <= last ? chunk[i + 1] : chunk[last];
      out.push(a + (b - a) * f);
      pos += this.step;
    }
    this.pos = pos - chunk.length;
    this.prev = chunk[last];
    return Float32Array.from(out);
  }
}

/**
 * Records the microphone between start() and stop(), turns it into spectrogram frames exactly
 * like the recognizer would, and draws the microphone + spectrogram panels.
 */
export class Recorder {
  static async create(stream, els, opts) {
    // Prefer a 44.1 kHz context (the model's rate). Some browsers (older Firefox, some Safari)
    // refuse to connect a microphone running at another rate to it: then use the device's own
    // rate and resample to 44.1 kHz ourselves.
    for (const rate of [SAMPLE_RATE, null]) {
      let ctx;
      try {
        ctx = rate ? new AudioContext({ sampleRate: rate }) : new AudioContext();
        await ctx.audioWorklet.addModule(workletUrl);
        return new Recorder(ctx, stream, els, opts);
      } catch (err) {
        ctx?.close();
        if (!rate) throw err;
        console.warn(`44.1 kHz audio context not usable here (${err.message}); resampling instead.`);
      }
    }
  }

  constructor(ctx, stream, els, { bins, maxSeconds }) {
    this.ctx = ctx;
    this.els = els;
    this.bins = bins;
    this.maxFrames = Math.round((maxSeconds * SAMPLE_RATE) / FRAME_SIZE);
    this.spectrum = createSpectrum(FFT_SIZE);

    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'recorder-processor');
    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(node).connect(mute).connect(ctx.destination); // must be pulled to run; stays silent
    // Samples are only kept while recording. (The graph itself keeps running: in Chrome, a
    // suspend()/resume() cycle can leave a MediaStream source silent.)
    this.resampler = ctx.sampleRate === SAMPLE_RATE ? null : new LinearResampler(ctx.sampleRate, SAMPLE_RATE);
    node.port.onmessage = (e) => this.recording && this.onChunk(this.resampler ? this.resampler.process(e.data) : e.data);

    // One extra frame of samples: the first spectrum needs 2048 samples.
    this.samples = new Float32Array((this.maxFrames + 1) * FRAME_SIZE);
    this.length = 0;
    this.frames = [];
    this.recording = false;
    this.peakRms = 0;

    this.wctx = els.waveCanvas.getContext('2d');
    fitCanvas(els.waveCanvas, () => this.redrawWave());
    this.sctx = els.specCanvas.getContext('2d');
    this.clearSpec(this.maxFrames);
  }

  get seconds() {
    return this.length / SAMPLE_RATE;
  }

  async start() {
    this.frames = [];
    this.samples.fill(0);
    this.length = 0;
    this.peakRms = 0;
    this.lastSoundAt = performance.now();
    this.shown = null;
    this.setWindow(null);
    this.clearSpec(this.maxFrames);
    this.resampler?.reset();
    this.recording = true;
    await this.ctx.resume();
  }

  onChunk(chunk) {
    if (this.length + chunk.length > this.samples.length) return;
    this.samples.set(chunk, this.length);
    this.length += chunk.length;

    let sum = 0;
    for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
    const rms = Math.sqrt(sum / chunk.length);
    this.peakRms = Math.max(this.peakRms, rms);
    if (rms > 0.004) this.lastSoundAt = performance.now();
    const db = 20 * Math.log10(rms + 1e-8);
    this.els.meterFill.style.transform = `scaleX(${Math.min(1, Math.max(0, (db + 70) / 60))})`;

    // A new frame every 1024 samples, each looking at the latest 2048.
    while (FFT_SIZE + this.frames.length * FRAME_SIZE <= this.length) {
      const frame = frameAt(this.spectrum, this.samples, this.frames.length, this.bins);
      this.frames.push(frame);
      this.drawSpecColumn(frame, this.frames.length - 1);
    }
    if (this.frames.length >= this.maxFrames) this.onMaxReached?.();
  }

  /** Everything recorded, as raw 44.1 kHz samples (for saving a demo clip). */
  rawSamples() {
    return this.samples.slice(0, this.length);
  }

  /** Waveform samples aligned with the frames: frame i ↔ samples of its newest 1024. */
  frameAlignedSamples() {
    return this.samples.slice(FRAME_SIZE, FRAME_SIZE + this.frames.length * FRAME_SIZE);
  }

  /** Stop recording; returns the frames and the frame-aligned samples. */
  stop() {
    this.recording = false;
    this.els.meterFill.style.transform = 'scaleX(0)';
    return { frames: this.frames, samples: this.frameAlignedSamples() };
  }

  /** Called every animation frame while recording: waveform grows along a fixed timeline. */
  update() {
    if (!this.recording) return;
    const shown = this.samples.subarray(FRAME_SIZE);
    this.drawWave(shown, Math.max(0, this.length - FRAME_SIZE), this.maxFrames * FRAME_SIZE);
  }

  // ---------------------------------------------------------------- drawing

  clearSpec(width) {
    const c = this.els.specCanvas;
    c.width = width;
    c.height = this.bins;
    this.sctx.fillStyle = 'rgb(12,18,34)';
    this.sctx.fillRect(0, 0, width, this.bins);
    this.column = this.sctx.createImageData(1, this.bins);
  }

  drawSpecColumn(frame, x) {
    const px = this.column.data;
    for (let b = 0; b < this.bins; b++) {
      const i = lutIndex((frame[b] + 100) / 75); // dB → 0..1
      const o = (this.bins - 1 - b) * 4; // low frequencies at the bottom
      px[o] = LUT[i];
      px[o + 1] = LUT[i + 1];
      px[o + 2] = LUT[i + 2];
      px[o + 3] = 255;
    }
    this.sctx.putImageData(this.column, x, 0);
  }

  drawWave(samples, length, timeline) {
    const g = this.wctx, c = this.els.waveCanvas;
    const W = c.width, H = c.height, mid = H / 2;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = 'rgba(141,180,255,0.15)';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, mid);
    g.lineTo(W, mid);
    g.stroke();
    if (!length) return;
    // Min/max envelope per pixel column, with mild auto-gain so quiet speech is still visible.
    let peak = 0.05;
    for (let i = 0; i < length; i++) peak = Math.max(peak, Math.abs(samples[i]));
    const gain = Math.min(8, 0.9 / peak);
    const per = timeline / W;
    g.fillStyle = this.recording ? '#8db4ff' : '#ff8f52';
    for (let x = 0; x < W; x++) {
      const a = Math.floor(x * per);
      if (a >= length) break;
      const b = Math.min(length, Math.max(a + 1, Math.floor((x + 1) * per)));
      let lo = 1, hi = -1;
      for (let i = a; i < b; i++) {
        const v = samples[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const y0 = mid - hi * gain * mid, y1 = mid - lo * gain * mid;
      g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
  }

  redrawWave() {
    if (this.shown) this.drawWave(this.shown.samples, this.shown.samples.length, this.shown.samples.length);
  }

  /**
   * After recognition: show the whole recording fitted to the panels and mark the 1-second
   * window the network actually classified.
   */
  showRecording(frames, samples, windowStart, windowLen) {
    this.shown = { samples };
    this.clearSpec(frames.length);
    frames.forEach((f, x) => this.drawSpecColumn(f, x));
    this.redrawWave();
    this.setWindow({ left: windowStart / frames.length, width: windowLen / frames.length });
  }

  setWindow(win) {
    for (const el of [this.els.specWindow, this.els.waveWindow]) {
      el.classList.toggle('hidden', !win);
      if (win) {
        el.style.left = `${win.left * 100}%`;
        el.style.width = `${win.width * 100}%`;
      }
    }
  }
}

/** The frozen "image of sound": exactly the (normalized) frames × bins tensor fed to the model. */
export function drawInputHeatmap(canvas, data, frames, bins) {
  canvas.width = frames;
  canvas.height = bins;
  const g = canvas.getContext('2d');
  const img = g.createImageData(frames, bins);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const v = data[f * bins + b];
      const t = Number.isFinite(v) ? (v + 1.5) / 4.5 : 0; // z-scores ≈ -1.5 … 3
      const i = lutIndex(t);
      const o = ((bins - 1 - b) * frames + f) * 4;
      img.data[o] = LUT[i];
      img.data[o + 1] = LUT[i + 1];
      img.data[o + 2] = LUT[i + 2];
      img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}

/** Saliency overlay for the model-input heatmap: orange → cream, transparent where it didn't matter. */
export function drawSaliency(canvas, sal, frames, bins) {
  canvas.width = frames;
  canvas.height = bins;
  const g = canvas.getContext('2d');
  const img = g.createImageData(frames, bins);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const v = sal[f * bins + b];
      const o = ((bins - 1 - b) * frames + f) * 4;
      const t = Math.min(1, v * 1.25);
      img.data[o] = 255;
      img.data[o + 1] = Math.round(143 + (239 - 143) * t * t);
      img.data[o + 2] = Math.round(82 + (230 - 82) * t * t);
      img.data[o + 3] = Math.round(255 * Math.pow(v, 0.8));
    }
  }
  g.putImageData(img, 0, 0);
}
