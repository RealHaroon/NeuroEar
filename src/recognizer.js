import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

// Visualization budget (keeps the 3D view at ~30fps on an integrated GPU).
const MAX_CHANNELS = 8; // feature maps shown per conv/pool layer
const MAX_FRAMES = 43; // max time steps per tile
const MAX_BINS = 64; // max frequency bins per tile
const MAX_UNITS = 128; // max neurons shown per dense/flatten layer

export const LABEL_NOISE = '_background_noise_';
export const LABEL_UNKNOWN = '_unknown_';

export function prettyLabel(w) {
  if (w === LABEL_NOISE) return 'noise';
  if (w === LABEL_UNKNOWN) return 'unknown';
  return w;
}

/** How a layer's output gets reduced for display (computed once from its shape). */
function makePlan(shape) {
  if (shape.length === 4) {
    const [, H, W, C] = shape;
    return { rank: 4, H, W, C, k: Math.min(MAX_CHANNELS, C), h: Math.min(H, MAX_FRAMES), w: Math.min(W, MAX_BINS) };
  }
  const n = shape.slice(1).reduce((a, b) => a * b, 1);
  const group = Math.ceil(n / Math.min(n, MAX_UNITS));
  return { rank: 2, n, group, count: Math.ceil(n / group) };
}

/**
 * Loads the recognizer from the local /model folder, pulls out its tf.LayersModel and builds an
 * activation model whose outputs are every layer's output.
 */
export async function loadModel() {
  await tf.ready();
  const modelURL = new URL('model/model.json', document.baseURI).href;
  const metadataURL = new URL('model/metadata.json', document.baseURI).href;

  const recognizer = speechCommands.create('BROWSER_FFT', undefined, modelURL, metadataURL);
  await recognizer.ensureModelLoaded();

  // BrowserFftSpeechCommandRecognizer keeps its tf.LayersModel in `.model` (see
  // browser_fft_recognizer.js: `this.model = model`). Fall back to loading it ourselves.
  let model = recognizer.model;
  if (!model || !Array.isArray(model.layers)) model = await tf.loadLayersModel(modelURL);

  console.log(`%cNeuroEar · tfjs ${tf.version.tfjs} · backend: ${tf.getBackend()}`, 'color:#ff8f52;font-weight:bold');
  model.summary();

  const actModel = tf.model({ inputs: model.inputs, outputs: model.layers.map((l) => l.output) });
  const inputShape = model.inputs[0].shape; // [null, frames, bins, 1]

  const layers = [
    { index: -1, name: 'input', type: 'Input', shape: inputShape, cfg: {}, params: 0, isInput: true },
    ...model.layers.map((layer, index) => ({
      index,
      name: layer.name,
      type: layer.getClassName(),
      shape: layer.outputShape,
      cfg: layer.getConfig(),
      params: layer.countParams(),
    })),
  ];
  for (const l of layers) l.plan = makePlan(l.shape);
  // Input tile is shown at full resolution: it *is* the spectrogram the model sees.
  layers[0].plan = { ...layers[0].plan, h: inputShape[1], w: inputShape[2] };
  const last = layers[layers.length - 1];
  last.isOutput = true;

  return { recognizer, model, actModel, inputShape, layers, words: recognizer.wordLabels(), backend: tf.getBackend(), memory: () => tf.memory() };
}

/** Same normalization as speech-commands' browser_fft_utils.normalize(): z-score with tf epsilon. */
function normalize(x) {
  const { mean, variance } = tf.moments(x);
  return tf.div(tf.sub(x, mean), tf.add(tf.sqrt(variance), tf.backend().epsilon()));
}

function reduceForDisplay(t, plan) {
  if (plan.rank === 4) {
    const { k, h, w, H, W } = plan;
    const { indices } = tf.topk(t.mean([0, 1, 2]), k); // most active channels first
    let g = tf.gather(t, indices, 3);
    if (h !== H || w !== W) g = tf.image.resizeBilinear(g, [h, w]);
    return [tf.transpose(g.squeeze([0]), [2, 0, 1]), indices]; // [k, frames, bins]
  }
  const flat = t.reshape([-1]);
  if (plan.group === 1) return [flat];
  const padded = flat.pad([[0, plan.count * plan.group - plan.n]]);
  return [padded.reshape([plan.count, plan.group]).max(1)];
}

/**
 * Runs the activation model on exactly the input the recognizer used (the raw spectrogram from
 * the listen() callback, normalized the same way) and returns display-sized activations.
 */
export async function computeActivations(ctx, spectrogramData) {
  const [, frames, bins, ch] = ctx.inputShape;
  const plans = ctx.layers.map((l) => l.plan);
  const tensors = tf.tidy(() => {
    const xn = normalize(tf.tensor4d(spectrogramData, [1, frames, bins, ch]));
    let outs = ctx.actModel.predict(xn);
    if (!Array.isArray(outs)) outs = [outs];
    const all = [[tf.transpose(xn.squeeze([0]), [2, 0, 1])], ...outs.map((o, i) => reduceForDisplay(o, plans[i + 1]))];
    return [...all.flat(), outs[outs.length - 1].reshape([-1])];
  });
  const arrays = await Promise.all(tensors.map((t) => t.data()));
  tf.dispose(tensors);

  let j = 0;
  const acts = plans.map((plan) => {
    const data = arrays[j++];
    const channels = plan.rank === 4 && plan.C > 1 ? arrays[j++] : null;
    let max = 0;
    for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
    return { plan, data, channels, max };
  });
  return { acts, finalOutput: arrays[j] };
}

// Open-set rejection. The model always picks one of its 20 classes and is often over-confident
// on words it never heard, so a word is only accepted when all of these hold:
export const ACCEPT = {
  minProb: 0.9, // the classified window must be this sure
  minShiftedProb: 0.75, // average probability of the same word when the window is nudged in time
  maxSpeechSec: 1.2, // speech longer than this is not a single command word
};
const SHIFTS = [-6, -3, 0, 3, 6]; // frames (≈ 23 ms each) for the shift test
const SPEECH_DB = 9; // a frame counts as speech when its energy is this far above the noise floor
const MAX_GAP_FRAMES = 8; // pauses up to ≈ 0.19 s still belong to the same utterance
const FRAME_SEC = 1024 / 44100;

/** Mean power (dB) of one spectrogram frame. */
function frameEnergyDb(frame) {
  let p = 0;
  for (let i = 0; i < frame.length; i++) p += 10 ** (frame[i] / 10);
  return 10 * Math.log10(p / frame.length + 1e-30);
}

/**
 * Where is the speech? Splits the recording into speech segments (gaps shorter than ~0.2 s are
 * bridged, so a multi-word phrase stays one segment) and returns the loudest one as
 * { start, end, centroid } in frame indices, or null for silence.
 */
function findSpeech(frames) {
  const e = frames.map(frameEnergyDb);
  const sorted = [...e].sort((a, b) => a - b);
  const floor = sorted[Math.floor(sorted.length * 0.1)];
  const peak = sorted[sorted.length - 1];
  if (peak - floor < SPEECH_DB) return null;
  const thr = Math.max(floor + SPEECH_DB, peak - 35);

  const segments = [];
  let cur = null;
  e.forEach((v, i) => {
    if (v <= thr) return;
    if (cur && i - cur.end <= MAX_GAP_FRAMES) cur.end = i;
    else segments.push((cur = { start: i, end: i }));
  });
  for (const seg of segments) {
    seg.power = 0;
    let csum = 0;
    for (let i = seg.start; i <= seg.end; i++) {
      const w = 10 ** ((e[i] - peak) / 10);
      seg.power += w;
      csum += w * i;
    }
    seg.centroid = csum / seg.power;
  }
  return segments.reduce((a, b) => (b.power > a.power ? b : a));
}

/**
 * The model hears exactly one second (43 frames). Center that window on the speech in the
 * recording, classify it, then check the answer survives small shifts in time.
 */
export async function classifyRecording(ctx, recorded) {
  const [, F, B] = ctx.inputShape;
  let frames = recorded;
  let padBefore = 0;
  if (frames.length < F) {
    // Too short: pad both sides with the quietest frame we recorded.
    const quiet = frames.reduce((q, f) => (frameEnergyDb(f) < frameEnergyDb(q) ? f : q), frames[0]);
    padBefore = Math.floor((F - frames.length) / 2);
    frames = [...Array(padBefore).fill(quiet), ...frames, ...Array(F - frames.length - padBefore).fill(quiet)];
  }

  const speech = findSpeech(frames);
  const clampStart = (s) => Math.min(frames.length - F, Math.max(0, Math.round(s)));
  const center = clampStart((speech ? speech.centroid : frames.length / 2) - F / 2);
  const starts = SHIFTS.map((d) => clampStart(center + d));

  const flat = new Float32Array(starts.length * F * B);
  starts.forEach((s, w) => {
    for (let f = 0; f < F; f++) flat.set(frames[s + f], (w * F + f) * B);
  });

  // Per-window z-score normalization: identical to browser_fft_utils.normalize() on each example.
  const normalized = tf.tidy(() => {
    const x = tf.tensor4d(flat, [starts.length, F, B, 1]);
    const { mean, variance } = tf.moments(x, [1, 2, 3], true);
    return tf.div(tf.sub(x, mean), tf.add(tf.sqrt(variance), tf.backend().epsilon()));
  });
  const { scores: perWindow } = await ctx.recognizer.recognize(normalized);
  normalized.dispose();

  const mid = SHIFTS.indexOf(0);
  const scores = perWindow[mid];
  let top = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[top]) top = i;
  const word = ctx.words[top];
  const prob = scores[top];
  const shifted = perWindow.reduce((a, sc) => a + sc[top], 0) / perWindow.length;
  const speechSec = speech ? (speech.end - speech.start + 1) * FRAME_SEC : 0;

  let decision;
  if (!speech || word === LABEL_NOISE) {
    decision = { accepted: false, kind: 'noise', reason: speech ? 'the network hears only background noise' : 'no speech in the recording' };
  } else if (word === LABEL_UNKNOWN) {
    decision = { accepted: false, kind: 'unknown', reason: `the network itself says “unknown” (${Math.round(prob * 100)}%)` };
  } else if (speechSec > ACCEPT.maxSpeechSec) {
    decision = { accepted: false, kind: 'unknown', reason: `${speechSec.toFixed(1)} s of speech is longer than one command word` };
  } else if (prob < ACCEPT.minProb) {
    decision = { accepted: false, kind: 'unknown', reason: `closest word “${word}” is only ${Math.round(prob * 100)}% likely` };
  } else if (shifted < ACCEPT.minShiftedProb) {
    decision = { accepted: false, kind: 'unknown', reason: `looked like “${word}”, but changed when the window shifted (${Math.round(shifted * 100)}% on average)` };
  } else {
    decision = { accepted: true, kind: 'word', reason: `${Math.round(prob * 100)}% · stable when shifted (${Math.round(shifted * 100)}%)` };
  }
  Object.assign(decision, { word, prob, shifted, speechSec });

  return {
    frames,
    padBefore,
    windowStart: center,
    windowLen: F,
    windowData: flat.slice(mid * F * B, (mid + 1) * F * B),
    scores,
    decision,
    speech,
  };
}
