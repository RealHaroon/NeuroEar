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

  // For saliency: everything up to the last layer, plus that layer's weights, so we can take the
  // gradient of a class's raw score (logit) — softmax saturates at 100% and kills the gradient.
  const lastLayer = model.layers[model.layers.length - 1];
  const head =
    lastLayer.getClassName() === 'Dense' && model.layers.length > 1
      ? { body: tf.model({ inputs: model.inputs, outputs: model.layers[model.layers.length - 2].output }), weights: lastLayer.getWeights() }
      : null;

  return { recognizer, model, actModel, head, inputShape, layers, words: recognizer.wordLabels(), backend: tf.getBackend(), memory: () => tf.memory() };
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
export const DEFAULT_ACCEPT = Object.freeze({
  minProb: 0.9, // the classified window must be this sure
  minShiftedProb: 0.75, // average probability of the same word when the window is nudged in time
  maxSpeechSec: 1.2, // speech longer than this is not a single command word
});
/** The thresholds in use (calibration can change them). */
export const ACCEPT = { ...DEFAULT_ACCEPT };
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
  const argmax = (sc) => sc.reduce((b, v, i) => (v > sc[b] ? i : b), 0);
  let top = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[top]) top = i;
  const word = ctx.words[top];
  const prob = scores[top];
  const shifted = perWindow.reduce((a, sc) => a + sc[top], 0) / perWindow.length;
  const speechSec = speech ? (speech.end - speech.start + 1) * FRAME_SEC : 0;

  const decision = decide({ word, prob, shifted, speechSec, hasSpeech: !!speech });
  // What each nudged window heard, for the shift-test display. Near the edge of the recording
  // several shifts clamp to the same window; show each window once.
  const shifts = perWindow
    .map((sc, i) => {
      const t = argmax(sc);
      return { start: starts[i], ms: Math.round((starts[i] - center) * FRAME_SEC * 1000), word: ctx.words[t], prob: sc[t], probOfMain: sc[top] };
    })
    .filter((sh, i, all) => all.findIndex((o) => o.start === sh.start) === i);

  return {
    frames,
    padBefore,
    windowStart: center,
    windowLen: F,
    windowData: flat.slice(mid * F * B, (mid + 1) * F * B),
    scores,
    decision,
    shifts,
    speech,
  };
}

/**
 * Accept a command word, or explain why the answer is "unknown" / "noise".
 * `stats` = { word, prob, shifted, speechSec, hasSpeech } from classifyRecording().
 */
export function decide(stats, accept = ACCEPT) {
  const { word, prob, shifted, speechSec, hasSpeech } = stats;
  let d;
  if (!hasSpeech || word === LABEL_NOISE) {
    d = { accepted: false, kind: 'noise', reason: hasSpeech ? 'the network hears only background noise' : 'no speech in the recording' };
  } else if (word === LABEL_UNKNOWN) {
    d = { accepted: false, kind: 'unknown', reason: `the network itself says “unknown” (${Math.round(prob * 100)}%)` };
  } else if (speechSec > accept.maxSpeechSec) {
    d = { accepted: false, kind: 'unknown', reason: `${speechSec.toFixed(1)} s of speech is longer than one command word` };
  } else if (prob < accept.minProb) {
    d = { accepted: false, kind: 'unknown', reason: `closest word “${word}” is only ${Math.round(prob * 100)}% likely` };
  } else if (shifted < accept.minShiftedProb) {
    d = { accepted: false, kind: 'unknown', reason: `looked like “${word}”, but changed when the window shifted (${Math.round(shifted * 100)}% on average)` };
  } else {
    d = { accepted: true, kind: 'word', reason: `${Math.round(prob * 100)}% · stable when shifted (${Math.round(shifted * 100)}%)` };
  }
  return Object.assign(d, stats);
}

const SALIENCY_SAMPLES = 16;
const SALIENCY_NOISE = 0.15; // noise std as a fraction of the input's value range

/**
 * SmoothGrad saliency: how much each time-frequency cell of the (normalized) input changes the
 * score of class `classIndex`. Gradients are averaged over noisy copies of the input to remove
 * speckle. Returns a Float32Array [frames × bins] scaled to 0..1.
 */
export async function computeSaliency(ctx, spectrogramData, classIndex) {
  const [, F, B, C] = ctx.inputShape;
  const sal = tf.tidy(() => {
    const xn = normalize(tf.tensor4d(spectrogramData, [1, F, B, C]));
    const range = tf.sub(tf.max(xn), tf.min(xn));
    const noisy = tf.add(tf.tile(xn, [SALIENCY_SAMPLES, 1, 1, 1]), tf.mul(tf.randomNormal([SALIENCY_SAMPLES, F, B, C]), tf.mul(range, SALIENCY_NOISE)));
    let score;
    if (ctx.head) {
      const [kernel, bias] = ctx.head.weights;
      const w = tf.slice(kernel, [0, classIndex], [-1, 1]);
      const b = bias ? tf.slice(bias, [classIndex], [1]) : tf.scalar(0);
      score = (x) => tf.sum(tf.add(tf.matMul(ctx.head.body.apply(x), w), b));
    } else {
      score = (x) => tf.sum(tf.slice(ctx.model.apply(x), [0, classIndex], [-1, 1]));
    }
    const g = tf.abs(tf.grad(score)(noisy)).mean([0, 3]); // [F, B]
    // Light blur so the map shows regions rather than single pixels.
    const blurred = tf.avgPool(g.reshape([1, F, B, 1]), [3, 3], 1, 'same').reshape([F * B]);
    return blurred;
  });
  const data = await sal.data();
  sal.dispose();
  // Scale by the 99th percentile so one hot pixel doesn't wash everything else out.
  const sorted = Float32Array.from(data).sort();
  const p99 = sorted[Math.floor(sorted.length * 0.99)] || 1;
  for (let i = 0; i < data.length; i++) data[i] = Math.min(1, data[i] / p99);
  return data;
}

/**
 * Run every model path once on dummy data so WebGL compiles its shaders now, not during the
 * first live recognition. Returns the time it took in ms.
 */
export async function warmUp(ctx) {
  const t0 = performance.now();
  const [, F, B] = ctx.inputShape;
  const x = tf.randomNormal([SHIFTS.length, F, B, 1]);
  await ctx.recognizer.recognize(x);
  x.dispose();
  const spec = new Float32Array(F * B);
  for (let i = 0; i < spec.length; i++) spec[i] = -90 + 40 * Math.random();
  await computeActivations(ctx, spec);
  await computeSaliency(ctx, spec, 0);
  return performance.now() - t0;
}
