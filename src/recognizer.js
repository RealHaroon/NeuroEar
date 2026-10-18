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

const WINDOW_STRIDE = 2; // frames between candidate 1-second windows (≈ 46 ms)

/**
 * The model hears exactly one second (43 frames). Slide that window over the whole recording,
 * score every window in one batch with the recognizer, and keep the one where a real word is
 * most likely.
 */
export async function classifyRecording(ctx, recorded) {
  const [, F, B] = ctx.inputShape;
  let frames = recorded;
  let padBefore = 0;
  if (frames.length < F) {
    // Too short: pad both sides with the quietest frame we recorded.
    const energy = (f) => f.reduce((a, v) => a + v, 0);
    const quiet = frames.reduce((q, f) => (energy(f) < energy(q) ? f : q), frames[0]);
    padBefore = Math.floor((F - frames.length) / 2);
    frames = [...Array(padBefore).fill(quiet), ...frames, ...Array(F - frames.length - padBefore).fill(quiet)];
  }

  const starts = [];
  for (let s = 0; s + F <= frames.length; s += WINDOW_STRIDE) starts.push(s);
  if (starts[starts.length - 1] !== frames.length - F) starts.push(frames.length - F);

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
  const { scores } = await ctx.recognizer.recognize(normalized);
  normalized.dispose();
  const perWindow = starts.length === 1 ? [scores] : scores;

  const realWords = ctx.words.map((w, i) => (w === LABEL_NOISE || w === LABEL_UNKNOWN ? -1 : i)).filter((i) => i >= 0);
  let best = 0, bestScore = -1;
  perWindow.forEach((sc, w) => {
    const m = Math.max(...realWords.map((i) => sc[i]));
    if (m > bestScore) {
      bestScore = m;
      best = w;
    }
  });

  return {
    frames,
    padBefore,
    windowStart: starts[best],
    windowLen: F,
    windowData: flat.slice(best * F * B, (best + 1) * F * B),
    scores: perWindow[best],
    numWindows: starts.length,
  };
}
