import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/700.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/700.css';
import './style.css';

import { loadModel, computeActivations, classifyRecording, prettyLabel } from './recognizer.js';
import { Recorder, drawInputHeatmap, SAMPLE_RATE } from './audio.js';
import { OutputPanel } from './output.js';
import { Network3D } from './network3d.js';
import { Layers2D } from './layers2d.js';
import { Presenter } from './presenter.js';

const MAX_RECORD_SECONDS = 3; // recording stops (and is recognized) automatically after this
const SILENCE_RMS = 0.004; // quieter than this for the whole recording = "no audio detected"
const FRAME_SIZE = 1024;

const $ = (id) => document.getElementById(id);
const startEl = $('start');
const startBtn = $('start-btn');
const startStatus = $('start-status');
const startError = $('start-error');
const recBtn = $('rec-btn');

let ctx, recorder, output, network, layers2d, presenter;
let state = 'idle'; // idle | recording | processing
let last = null; // last result: { word, score, scores, acts }

function setStatus(text, cls = '') {
  const el = $('status');
  el.textContent = text;
  el.className = `status ${cls}`;
}

function showStartError(title, body) {
  startError.innerHTML = `<b>${title}</b><br>${body}`;
  startError.classList.remove('hidden');
  startBtn.disabled = false;
  startBtn.textContent = 'Try again';
}

// Start loading the network immediately; audio has to wait for a click.
const modelPromise = loadModel();
modelPromise.then(
  (c) => (startStatus.textContent = `Neural network ready · ${c.model.layers.length} layers · ${c.model.countParams().toLocaleString()} parameters`),
  (err) => {
    console.error(err);
    startStatus.textContent = 'Model failed to load';
    showStartError(
      'Could not load the speech model.',
      `Expected the files in <code>public/model/</code> (model.json, metadata.json, weight shards).<br>Run <code>npm run setup-model</code> once while online, then reload.<br><small>${err.message}</small>`
    );
  }
);

startBtn.addEventListener('click', start);
recBtn.addEventListener('click', () => toggleRecording());
// Keep keyboard focus off the button: otherwise Space would trigger both the shortcut and a click.
recBtn.addEventListener('mousedown', (e) => e.preventDefault());

async function start() {
  startBtn.disabled = true;
  startError.classList.add('hidden');
  startBtn.textContent = 'Starting…';
  try {
    ctx = await modelPromise;
  } catch {
    return; // error already shown
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    showStartError('Microphone API unavailable.', 'Open the app via <code>http://localhost:5173</code> (mic access needs localhost or https).');
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (err) {
    const msg =
      err.name === 'NotAllowedError'
        ? ['Microphone access was blocked.', 'Click the mic/lock icon in the address bar, allow the microphone, then press “Try again”.']
        : err.name === 'NotFoundError'
          ? ['No microphone found.', 'Plug in or enable a microphone (check Ubuntu Settings → Sound → Input), then try again.']
          : ['Could not open the microphone.', `${err.name}: ${err.message}`];
    showStartError(...msg);
    return;
  }

  try {
    await setupUI(stream);
  } catch (err) {
    console.error(err);
    showStartError('Audio setup failed.', `${err.name}: ${err.message}`);
    return;
  }
  stream.getAudioTracks()[0]?.addEventListener('ended', () => showBanner('Microphone disconnected — reconnect it and reload the page.'));
  startEl.classList.add('hidden');
  startBtn.blur();
  setStatus(`ready · tfjs ${ctx.backend}`, 'ok');
}

async function setupUI(stream) {
  const { layers, words } = ctx;
  $('try-words').innerHTML = words
    .filter((w) => !w.startsWith('_'))
    .map((w) => `<span>${w}</span>`)
    .join('');
  $('net-summary').textContent = `${layers.length - 1} layers · ${ctx.model.countParams().toLocaleString()} params`;
  $('input-dims').textContent = `${ctx.inputShape[1]} frames × ${ctx.inputShape[2]} bins`;

  recorder = await Recorder.create(
    stream,
    { waveCanvas: $('wave'), meterFill: $('meter-fill'), specCanvas: $('spec-live'), specWindow: $('spec-window'), waveWindow: $('wave-window') },
    { bins: ctx.inputShape[2], maxSeconds: MAX_RECORD_SECONDS }
  );
  recorder.onMaxReached = () => state === 'recording' && toggleRecording();
  output = new OutputPanel(words, { bars: $('bars'), word: $('word'), conf: $('conf'), history: $('history') });
  network = new Network3D($('net3d'), layers, words, $('tooltip'));
  layers2d = new Layers2D($('layers2d'), layers);
  presenter = new Presenter({
    layers,
    panels: { mic: $('stage-mic'), spec: $('stage-spec'), net: $('stage-net'), out: $('stage-out') },
    captionEl: $('caption'),
    network,
    output,
    layers2d,
    getLast: () => last,
    canStep: () => state === 'idle' && last != null,
    onRecordToggle: () => toggleRecording(),
  });
  presenter.showIdleCaption();

  const loop = () => {
    if (state === 'recording') {
      recorder.update();
      $('rec-time').textContent = `${recorder.seconds.toFixed(1)} s / ${MAX_RECORD_SECONDS} s`;
      const silent = performance.now() - recorder.lastSoundAt > 1500;
      $('no-audio').classList.toggle('hidden', !silent);
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  window.__neuroear = { ctx, network, presenter, recorder, get last() { return last; }, toggleRecording, recognizeFrames };
}

function showBanner(text) {
  const el = $('no-audio');
  el.textContent = text;
  el.classList.remove('hidden');
}

function setRecUI(mode) {
  recBtn.dataset.mode = mode;
  recBtn.disabled = mode === 'processing';
  recBtn.textContent = { idle: '● Record', recording: '■ Recognize', processing: 'Thinking…' }[mode];
  const badge = $('rec-badge');
  badge.classList.toggle('hidden', mode !== 'recording');
  badge.textContent = '● recording';
  document.body.classList.toggle('recording', mode === 'recording');
}

async function toggleRecording() {
  if (!recorder) return;
  if (state === 'idle') {
    state = 'recording';
    presenter.reset();
    output.setRecording();
    $('no-audio').classList.add('hidden');
    $('rec-time').textContent = '';
    setRecUI('recording');
    presenter.showRecordingCaption();
    setStatus('');
    await recorder.start();
  } else if (state === 'recording') {
    state = 'processing';
    setRecUI('processing');
    setStatus('recognizing…');
    const { frames, samples } = recorder.stop();
    const silent = recorder.peakRms < SILENCE_RMS;
    $('no-audio').classList.add('hidden');
    try {
      if (frames.length === 0) throw new Error('No audio frames were captured.');
      await recognizeFrames(frames, samples);
      if (silent) showBanner('No audio detected — is the mic muted?');
    } catch (err) {
      console.error(err);
      showBanner(`Recognition failed: ${err.message}`);
      presenter.showIdleCaption();
    } finally {
      state = 'idle';
      setRecUI('idle');
      setStatus(`ready · tfjs ${ctx.backend}`, 'ok');
    }
  }
}

/** Classify the 1-second window around the speech, then run it through the whole pipeline. */
async function recognizeFrames(recordedFrames, samples) {
  const F = ctx.inputShape[1], B = ctx.inputShape[2];
  const r = await classifyRecording(ctx, recordedFrames);
  const { acts, finalOutput } = await computeActivations(ctx, r.windowData);

  // The activation model must reproduce the recognizer's own scores for the same window.
  let maxDiff = 0;
  for (let i = 0; i < finalOutput.length; i++) maxDiff = Math.max(maxDiff, Math.abs(finalOutput[i] - r.scores[i]));
  if (!last) console.log(`Activation model check: final layer vs recognizer.recognize() scores, max |Δ| = ${maxDiff.toExponential(2)}`);
  const verify = $('verify');
  verify.textContent = maxDiff < 1e-4 ? `✓ activations verified (Δ ${maxDiff.toExponential(0)})` : `✗ activation mismatch Δ ${maxDiff.toExponential(1)}`;
  verify.className = `verify ${maxDiff < 1e-4 ? 'ok' : 'bad'}`;

  const d = r.decision;
  const best = ctx.words.indexOf(d.word);
  last = { word: d.word, score: d.prob, decision: d, scores: Float32Array.from(r.scores), acts };

  // Pad the waveform like the frames were padded, so the highlighted window lines up.
  const padded = new Float32Array(r.frames.length * FRAME_SIZE);
  padded.set(samples.subarray(0, Math.min(samples.length, padded.length - r.padBefore * FRAME_SIZE)), r.padBefore * FRAME_SIZE);
  recorder.showRecording(r.frames, padded, r.windowStart, r.windowLen);
  drawInputHeatmap($('spec-input'), acts[0].data, F, B);
  $('input-word').textContent = d.accepted ? `“${prettyLabel(d.word)}”` : `(${d.kind})`;
  $('rec-time').textContent = `${((recordedFrames.length * FRAME_SIZE) / SAMPLE_RATE).toFixed(1)} s recorded · 1 s window centred on your speech`;

  output.updateScores(r.scores, d.accepted ? best : -1);
  output.showResult(d);
  network.play(acts);
  layers2d.update(acts, maxDiff);
  presenter.showResultCaption();
}
