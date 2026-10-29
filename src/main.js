import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/700.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/700.css';
import './style.css';

import { loadModel, warmUp, computeActivations, computeSaliency, classifyRecording, prettyLabel } from './recognizer.js';
import { Recorder, drawInputHeatmap, drawSaliency, framesFromSamples, SAMPLE_RATE } from './audio.js';
import { DemoClips, canSaveClips } from './demo.js';
import { Calibration, loadSavedThresholds } from './calibration.js';
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

let ctx, recorder, output, network, layers2d, presenter, demo, calibration;
let state = 'idle'; // idle | recording | processing
let last = null; // last result: { word, score, scores, acts }
let lastRecording = null; // raw samples of the last mic recording, for "save as demo clip"
const saveBtn = $('save-clip');
const salBtn = $('sal-btn');
const mobileBar = $('mobile-bar');
const mb = (name) => mobileBar.querySelector(`[data-mobile="${name}"]`);
let saliencyByUser = false; // S key / "focus" button
let saliencyByStage = false; // presenter's "Attention" step
let saliencyToken = 0; // ignore results of a saliency pass that a newer recording overtook

/** Show the saliency overlay when asked for (by the user or the presenter step) and available. */
function applySaliency() {
  const on = (saliencyByUser || saliencyByStage) && !!last?.saliency;
  $('input-wrap').classList.toggle('show-sal', on);
  salBtn.classList.toggle('on', on);
  mb('focus').classList.toggle('on', on);
  $('input-title').textContent = on ? 'the network’s focus' : 'the image of sound';
}

loadSavedThresholds();

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

// Start loading the network immediately (and compile its GPU shaders); audio has to wait for a click.
const modelPromise = loadModel().then(async (c) => {
  startStatus.textContent = 'Warming up the neural network…';
  const ms = await warmUp(c);
  console.log(`Warm-up (shader compilation) took ${Math.round(ms)} ms`);
  return c;
});
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
saveBtn.addEventListener('mousedown', (e) => e.preventDefault());
saveBtn.addEventListener('click', saveDemoClip);
salBtn.addEventListener('mousedown', (e) => e.preventDefault());
salBtn.addEventListener('click', toggleSaliency);
function toggleSaliency() {
  saliencyByUser = !saliencyByUser;
  applySaliency();
}

// Bottom control bar on phones (no keyboard there).
mobileBar.addEventListener('mousedown', (e) => e.preventDefault());
mobileBar.addEventListener('click', (e) => {
  const action = e.target.closest('[data-mobile]')?.dataset.mobile;
  if (!presenter) return;
  if (action === 'record') toggleRecording();
  else if (action === 'prev') presenter.prev();
  else if (action === 'next') presenter.next();
  else if (action === 'focus') toggleSaliency();
});

function setSaliencyAvailable(available) {
  salBtn.disabled = !available;
  mb('focus').disabled = !available;
}

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
  output = new OutputPanel(words, { bars: $('bars'), word: $('word'), conf: $('conf'), history: $('history'), shiftTest: $('shift-test') });
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
    onDigit: (n) => demo.play(n - 1),
    onCalibrate: () => calibration.toggle(),
    onSaliencyToggle: toggleSaliency,
    setStageSaliency: (on) => {
      saliencyByStage = on;
      applySaliency();
    },
  });
  presenter.showIdleCaption();
  demo = new DemoClips($('demo-clips'), { onPlay: playClip });
  demo.load();
  calibration = new Calibration($('calibration'), words);
  $('cal-btn').addEventListener('click', () => calibration.toggle());

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

  window.__neuroear = { ctx, network, presenter, recorder, demo, calibration, get last() { return last; }, toggleRecording, recognizeFrames };
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
  mb('record').dataset.mode = mode;
  mb('record').disabled = mode === 'processing';
  mb('record').textContent = recBtn.textContent;
  mb('prev').disabled = mb('next').disabled = mode !== 'idle' || !last;
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
    saveBtn.classList.add('hidden');
    presenter.showRecordingCaption();
    setStatus('');
    await recorder.start();
  } else if (state === 'recording') {
    state = 'processing';
    setRecUI('processing');
    setStatus('recognizing…');
    const { frames, samples } = recorder.stop();
    lastRecording = recorder.rawSamples();
    const silent = recorder.peakRms < SILENCE_RMS;
    $('no-audio').classList.add('hidden');
    try {
      if (frames.length === 0) throw new Error('No audio frames were captured.');
      const d = await recognizeFrames(frames, samples);
      calibration.addResult(d);
      if (canSaveClips) {
        saveBtn.dataset.name = d.accepted ? d.word : d.kind;
        saveBtn.textContent = `💾 Save as demo clip “${saveBtn.dataset.name}”`;
        saveBtn.classList.remove('hidden');
      }
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
  last = { word: d.word, score: d.prob, decision: d, shifts: r.shifts, scores: Float32Array.from(r.scores), acts };
  applySaliency(); // hides the previous recording's overlay until the new one is ready

  // Pad the waveform like the frames were padded, so the highlighted window lines up.
  const padded = new Float32Array(r.frames.length * FRAME_SIZE);
  padded.set(samples.subarray(0, Math.min(samples.length, padded.length - r.padBefore * FRAME_SIZE)), r.padBefore * FRAME_SIZE);
  recorder.showRecording(r.frames, padded, r.windowStart, r.windowLen);
  drawInputHeatmap($('spec-input'), acts[0].data, F, B);
  $('input-word').textContent = d.accepted ? `“${prettyLabel(d.word)}”` : `(${d.kind})`;
  $('rec-time').textContent = `${((recordedFrames.length * FRAME_SIZE) / SAMPLE_RATE).toFixed(1)} s recorded · 1 s window centred on your speech`;

  output.updateScores(r.scores, d.accepted ? best : -1);
  output.showResult(d);
  output.showShiftTest(r.shifts, d.word);
  network.play(acts);
  layers2d.update(acts, maxDiff);
  presenter.showResultCaption();
  updateSaliency(last, r.windowData, best);
  return d;
}

/** Saliency for the answer (in the background, so it doesn't delay the result). */
async function updateSaliency(result, windowData, classIndex) {
  const token = ++saliencyToken;
  setSaliencyAvailable(false);
  try {
    const sal = await computeSaliency(ctx, windowData, classIndex);
    if (token !== saliencyToken) return;
    drawSaliency($('spec-saliency'), sal, ctx.inputShape[1], ctx.inputShape[2]);
    result.saliency = sal;
    setSaliencyAvailable(true);
    applySaliency();
  } catch (err) {
    console.error('Saliency failed', err);
  }
}

/** Play a backup demo clip out loud, then run it through exactly the same pipeline. */
async function playClip(samples, file) {
  if (state !== 'idle') return;
  state = 'processing';
  setRecUI('processing');
  saveBtn.classList.add('hidden');
  presenter.reset();
  output.setRecording();
  $('no-audio').classList.add('hidden');
  presenter.setCaption('Demo clip', `playing ${file}`, null);
  try {
    const ac = recorder.ctx;
    const buf = ac.createBuffer(1, samples.length, SAMPLE_RATE);
    buf.copyToChannel(samples, 0);
    const src = ac.createBufferSource();
    src.buffer = buf;
    src.connect(ac.destination);
    // Let the audience hear it first (with a fallback in case "ended" never fires).
    await new Promise((resolve) => {
      src.onended = resolve;
      setTimeout(resolve, buf.duration * 1000 + 500);
      src.start();
    });
    const clip = framesFromSamples(samples, ctx.inputShape[2], MAX_RECORD_SECONDS);
    if (!clip.frames.length) throw new Error('clip is too short');
    await recognizeFrames(clip.frames, clip.samples);
    $('rec-time').textContent = `demo clip ${file}`;
  } catch (err) {
    console.error(err);
    showBanner(`Demo clip failed: ${err.message}`);
  } finally {
    state = 'idle';
    setRecUI('idle');
  }
}

async function saveDemoClip() {
  if (!lastRecording) return;
  saveBtn.disabled = true;
  try {
    const file = await demo.save(lastRecording, saveBtn.dataset.name || 'clip');
    saveBtn.textContent = `✓ saved public/demo/${file}`;
  } catch (err) {
    saveBtn.textContent = `✗ ${err.message}`;
  } finally {
    saveBtn.disabled = false;
  }
}
