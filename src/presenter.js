import { prettyLabel } from './recognizer.js';
import { formatShape } from './network3d.js';

/** Builds the step-through stages from the model's real layers. */
export function buildStages(layers) {
  const stages = [
    { panels: ['mic'], title: 'Microphone', text: 'sound is just air pressure over time', detail: '44,100 samples per second' },
    {
      panels: ['spec'],
      title: 'Spectrogram',
      text: 'FFT turns sound into an image of frequencies over time',
      detail: `${layers[0].shape[1]} frames × ${layers[0].shape[2]} frequency bins = the network's input`,
    },
  ];
  let seenConv = false;
  layers.forEach((L, i) => {
    if (L.isInput) return;
    const shape = formatShape(L.shape);
    const c = L.cfg;
    let s;
    if (L.type === 'Conv2D') {
      s = {
        title: 'Conv layer',
        text: `filters slide over the ${seenConv ? 'feature maps' : 'spectrogram'} and detect patterns`,
        detail: `${L.name}: ${c.filters} filters of ${c.kernelSize.join('×')} → ${shape}`,
      };
      seenConv = true;
    } else if (L.type === 'MaxPooling2D') {
      s = { title: 'Pooling', text: 'keeps the strongest signals, shrinks the map', detail: `${L.name}: ${c.poolSize.join('×')} max → ${shape}` };
    } else if (L.type === 'Flatten') {
      s = { title: 'Flatten', text: 'unrolls the feature maps into one long list of numbers', detail: `${L.name}: ${shape} values` };
    } else if (L.type === 'Dropout') {
      s = { title: 'Dropout', text: 'switches random neurons off while training; passes everything through now', detail: `${L.name}: rate ${c.rate}` };
    } else if (L.type === 'Dense' && c.activation === 'softmax') {
      s = { title: 'Softmax', text: 'turns scores into probabilities that sum to 1', detail: `${L.name}: ${c.units} outputs, one per word`, panels: ['net', 'out'] };
    } else if (L.type === 'Dense') {
      s = { title: 'Dense layer', text: 'combines all features', detail: `${L.name}: ${c.units} neurons, each sees every input` };
    } else {
      s = { title: L.type, text: 'transforms the signal', detail: `${L.name} → ${shape}` };
    }
    stages.push({ panels: ['net'], layer: i, ...s });
  });
  stages.push({ panels: ['out'], title: 'Prediction', text: 'the word with the highest probability wins', detail: null, prediction: true });
  return stages;
}

/** Keyboard-driven presenter mode: record/recognize, step through the pipeline, captions. */
export class Presenter {
  constructor({ layers, panels, captionEl, network, output, layers2d, getLast, canStep, onRecordToggle }) {
    Object.assign(this, { panels, captionEl, network, output, layers2d, getLast, canStep, onRecordToggle });
    this.stages = buildStages(layers);
    this.step = -1;
    this.captionsOn = true;
    window.addEventListener('keydown', (e) => this.onKey(e));
  }

  onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    switch (e.key) {
      case ' ':
      case 'Enter':
        e.preventDefault();
        this.onRecordToggle();
        break;
      case 'ArrowRight':
        e.preventDefault();
        if (this.canStep()) this.goTo(Math.min(this.stages.length - 1, this.step + 1));
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (this.canStep() && this.step > 0) this.goTo(this.step - 1);
        break;
      case 'Escape':
        if (this.step >= 0) this.goTo(-1);
        break;
      case 'f':
      case 'F':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen().catch(() => {});
        break;
      case 'r':
      case 'R':
        this.output.reset();
        break;
      case 'h':
      case 'H':
        this.captionsOn = !this.captionsOn;
        this.captionEl.classList.toggle('hidden', !this.captionsOn);
        break;
      case 'd':
      case 'D':
        this.layers2d.toggle();
        break;
    }
  }

  /** Leave step mode (if active) and restore all panels. */
  reset() {
    if (this.step >= 0) this.goTo(-1, false);
  }

  goTo(step, updateCaption = true) {
    this.step = step;
    const stage = this.stages[step];
    for (const [name, el] of Object.entries(this.panels)) {
      el.classList.toggle('active', !!stage && stage.panels.includes(name));
      el.classList.toggle('dimmed', !!stage && !stage.panels.includes(name));
    }
    if (!stage) {
      this.network.focus(null);
      if (updateCaption) this.showResultCaption();
      return;
    }
    this.network.focus(stage.layer ?? null);
    let detail = stage.detail;
    if (stage.prediction) {
      const last = this.getLast();
      const d = last?.decision;
      detail = !d ? null : d.accepted ? `“${prettyLabel(d.word)}” with ${(d.prob * 100).toFixed(1)}% probability` : `but here the answer is “${d.kind}”: ${d.reason}`;
    }
    this.setCaption(stage.title, stage.text, detail, `${step + 1}/${this.stages.length}`);
  }

  setCaption(title, text, detail, counter = '') {
    this.captionEl.innerHTML =
      `<span class="cap-main"><span class="cap-title">${title}:</span> <span class="cap-text">${text}</span>` +
      (detail ? ` <span class="cap-detail">— ${detail}</span>` : '') +
      '</span>' +
      (counter ? `<span class="cap-step">${counter}</span>` : '');
  }

  showIdleCaption() {
    this.setCaption('Ready', 'press Record (Space), say one word, then press Recognize', null);
  }

  showRecordingCaption() {
    this.setCaption('Recording', 'say one word, then press Recognize (Space)', 'stops automatically after 3 s');
  }

  showResultCaption() {
    const last = this.getLast();
    if (!last) return this.showIdleCaption();
    const d = last.decision;
    const heard = d.accepted
      ? `the network heard “${prettyLabel(d.word)}”`
      : d.kind === 'noise'
        ? `no command word — ${d.reason}`
        : `not one of the 18 words — ${d.reason}`;
    this.setCaption('Result', heard, 'press → to walk through the pipeline · Space to record again');
  }
}
