import { ACCEPT, DEFAULT_ACCEPT, decide, prettyLabel } from './recognizer.js';

// Non-command words to test rejection with. "tree" and "green" sound like "three" on purpose.
const NON_WORDS = ['banana', 'hello', 'tree', 'green', 'water', 'computer'];
const STORAGE_KEY = 'neuroear.accept';

/** Load thresholds saved by an earlier calibration on this machine. */
export function loadSavedThresholds() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && saved.minProb > 0 && saved.minShiftedProb > 0) Object.assign(ACCEPT, saved);
  } catch {
    // no saved calibration (or storage blocked): keep defaults
  }
}

function saveThresholds() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ACCEPT));
  } catch {
    // storage blocked: thresholds still apply until reload
  }
}

/** Is this take handled correctly with these thresholds? */
function grade(take, accept) {
  const d = decide(take.stats, accept);
  if (take.isWord) return d.accepted ? (d.word === take.prompt ? 'ok' : 'wrong') : 'rejected';
  return d.accepted ? 'false-accept' : 'ok';
}

function summarize(takes, accept) {
  const g = takes.map((t) => grade(t, accept));
  const words = takes.filter((t) => t.isWord).length;
  return {
    correctWords: g.filter((x, i) => x === 'ok' && takes[i].isWord).length,
    words,
    rejected: g.filter((x) => x === 'rejected').length,
    wrong: g.filter((x) => x === 'wrong').length,
    nonWords: takes.length - words,
    falseAccepts: g.filter((x) => x === 'false-accept').length,
  };
}

/**
 * Grid-search the two probability thresholds. A false accept or wrong word costs 3× a rejected
 * command word. Among equally good settings, prefer the strictest (loosening that gains nothing
 * only lets more unheard words through), backed off by a small margin so a slightly less clear
 * repeat of the weakest accepted word still passes.
 */
export function suggestThresholds(takes) {
  const P_MARGIN = 3, S_MARGIN = 5; // percentage points
  const best = { score: -Infinity, pts: [] };
  for (let p = 50; p <= 99; p++) {
    for (let s = 40; s <= 95; s += 5) {
      const accept = { ...ACCEPT, minProb: p / 100, minShiftedProb: s / 100 };
      const r = summarize(takes, accept);
      const score = r.correctWords - 3 * (r.falseAccepts + r.wrong);
      if (score > best.score) Object.assign(best, { score, pts: [] });
      if (score === best.score) best.pts.push([p, s]);
    }
  }
  const maxP = Math.max(...best.pts.map(([p]) => p));
  const p = Math.max(Math.min(...best.pts.map(([q]) => q)), maxP - P_MARGIN);
  const sAtP = best.pts.filter(([q]) => q === maxP).map(([, s]) => s);
  const s = Math.max(Math.min(...sAtP), Math.max(...sAtP) - S_MARGIN);
  return { ...ACCEPT, minProb: p / 100, minShiftedProb: s / 100 };
}

/** Guided calibration: say each command word and a few non-words, then tune the thresholds. */
export class Calibration {
  constructor(root, words) {
    this.root = root;
    this.prompts = [...words.filter((w) => !w.startsWith('_')).map((w) => ({ prompt: w, isWord: true })), ...NON_WORDS.map((w) => ({ prompt: w, isWord: false }))];
    this.takes = [];
    this.index = 0;
    root.addEventListener('click', (e) => {
      const action = e.target.closest('[data-action]')?.dataset.action;
      if (action) this[action]();
    });
  }

  get active() {
    return !this.root.classList.contains('hidden');
  }

  toggle() {
    this.root.classList.toggle('hidden');
    this.render();
  }

  close() {
    this.root.classList.add('hidden');
  }

  /** Called with each recognition's decision while the panel is open. */
  addResult(decision) {
    if (!this.active || this.index >= this.prompts.length) return;
    this.takes = this.takes.filter((t) => t.prompt !== this.prompts[this.index].prompt);
    this.takes.push({ ...this.prompts[this.index], stats: decision });
    this.index++;
    this.render();
  }

  skip() {
    if (this.index < this.prompts.length) this.index++;
    this.render();
  }

  back() {
    if (this.index > 0) this.index--;
    this.render();
  }

  restart() {
    this.takes = [];
    this.index = 0;
    this.render();
  }

  apply() {
    Object.assign(ACCEPT, suggestThresholds(this.takes));
    saveThresholds();
    this.render();
  }

  resetDefaults() {
    Object.assign(ACCEPT, DEFAULT_ACCEPT);
    saveThresholds();
    this.render();
  }

  render() {
    const p = this.prompts[this.index];
    const fmt = (x) => `${Math.round(x * 100)}%`;
    const ask = !p
      ? '<div class="cal-ask">All done — check the suggestion below.</div>'
      : `<div class="cal-ask">${p.isWord ? 'Say' : 'Say a <b>non</b>-command word'}: <span class="accent">“${p.prompt}”</span><small>${this.index + 1} / ${this.prompts.length} · press Space, speak, press Space</small></div>`;

    const rows = this.takes
      .map((t) => {
        const s = t.stats;
        const now = grade(t, ACCEPT);
        const label = { ok: '✓', rejected: 'rejected', wrong: `✗ heard “${s.word}”`, 'false-accept': `✗ accepted “${s.word}”` }[now];
        return `<tr class="${now}"><td>${t.isWord ? '' : '<span class="dim">non-word</span> '}${t.prompt}</td><td>${prettyLabel(s.word)}</td><td>${fmt(s.prob)}</td><td>${fmt(s.shifted)}</td><td>${s.speechSec.toFixed(2)} s</td><td>${label}</td></tr>`;
      })
      .join('');

    let summary = '';
    if (this.takes.length) {
      const line = (name, a) => {
        const r = summarize(this.takes, a);
        return `<tr><td>${name}</td><td>${fmt(a.minProb)}</td><td>${fmt(a.minShiftedProb)}</td><td>${r.correctWords}/${r.words}</td><td>${r.rejected}</td><td>${r.wrong}</td><td>${r.falseAccepts}/${r.nonWords}</td></tr>`;
      };
      const sug = suggestThresholds(this.takes);
      summary = `<table class="cal-sum"><tr><th></th><th>min prob</th><th>min shifted</th><th>words OK</th><th>rejected</th><th>wrong word</th><th>non-words accepted</th></tr>${line('current', ACCEPT)}${line('suggested', sug)}${line('defaults', DEFAULT_ACCEPT)}</table>`;
    }

    this.root.innerHTML = `
      <div class="cal-head"><b>Calibrate for your voice &amp; mic</b><button data-action="close" class="cal-x" title="close (C)">×</button></div>
      ${ask}
      <div class="cal-buttons">
        <button data-action="back">← back</button><button data-action="skip">skip →</button><button data-action="restart">restart</button>
        <span class="cal-sep"></span>
        <button data-action="apply" ${this.takes.length ? '' : 'disabled'} class="primary">Apply suggested</button><button data-action="resetDefaults">Reset to defaults</button>
      </div>
      ${summary}
      <div class="cal-table"><table><tr><th>asked</th><th>network's top word</th><th>prob</th><th>shifted</th><th>speech</th><th>with current thresholds</th></tr>${rows}</table></div>
      <div class="dim cal-note">Thresholds are saved in this browser. Calibrate on the laptop and in the room you present in.</div>`;
  }
}
