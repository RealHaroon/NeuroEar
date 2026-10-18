import { prettyLabel, LABEL_NOISE, LABEL_UNKNOWN } from './recognizer.js';

const HISTORY_SIZE = 8;

/** Softmax bars (live, sorted), the big winning word, confidence and history. */
export class OutputPanel {
  constructor(words, { bars, word, conf, history }) {
    this.words = words;
    this.wordEl = word;
    this.confEl = conf;
    this.historyEl = history;
    this.barsEl = bars;
    this.history = [];
    this.rows = words.map((w) => {
      const row = document.createElement('div');
      row.className = 'bar-row' + (w === LABEL_NOISE || w === LABEL_UNKNOWN ? ' meta' : '');
      row.innerHTML = `<span class="bar-label">${prettyLabel(w)}</span><span class="bar-track"><span class="bar-fill"></span></span><span class="bar-val">0%</span>`;
      bars.appendChild(row);
      return { row, fill: row.querySelector('.bar-fill'), val: row.querySelector('.bar-val') };
    });
    bars.style.setProperty('--rows', words.length);
    this.updateScores(new Float32Array(words.length));
  }

  updateScores(scores, winner = -1) {
    const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]);
    order.forEach((idx, rank) => {
      const r = this.rows[idx];
      r.row.style.transform = `translateY(calc(${rank} * var(--row-h)))`;
      r.fill.style.transform = `scaleX(${scores[idx]})`;
      r.val.textContent = `${Math.round(scores[idx] * 100)}%`;
      r.row.classList.toggle('top', idx === winner);
    });
  }

  /** While recording: clear the previous result. */
  setRecording() {
    this.updateScores(new Float32Array(this.words.length));
    this.wordEl.textContent = '…';
    this.wordEl.classList.add('unknown');
    this.wordEl.classList.remove('pop');
    this.confEl.textContent = 'listening';
  }

  /** Show the app's decision: an accepted command word, or "unknown" / "noise" with the reason. */
  showResult(d) {
    this.wordEl.classList.remove('pop');
    void this.wordEl.offsetWidth; // restart the CSS animation
    this.wordEl.classList.add('pop');
    this.wordEl.classList.toggle('unknown', !d.accepted);
    if (d.accepted) {
      this.wordEl.textContent = prettyLabel(d.word);
      this.confEl.textContent = `${(d.prob * 100).toFixed(1)}% confident`;
      this.history.unshift(d.word);
      this.history.length = Math.min(this.history.length, HISTORY_SIZE);
      this.renderHistory();
    } else {
      this.wordEl.textContent = d.kind; // "unknown" or "noise"
      this.confEl.textContent = d.reason.charAt(0).toUpperCase() + d.reason.slice(1);
    }
  }

  renderHistory() {
    this.historyEl.innerHTML = this.history.map((w, i) => `<span class="chip${i === 0 ? ' newest' : ''}">${w}</span>`).join('');
  }

  reset() {
    this.history = [];
    this.renderHistory();
    this.wordEl.textContent = '—';
    this.wordEl.classList.remove('unknown');
    this.confEl.textContent = '';
  }
}
