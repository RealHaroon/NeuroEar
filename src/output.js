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

  /** `confident` = above the probability threshold and a real word. */
  showResult(word, score, confident) {
    this.wordEl.classList.remove('pop');
    void this.wordEl.offsetWidth; // restart the CSS animation
    this.wordEl.classList.add('pop');
    this.wordEl.classList.toggle('unknown', !confident);
    if (word === LABEL_NOISE) {
      this.wordEl.textContent = '—';
      this.confEl.textContent = 'No word heard — try again, a bit louder';
    } else if (word === LABEL_UNKNOWN) {
      this.wordEl.textContent = '?';
      this.confEl.textContent = `Not one of the 18 words (“unknown” ${Math.round(score * 100)}%)`;
    } else if (!confident) {
      this.wordEl.textContent = '?';
      this.confEl.textContent = `Not sure — best guess “${prettyLabel(word)}” (${Math.round(score * 100)}%)`;
    } else {
      this.wordEl.textContent = prettyLabel(word);
      this.confEl.textContent = `${(score * 100).toFixed(1)}% confident`;
      this.history.unshift(word);
      this.history.length = Math.min(this.history.length, HISTORY_SIZE);
      this.renderHistory();
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
