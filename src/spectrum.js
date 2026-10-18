// Offline re-implementation of what an AnalyserNode (fftSize 2048, smoothing 0) returns from
// getFloatFrequencyData(): Blackman window → FFT → |X| / N → dB. This is the feature extractor
// speech-commands uses, without depending on setInterval timing.

export function createSpectrum(N) {
  const window = new Float32Array(N);
  for (let n = 0; n < N; n++) window[n] = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / N) + 0.08 * Math.cos((4 * Math.PI * n) / N);
  const re = new Float64Array(N), im = new Float64Array(N);
  const levels = Math.log2(N);
  const rev = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(N / 2), sin = new Float64Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / N);
    sin[i] = -Math.sin((2 * Math.PI * i) / N);
  }

  /** dB magnitudes of the first `bins` frequency bins for samples[end - N, end). */
  return function spectrum(samples, end, bins) {
    for (let i = 0; i < N; i++) {
      re[rev[i]] = samples[end - N + i] * window[i];
      im[rev[i]] = 0;
    }
    for (let size = 2; size <= N; size *= 2) {
      const half = size / 2, step = N / size;
      for (let start = 0; start < N; start += size) {
        for (let k = 0; k < half; k++) {
          const a = start + k, b = a + half;
          const wr = cos[k * step], wi = sin[k * step];
          const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
    const out = new Float32Array(bins);
    for (let k = 0; k < bins; k++) out[k] = 20 * Math.log10(Math.hypot(re[k], im[k]) / N);
    return out;
  };
}
