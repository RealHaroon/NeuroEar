// Activation colormap built from the slide palette: deep navy → blue → orange → cream.
const STOPS = [
  [0.0, [12, 18, 34]],
  [0.22, [34, 56, 118]],
  [0.48, [141, 180, 255]],
  [0.74, [255, 143, 82]],
  [1.0, [243, 239, 230]],
];

export const LUT = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let s = 0;
  while (s < STOPS.length - 2 && t > STOPS[s + 1][0]) s++;
  const [t0, c0] = STOPS[s];
  const [t1, c1] = STOPS[s + 1];
  const f = (t - t0) / (t1 - t0);
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
}

/** Index into LUT for a value in [0, 1]. */
export function lutIndex(t) {
  if (!(t > 0)) return 0;
  return t >= 1 ? 255 * 3 : ((t * 255) | 0) * 3;
}

/** Write a colormapped pixel into an RGBA buffer. */
export function putColor(buf, offset, t) {
  const i = lutIndex(t);
  buf[offset] = LUT[i];
  buf[offset + 1] = LUT[i + 1];
  buf[offset + 2] = LUT[i + 2];
  buf[offset + 3] = 255;
}
