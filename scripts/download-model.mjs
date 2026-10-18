// Downloads the pretrained speech-commands BROWSER_FFT model (18 words) into public/model
// so the app works fully offline. Usage: npm run setup-model  (add --force to re-download)
import { mkdir, writeFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const BASE = 'https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.5/browser_fft/18w';
const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public', 'model');
const force = process.argv.includes('--force');

async function exists(p) { try { await access(p); return true; } catch { return false; } }

async function get(name) {
  const dest = path.join(OUT, name);
  if (!force && await exists(dest)) { console.log(`  ✓ ${name} (already present)`); return dest; }
  const res = await fetch(`${BASE}/${name}`);
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
  console.log(`  ↓ ${name}`);
  return dest;
}

await mkdir(OUT, { recursive: true });
console.log(`Downloading model into ${OUT}`);
const modelJson = JSON.parse(await (await import('node:fs/promises')).readFile(await get('model.json'), 'utf8'));
await get('metadata.json');
for (const group of modelJson.weightsManifest) for (const p of group.paths) await get(p);
console.log('Done. The app now loads the model from /model/ with no network access.');
