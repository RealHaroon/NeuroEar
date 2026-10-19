import { defineConfig } from 'vite';
import { existsSync, mkdirSync, readdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DEMO_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'public', 'demo');

/** public/demo/index.json lists the clips, so the built app can find them without a server. */
function writeDemoIndex() {
  mkdirSync(DEMO_DIR, { recursive: true });
  const files = readdirSync(DEMO_DIR).filter((f) => f.endsWith('.wav')).sort();
  writeFileSync(path.join(DEMO_DIR, 'index.json'), JSON.stringify(files, null, 2) + '\n');
  return files;
}

/** Dev-only endpoints that save / delete backup demo clips in public/demo. */
function demoClips() {
  const send = (res, code, body) => {
    res.statusCode = code;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(body));
  };
  return {
    name: 'neuroear-demo-clips',
    buildStart: writeDemoIndex,
    configureServer(server) {
      writeDemoIndex();
      // Serve public/demo straight from disk: Vite caches the public folder's file list at startup,
      // so clips saved later would otherwise 404 (and index.json would be stale).
      server.middlewares.use('/demo', (req, res, next) => {
        const file = path.basename(decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
        if (file === 'index.json') return send(res, 200, writeDemoIndex());
        const full = path.join(DEMO_DIR, file);
        if (!file.endsWith('.wav') || !existsSync(full)) return next();
        res.setHeader('content-type', 'audio/wav');
        res.end(readFileSync(full));
      });
      server.middlewares.use('/__neuroear/save-clip', (req, res) => {
        if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
        const name = new URL(req.url, 'http://localhost').searchParams.get('name')?.toLowerCase().replace(/[^a-z0-9_-]/g, '') || 'clip';
        const chunks = [];
        let size = 0;
        req.on('data', (c) => {
          size += c.length;
          if (size < 5e6) chunks.push(c);
        });
        req.on('end', () => {
          if (size >= 5e6) return send(res, 413, { error: 'clip too large' });
          let file = `${name}.wav`;
          for (let n = 2; existsSync(path.join(DEMO_DIR, file)); n++) file = `${name}-${n}.wav`;
          writeFileSync(path.join(DEMO_DIR, file), Buffer.concat(chunks));
          send(res, 200, { file, files: writeDemoIndex() });
        });
      });
      server.middlewares.use('/__neuroear/delete-clip', (req, res) => {
        if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
        const file = path.basename(new URL(req.url, 'http://localhost').searchParams.get('file') || '');
        if (!file.endsWith('.wav') || !existsSync(path.join(DEMO_DIR, file))) return send(res, 404, { error: 'no such clip' });
        unlinkSync(path.join(DEMO_DIR, file));
        send(res, 200, { files: writeDemoIndex() });
      });
    },
  };
}

export default defineConfig({
  plugins: [demoClips()],
  server: {
    host: 'localhost',
    port: 5173,
    // The project lives under "Local Disk (D:)"; the ":" confuses Vite's path allow-list
    // (it looks like a Windows drive), so relax it. The server only listens on localhost.
    fs: { strict: false },
    // Saving a demo clip must not reload the page.
    watch: { ignored: ['**/public/demo/**'] },
  },
  preview: { host: 'localhost', port: 4173 },
  build: {
    chunkSizeWarningLimit: 4000,
    // Keep the AudioWorklet as a real file: some browsers refuse data: URLs in addModule().
    assetsInlineLimit: (file) => (file.includes('worklet') ? false : undefined),
  },
});
