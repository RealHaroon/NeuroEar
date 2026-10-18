import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: 'localhost',
    port: 5173,
    // The project lives under "Local Disk (D:)"; the ":" confuses Vite's path allow-list
    // (it looks like a Windows drive), so relax it. The server only listens on localhost.
    fs: { strict: false },
  },
  preview: { host: 'localhost', port: 4173 },
  build: {
    chunkSizeWarningLimit: 4000,
    // Keep the AudioWorklet as a real file: some browsers refuse data: URLs in addModule().
    assetsInlineLimit: (file) => (file.includes('worklet') ? false : undefined),
  },
});
