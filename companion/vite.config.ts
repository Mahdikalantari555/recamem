import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  // Relative asset URLs: the built app is served by the Python API on the same
  // origin, so no absolute localhost address ever reaches the browser.
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    // Bind all interfaces so the sandbox preview proxy can reach the dev
    // server. The Python API stays loopback-only and is reached through this
    // dev server's /api proxy, so the browser never dials a host directly.
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // The sandbox preview serves the app through a proxy host; permit it
    // without permitting arbitrary hosts.
    allowedHosts: ['.e2b.app', 'localhost', '127.0.0.1'],
    proxy: { '/api': 'http://127.0.0.1:8000' }
  }
});
