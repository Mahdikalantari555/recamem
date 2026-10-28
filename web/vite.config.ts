import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Tauri shell loads `dist/` from disk; the browser fallback is served by
// `recamem serve --web`, which also proxies the API on the same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
