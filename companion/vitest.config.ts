import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Test-only config: the component tests need the *browser* build of Svelte
// (mount/lifecycle), while the production build stays in vite.config.ts.
export default defineConfig({
  plugins: [svelte({ compilerOptions: { dev: true } })],
  resolve: {
    conditions: ['browser', 'module', 'import'],
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
    server: { deps: { inline: ['svelte', '@testing-library/svelte'] } },
  },
});
