# RecaMem web UI

React + TypeScript + Vite + Tailwind, with shadcn-style components copied into
`src/components/ui` (no runtime UI dependency). The app talks to the Python
backend through `/api/v1` only — it never imports model code.

```bash
npm install
npm run build      # emits web/dist, served by `recamem serve --web`
npm run dev        # dev server on 127.0.0.1:5173, proxying /api to :8000
```

Start the backend first:

```bash
recamem serve --rest --port 8000
```

Design tokens live in `tailwind.config.ts` and `src/styles/globals.css`; see
`docs/ui-design.md` §4 for the source values. Pages ship progressively: Playground
and a minimal Experiments view are live; Memory Explorer, Benchmarks, Training,
Architecture Lab and Settings are labelled "not available yet" in the nav rather
than showing placeholder data.
