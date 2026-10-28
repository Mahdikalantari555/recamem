# RecaMem desktop shell (Tauri 2)

The shell loads the built frontend from `../web/dist` and talks to the Python API
on loopback. It contains no model logic.

## Requirements

- Rust stable (1.77+) with `cargo`
- Node 20+ (builds `web/dist` via `beforeBuildCommand`)
- Platform Tauri prerequisites (WebView2 on Windows, `webkit2gtk` on Linux)

## Build and run

```bash
npm --prefix web install
npm --prefix web run build
cd src-tauri && cargo tauri build      # or: cargo run
```

`recamem ui` prefers `src-tauri/target/release/recamem` when it exists and falls
back to serving `web/dist` in the browser otherwise.

**Status in this repository's CI environment:** no Rust toolchain and no platform
WebView are available, so the shell is scaffolded and reviewed but not compiled
here. `recamem ui --dry-run` reports which shell would launch.
