# Web demo profiling

Run from the project root with Node 22+ and the web demo dependencies installed:

```sh
node scripts/web/profile-demo.cjs --deep --out runs/web-profile/local
node scripts/web/profile-demo.cjs --control --out runs/web-profile/local
```

The default Chrome path is the local macOS installation. Select another installed browser with `--chrome /path/to/chrome` or `CHROME_PATH`. Use `--only desktop-map`, `desktop-globe`, `mobile-map`, or `mobile-globe` to isolate one case. Current model assets must pass the demo's `sync-models.cjs --check`.

The script starts a separate loopback server, launches a fresh browser for each case, and stops both when finished. It profiles the current initial view, cached panning, and zooming into new detail. `--deep` adds a globe view at 1 km altitude. Mobile emulates viewport, DPR, touch capability, and the demo's memory-saving configuration on the host computer; it does not reproduce phone hardware performance. Navigation uses deterministic mouse/keyboard input, including in the mobile configuration; this is not a pinch-gesture benchmark.

`report.md` summarizes measurements. Case JSON files retain component timers, process CPU deltas, rAF intervals, long tasks, WASM allocations, and resource metadata. Open `.cpuprofile` files in Chrome DevTools to inspect page and terrain-worker stacks. Browser traces are gzip-compressed JSON. Instrumented JavaScript copies and final screenshots are retained alongside the results for inspection.

Instrumentation is injected into responses through Playwright routing and leaves production files unchanged. Timers are cumulative and often nested; worker sums can exceed wall time. CPU profiles keep unattributed `(program)` samples separate from JavaScript/WASM. GPU-process CPU and WebGL submission timings do not measure GPU hardware execution time. rAF intervals show scheduling responsiveness, not app draw FPS. `--control` repeats startup, cached pan, and zoom/pan with no instrumentation or CPU sampling, writing `*-control.json` and `control-summary.json` to check overhead. Both modes use locally served assets and no CPU/network throttling; each case is a single run.
