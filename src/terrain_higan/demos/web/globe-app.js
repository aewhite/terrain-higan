"use strict";
(() => {
  const {
    GlobeRenderer,
    GlobeTileController,
    GlobeWalk,
    WalkMinimap,
    downloadWalkingScene,
    WALK_CEILING,
    WALK_MAX_SPEED,
    RADIUS,
    GLOBE_VERSION,
    greatCircle,
  } = TerrainHiGAN.globe;
  const canvas = document.getElementById("world");
  const element = (id) => document.getElementById(id);
  const hash = new URLSearchParams(location.hash.slice(1));
  const numeric = (name, fallback) =>
    hash.has(name) && Number.isFinite(Number(hash.get(name))) ? Number(hash.get(name)) : fallback;
  const state = {
    seed: TerrainHiGAN.parseSeed().toString(),
    map: ["realistic", "elevation", "contour"].includes(hash.get("map"))
      ? hash.get("map")
      : "realistic",
    exaggeration: [1, 5, 10].includes(numeric("relief", 1)) ? numeric("relief", 1) : 1,
  };
  try {
    if (hash.has("seed")) state.seed = TerrainHiGAN.parseSeed(hash.get("seed")).toString();
  } catch {}
  let renderer,
    controller,
    walk,
    minimap,
    worker,
    renderFrame = 0,
    pointerFrame = 0,
    statsTimer = 0,
    urlTimer = 0,
    hudUpdated = 0,
    hudStatus = "",
    exporting = false;
  let pointer = null,
    pointerDown = null,
    measurementStart = null,
    measurementEnd = null;
  let readyTimer = 0;
  const showStatus = (label, phase, detail = label) => {
    if (element("status").textContent !== label) element("status").textContent = label;
    element("status").title = detail;
    element("render-indicator").dataset.state = phase;
    element("render-indicator").setAttribute("aria-label", `Globe ${label.toLowerCase()}`);
  };
  const distanceLabel = (meters) =>
    meters >= 1000
      ? `${(meters / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} km`
      : `${meters.toLocaleString(undefined, { maximumFractionDigits: 1 })} m`;
  const elevationLabel = (meters) => `${Math.round(meters).toLocaleString()} m`;
  const fail = (message) => {
    clearTimeout(readyTimer);
    showStatus("Error", "failed", message);
    console.error(message);
  };
  const updateURL = () => {
    if (urlTimer && walk?.active) return;
    clearTimeout(urlTimer);
    urlTimer = setTimeout(() => {
      urlTimer = 0;
      const view = renderer.view();
      const params = new URLSearchParams({
        globe: GLOBE_VERSION,
        seed: state.seed,
        map: state.map,
        lat: view.latitude.toFixed(6),
        lon: view.longitude.toFixed(6),
        altitude: view.altitude.toFixed(1),
        relief: state.exaggeration.toString(),
        up: renderer.camera.up
          .toArray()
          .map((value) => value.toFixed(8))
          .join(","),
        ...(walk?.active
          ? {
              mode: "walk",
              heading: walk.heading
                .toArray()
                .map((value) => value.toFixed(10))
                .join(","),
              pitch: walk.pitch.toFixed(6),
              speed: walk.requestedSpeed.toString(),
              anchor: walk.anchor
                .toArray()
                .map((value) => value.toFixed(10))
                .join(","),
              orbitRelief: walk.previousRelief.toString(),
            }
          : {}),
      });
      history.replaceState(null, "", `${location.pathname}${location.search}#${params}`);
    }, 200);
    element("flat-link").href = `./#seed=${state.seed}&map=${state.map}`;
  };
  const updateMeasurement = () => {
    const preview =
      measurementStart && !measurementEnd && pointer
        ? renderer.surfacePick(pointer.x, pointer.y)
        : null;
    const end = measurementEnd || preview;
    renderer.measure(measurementStart, end, !measurementEnd);
    element("distance").textContent =
      measurementStart && end ? distanceLabel(greatCircle(measurementStart, end)) : "--";
    element("clear-measurement").disabled = !measurementStart;
  };
  const clearMeasurement = () => {
    measurementStart = measurementEnd = null;
    updateMeasurement();
    renderer.draw();
  };
  const updatePointer = () => {
    if (!pointer) {
      element("position").textContent = element("elevation").textContent = "--";
      return;
    }
    const point = renderer.surfacePick(pointer.x, pointer.y);
    if (!point) {
      element("position").textContent = element("elevation").textContent = "--";
      return;
    }
    const latitude = (Math.asin(point[1]) * 180) / Math.PI;
    const longitude = (Math.atan2(-point[2], point[0]) * 180) / Math.PI;
    element("position").textContent = `${latitude.toFixed(2)}°, ${longitude.toFixed(2)}°`;
    const height = renderer.sample(point, controller.display);
    element("elevation").textContent = height === null ? "Loading" : elevationLabel(height);
    element("elevation").title =
      height < 0 && state.map === "realistic"
        ? "Generated seabed elevation; water surface is at sea level"
        : "Generated terrain elevation";
  };
  const updateStats = () => {
    if (walk?.active) {
      for (const id of ["view-min", "view-max", "view-mean", "view-stddev"])
        element(id).textContent = "--";
      canvas.dataset.samples = 0;
      return;
    }
    const tiles = controller.display;
    let minimum = Infinity,
      maximum = -Infinity,
      sum = 0,
      squares = 0,
      count = 0;
    for (let row = 0; row < 18; row++)
      for (let column = 0; column < 28; column++) {
        const point = renderer.pick(
          ((column + 0.5) * innerWidth) / 28,
          ((row + 0.5) * innerHeight) / 18,
        );
        if (!point) continue;
        const value = renderer.sample(point, tiles);
        if (value === null) continue;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
        sum += value;
        squares += value * value;
        count++;
      }
    const values = count
      ? [
          minimum,
          maximum,
          sum / count,
          Math.sqrt(Math.max(0, squares / count - (sum / count) ** 2)),
        ]
      : null;
    ["view-min", "view-max", "view-mean", "view-stddev"].forEach((id, index) => {
      element(id).textContent = values ? elevationLabel(values[index]) : "--";
      element(id).title = `${count} visible screen-ray samples; terrain heights include bathymetry`;
    });
    canvas.dataset.samples = count;
  };
  const updateView = () => {
    if (!renderer || !controller) return;
    cancelAnimationFrame(pointerFrame);
    pointerFrame = 0;
    controller.update();
    minimap?.draw();
    const now = performance.now();
    const hudState = `${walk.status}:${walk.mouseBound}:${walk.requestedSpeed}`;
    if (walk.active && !walk.paused && now - hudUpdated < 100 && hudStatus === hudState) {
      renderer.draw();
      return;
    }
    hudUpdated = now;
    hudStatus = hudState;
    state.exaggeration = renderer.exaggeration;
    element("exaggeration").value = state.exaggeration;
    element("exaggeration").disabled = walk.active;
    element("toggle-walk").disabled = !walk.active && renderer.view().altitude >= WALK_CEILING;
    element("toggle-walk").textContent = walk.active ? "Exit walk" : "Walk";
    element("walk-controls").hidden = !walk.active;
    element("walk-hint").textContent = walk.active
      ? renderer.view().altitude < 0
        ? "Underwater"
        : "20 km detail · elevation-aware horizon"
      : walk.status !== "Paused"
        ? walk.status
        : renderer.view().altitude >= WALK_CEILING
          ? "Zoom below 20 km to walk"
          : "Mouse walk available";
    element("walk-status").textContent = walk.status;
    element("walk-mouse").textContent = walk.mouseBound ? "Bound" : "Free";
    element("walk-requested").textContent = `${walk.requestedSpeed.toFixed(1)} m/s`;
    element("walk-actual").textContent = `${walk.speed.toFixed(1)} m/s`;
    element("walk-ground").textContent = walk.waiting
      ? "Loading"
      : `${walk.aboveGround.toFixed(1)} m`;
    element("walk-speed").value = walk.requestedSpeed;
    element("pause-walk").textContent = walk.paused ? "Resume" : "Pause";
    element("pause-walk").disabled = !walk.prepared;
    element("export-scene").disabled = !walk.prepared || exporting;
    element("export-scene").textContent = exporting ? "Exporting…" : "Export terrain GLB";
    document.body.classList.toggle("walk-active", walk.active);
    Object.assign(canvas.dataset, {
      walk: walk.active,
      walkPaused: walk.paused,
      walkMouseBound: walk.mouseBound,
      walkSpeed: walk.speed,
      walkAltitude: renderer.view().altitude,
      walkClearance: walk.aboveGround,
      walkWaiting: walk.waiting,
      walkPrepared: walk.prepared,
      walkSceneTiles: walk.selection?.leaves.length || 0,
    });
    element("altitude").textContent = distanceLabel(renderer.view().altitude);
    const first = renderer.pick(innerWidth / 2, innerHeight / 2);
    const second = renderer.pick(innerWidth / 2 + 1, innerHeight / 2);
    const scale = first && second ? greatCircle(first, second) : null;
    element("resolution").textContent = scale
      ? `${(scale / devicePixelRatio).toLocaleString(undefined, { maximumFractionDigits: 2 })} m`
      : "--";
    element("scale-label").textContent = scale ? distanceLabel(scale * 88) : "--";
    updatePointer();
    updateMeasurement();
    renderer.draw();
    updateURL();
    clearTimeout(statsTimer);
    statsTimer = setTimeout(updateStats, 180);
  };
  const queueView = () => {
    if (renderFrame) return;
    renderFrame = requestAnimationFrame(() => {
      renderFrame = 0;
      updateView();
    });
  };
  const queuePointer = () => {
    if (renderFrame || pointerFrame) return;
    pointerFrame = requestAnimationFrame(() => {
      pointerFrame = 0;
      updatePointer();
      if (measurementStart && !measurementEnd) {
        updateMeasurement();
        renderer.draw();
      }
    });
  };
  renderer = new GlobeRenderer(canvas, queueView);
  renderer.map = state.map;
  renderer.exaggeration = state.exaggeration;
  renderer.setView(
    Math.max(-90, Math.min(90, numeric("lat", 20))),
    numeric("lon", -25),
    numeric("altitude", RADIUS * 2),
  );
  if (hash.has("up")) {
    const up = hash.get("up").split(",").map(Number);
    if (up.length === 3 && up.every(Number.isFinite) && Math.hypot(...up) > 0.5) {
      renderer.camera.up.set(...up).normalize();
      renderer.updateCamera();
    }
  }
  controller = new GlobeTileController(
    renderer,
    (data) => worker.postMessage(data),
    (status) => {
      if (status.error) {
        if (walk?.active) walk.pause();
        fail(status.error);
        return;
      }
      if (status.ready) {
        if (!readyTimer)
          readyTimer = setTimeout(() => {
            readyTimer = 0;
            if (!controller.sequence.length && !controller.failed) showStatus("Ready", "ready");
          }, 300);
      } else {
        clearTimeout(readyTimer);
        readyTimer = 0;
        showStatus("Loading", "rendering");
      }
      element("visible-tiles").textContent =
        `${status.visible} / ${status.visible + status.pending}`;
      const levels = controller.display.map((tile) => tile.level);
      if (levels.length) {
        const low = Math.min(...levels),
          high = Math.max(...levels);
        element("model").textContent =
          low === high ? TerrainHiGAN.modelName(high) : `z${low}–${high}${high > 10 ? "s" : ""}`;
        canvas.dataset.maxModelLevel = high;
      }
      canvas.dataset.visibleTiles = status.visible;
      canvas.dataset.cachedTiles = status.cached;
      canvas.dataset.ready = status.ready;
      canvas.dataset.tileCapacity = status.capacity;
      canvas.dataset.targetModelLevel = status.targetLevel;
      canvas.dataset.samplePixels = status.samplePixels;
    },
  );
  walk = new GlobeWalk(renderer, controller, queueView);
  walk.bindInput(canvas);
  minimap = new WalkMinimap(walk);
  element("toggle-walk").addEventListener("click", () => {
    clearMeasurement();
    if (walk.active) walk.exit();
    else walk.enter();
  });
  element("pause-walk").addEventListener("click", () => {
    if (walk.paused) walk.resume();
    else walk.pause();
    queueView();
  });
  element("walk-speed").addEventListener("input", () => {
    walk.requestedSpeed = Math.max(
      0,
      Math.min(WALK_MAX_SPEED, Number(element("walk-speed").value)),
    );
    queueView();
  });
  element("export-scene").addEventListener("click", async () => {
    walk.pause();
    exporting = true;
    queueView();
    try {
      await downloadWalkingScene(walk);
    } catch (error) {
      element("export-scene").title = error.message;
      showStatus("Export failed", "failed", error.message);
    } finally {
      exporting = false;
      queueView();
    }
  });
  const startWorker = () => {
    if (worker && !controller.failed) {
      worker.postMessage({ type: "reset" });
      controller.reset(state.seed);
      return;
    }
    worker?.terminate();
    const options = new URLSearchParams(location.search);
    options.set("bundle", TerrainHiGAN.bundle);
    worker = new Worker(`./globe-worker.js?${options}`);
    worker.onerror = (event) => controller.fail(event.message || "Globe worker failed");
    worker.onmessage = ({ data }) => {
      if (data.type === "bundle-changed") {
        location.reload();
        return;
      }
      if (data.type === "progress") {
        if (controller.failed || !controller.sequence.length) return;
        canvas.dataset.modelRuns = data.metrics.modelRuns;
        return;
      }
      if (controller.failed || data.id !== controller.active?.id) return;
      if (data.type === "tile") {
        element("backend").textContent =
          data.provider.name === "wasm" ? "CPU / WASM" : "WebGPU + WASM";
        element("backend").title =
          "Shared ONNX runtime policy: WASM z0–z15; optional WebGPU z16–z20";
        element("threads").textContent = `${data.metrics.inferenceWorkers} × 1`;
        Object.assign(canvas.dataset, {
          modelRuns: data.metrics.modelRuns,
          cacheBytes: data.metrics.cacheBytes,
          cacheHits: data.metrics.cacheHits,
          maxParallelRuns: data.metrics.maxParallelRuns,
        });
      }
      controller.receive(data, false);
      queueView();
    };
    controller.reset(state.seed);
  };
  const selectPalette = () => {
    document
      .querySelectorAll("[data-map]")
      .forEach((button) =>
        button.setAttribute("aria-pressed", String(button.dataset.map === state.map)),
      );
    renderer.restyle(controller.tiles.values(), state.map, state.exaggeration);
    queueView();
  };
  document.querySelectorAll("[data-map]").forEach((button) =>
    button.addEventListener("click", () => {
      state.map = button.dataset.map;
      selectPalette();
    }),
  );
  element("exaggeration").value = state.exaggeration;
  element("exaggeration").addEventListener("change", () => {
    state.exaggeration = Number(element("exaggeration").value);
    selectPalette();
  });
  const setSeed = (seed) => {
    if (walk.active) walk.pause();
    state.seed = seed;
    element("seed").value = seed;
    clearMeasurement();
    startWorker();
    queueView();
  };
  element("seed").value = state.seed;
  element("seed").addEventListener("change", () => {
    try {
      setSeed(TerrainHiGAN.parseSeed(element("seed").value).toString());
    } catch {
      element("seed").value = state.seed;
    }
  });
  element("randomize").addEventListener("click", () => {
    const words = crypto.getRandomValues(new Uint32Array(2));
    setSeed(((BigInt(words[0]) << 32n) | BigInt(words[1])).toString());
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (walk.active) {
      canvas.focus();
      return;
    }
    if (event.button !== 0) return;
    pointerDown = {
      x: event.clientX,
      y: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      id: event.pointerId,
    };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    pointer = { x: event.clientX, y: event.clientY };
    if (walk.active) {
      return;
    }
    if (pointerDown && Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 4)
      pointerDown.moved = true;
    if (pointerDown?.moved) {
      renderer.drag(pointerDown.lastX, pointerDown.lastY, event.clientX, event.clientY);
      pointerDown.lastX = event.clientX;
      pointerDown.lastY = event.clientY;
    }
    queuePointer();
  });
  canvas.addEventListener(
    "pointerup",
    (event) => {
      if (walk.active) return;
      if (pointerDown?.id === event.pointerId && !pointerDown.moved) {
        const point = renderer.surfacePick(event.clientX, event.clientY);
        if (point) {
          if (!measurementStart || measurementEnd) {
            measurementStart = point;
            measurementEnd = null;
          } else measurementEnd = point;
          updateMeasurement();
          renderer.draw();
        }
      }
      pointerDown = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      canvas.classList.remove("dragging");
    },
    { capture: true },
  );
  for (const name of ["pointercancel", "lostpointercapture"])
    canvas.addEventListener(name, () => {
      pointerDown = null;
      canvas.classList.remove("dragging");
    });
  canvas.addEventListener("pointerleave", (event) => {
    if (walk.active) {
      walk.mouse.x = walk.mouse.y = 0;
      if (!event.relatedTarget?.closest?.(".shell") && !walk.paused) walk.pause();
    }
    pointer = null;
    updatePointer();
    updateMeasurement();
    renderer.draw();
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const delta =
        event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
      if (walk.active) {
        walk.adjustSpeed(Math.exp(Math.max(-0.5, Math.min(0.5, -delta * 0.0015))));
        return;
      }
      renderer.zoom(
        Math.exp(Math.max(-0.5, Math.min(0.5, delta * 0.0015))),
        event.clientX,
        event.clientY,
      );
      queueView();
    },
    { passive: false },
  );
  element("clear-measurement").addEventListener("click", clearMeasurement);
  const shortcuts = element("shortcuts-dialog");
  const toggleShortcuts = () => {
    if (walk.active) walk.pause();
    if (shortcuts.open) shortcuts.close();
    else shortcuts.showModal();
    element("toggle-shortcuts").setAttribute("aria-expanded", String(shortcuts.open));
  };
  element("toggle-shortcuts").addEventListener("click", toggleShortcuts);
  element("close-shortcuts").addEventListener("click", () => shortcuts.close());
  shortcuts.addEventListener("close", () =>
    element("toggle-shortcuts").setAttribute("aria-expanded", "false"),
  );
  addEventListener("keydown", (event) => {
    if (walk.active && event.key === "Escape") {
      walk.pause();
      return;
    }
    if (
      walk.active &&
      event.code === "Space" &&
      !event.target.matches("input:not([type=range]), select")
    ) {
      event.preventDefault();
      if (event.repeat) return;
      if (walk.paused) walk.resume();
      else walk.pause();
      return;
    }
    if (event.target.matches("input, select") || event.ctrlKey || event.metaKey || event.altKey)
      return;
    if (event.key === "?") {
      event.preventDefault();
      toggleShortcuts();
      return;
    }
    if (shortcuts.open) return;
    if (event.key === "Escape") {
      if (document.body.classList.contains("map-only")) document.body.classList.remove("map-only");
      else clearMeasurement();
    }
    if (event.key.toLowerCase() === "f") document.body.classList.toggle("map-only");
    if (event.key.toLowerCase() === "r") element("randomize").click();
    if (event.key === "Home") {
      event.preventDefault();
      if (walk.active) walk.exit();
      renderer.setView(20, -25, RADIUS * 2);
      queueView();
    }
    if (["+", "=", "-"].includes(event.key)) {
      event.preventDefault();
      if (walk.active) {
        walk.adjustSpeed(event.key === "-" ? 1 / 1.25 : 1.25);
        return;
      }
      renderer.zoom(event.key === "-" ? 1.35 : 1 / 1.35);
      queueView();
    }
  });
  addEventListener("resize", () => {
    renderer.resize();
    queueView();
  });
  element("bundle").textContent = TerrainHiGAN.bundle.slice(0, 8);
  let checkingBundle = false;
  const checkBundle = async () => {
    if (checkingBundle) return;
    checkingBundle = true;
    try {
      const response = await fetch("./assets/models/manifest.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`Model manifest returned HTTP ${response.status}`);
      if ((await response.json()).bundle !== TerrainHiGAN.bundle) location.reload();
    } catch (error) {
      console.warn("Could not check globe bundle", error);
    } finally {
      checkingBundle = false;
    }
  };
  setInterval(checkBundle, 15000);
  addEventListener("focus", checkBundle);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && walk.active) walk.pause();
    if (!document.hidden) checkBundle();
  });
  addEventListener("blur", () => {
    if (walk.active) walk.pause();
  });
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    controller.fail("Graphics context lost; reload to retry");
  });
  selectPalette();
  if (hash.get("mode") === "walk" && numeric("altitude", RADIUS * 2) < WALK_CEILING) {
    element("experimental").open = true;
    const heading = (hash.get("heading") || "").split(",").map(Number);
    const anchor = (hash.get("anchor") || "").split(",").map(Number);
    walk.requestedSpeed = Math.max(0, Math.min(WALK_MAX_SPEED, numeric("speed", 1.4)));
    walk.enter({
      altitude: numeric("altitude", 1000),
      heading: heading.length === 3 && heading.every(Number.isFinite) ? heading : null,
      pitch: numeric("pitch", 0),
      anchor:
        anchor.length === 3 && anchor.every(Number.isFinite) && Math.hypot(...anchor) > 0.5
          ? anchor
          : null,
    });
    walk.previousRelief = [1, 5, 10].includes(numeric("orbitRelief", 1))
      ? numeric("orbitRelief", 1)
      : 1;
    walk.pause();
  }
  startWorker();
  updateView();
})();
