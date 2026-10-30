(() => {
  "use strict";
  const canvas = document.querySelector("#world");
  let walkView = null;
  const context = canvas.getContext("2d", { alpha: false });
  const ui = Object.fromEntries(
    [
      "backend",
      "threads",
      "seed",
      "status",
      "render-indicator",
      "visible-tiles",
      "shortcuts-dialog",
      "toggle-shortcuts",
      "model",
      "bundle",
      "resolution",
      "position",
      "elevation",
      "view-min",
      "view-max",
      "view-mean",
      "view-stddev",
      "scale-label",
      "distance",
      "clear-measurement",
    ].map((id) => [id, document.getElementById(id)]),
  );
  // Match the viewport area to Earth's surface using the nearest model level.
  const initialZoom = () => {
    const area = innerWidth * innerHeight * TerrainHiGAN.displayPixelRatio() ** 2;
    const earthArea = 4 * Math.PI * 6371000 ** 2;
    return Math.max(
      0,
      Math.min(
        20,
        Math.round(Math.log2(TerrainHiGAN.metersPerPixel(0) * Math.sqrt(area / earthArea))),
      ),
    );
  };
  const state = {
    seed: TerrainHiGAN.parseSeed(),
    zoom: initialZoom(),
    x: 0,
    y: 0,
    colormap: "realistic",
    dragging: false,
    pointer: null,
    measurementStart: null,
    measurementEnd: null,
  };
  const { metersPerPixel, clamp, elevationMeters, summarizeElevations, TileController } =
    TerrainHiGAN;
  const formatDistance = (meters) =>
    meters >= 1000 ? `${Math.round(meters / 1000)} km` : `${Math.max(1, Math.round(meters))} m`;
  const updateUrl = () => {
    const hash = new URLSearchParams({
      seed: state.seed.toString(),
      z: String(state.zoom),
      x: String(Math.round(state.x)),
      y: String(Math.round(state.y)),
      map: state.colormap,
    });
    history.replaceState(null, "", `#${hash}`);
  };
  const loadUrl = () => {
    const saved = new URLSearchParams(location.hash.slice(1));
    try {
      if (saved.has("seed")) state.seed = TerrainHiGAN.parseSeed(saved.get("seed"));
    } catch (_) {}
    const savedZoom = Number(saved.get("z"));
    if (saved.has("z") && Number.isFinite(savedZoom)) state.zoom = clamp(savedZoom, 0, 20);
    state.x = Number(saved.get("x")) || 0;
    state.y = Number(saved.get("y")) || 0;
    if (!Number.isFinite(state.x)) state.x = 0;
    if (!Number.isFinite(state.y)) state.y = 0;
    if (["realistic", "elevation", "contour"].includes(saved.get("map")))
      state.colormap = saved.get("map");
  };
  let modelWorker = null,
    modelBundle = TerrainHiGAN.bundle,
    bundleCheckPending = false;
  let pixelRatio = TerrainHiGAN.displayPixelRatio();
  // Display tiles group cached model samples at one sample per physical canvas pixel.
  const tileSize = TerrainHiGAN.mobileView ? 128 : 256;
  const modelZoom = () => Math.ceil(state.zoom);
  const displayScale = () => 2 ** (state.zoom - modelZoom());
  const updateProgress = () =>
    TerrainHiGAN.setRenderProgress(
      controller.sequence.filter((tile) => controller.tiles.has(tile.key)).length,
      controller.sequence.length,
      !!controller.error,
    );
  const showRenderState = (phase, message) => {
    ui.status.textContent = message;
    ui["render-indicator"].dataset.state = phase;
    ui["render-indicator"].title = message;
    ui["render-indicator"].setAttribute("aria-label", `Terrain: ${message}`);
  };
  const controller = new TileController({
    maxCacheTiles: TerrainHiGAN.mobileView ? 128 : 512,
    send: (request) => modelWorker.postMessage(request),
    onTile: receiveTile,
    onStatus: (_message, ready) => {
      if (walkView) return;
      showRenderState(ready ? "ready" : "rendering", ready ? "Ready" : "Rendering");
      const loaded = controller.sequence.filter((tile) => controller.tiles.has(tile.key)).length;
      updateProgress();
      ui["visible-tiles"].textContent = `${loaded} / ${controller.sequence.length}`;
      if (ready) {
        drawTiles();
        updateCursorElevation();
        queueViewStats();
      }
    },
    onError: (message) => {
      TerrainHiGAN.setRenderError(message);
      showRenderState("failed", `Model failed: ${message}`);
      console.error("Terrain model failed:", message);
    },
  });
  const visibleTiles = controller.tiles;
  let tileSequence = [],
    viewport = null;
  const tileKey = (tileX, tileY) =>
    `${state.seed}:${modelZoom()}:${state.colormap}:${tileX}:${tileY}`;
  const makeImage = (data) => {
    const image = new ImageData(new Uint8ClampedArray(data.pixels), data.width, data.height);
    const surface = document.createElement("canvas");
    surface.width = image.width;
    surface.height = image.height;
    surface.getContext("2d").putImageData(image, 0, 0);
    return surface;
  };
  const formatElevation = (value) => `${Math.round(value).toLocaleString()} m`;
  let statsTimer = 0,
    tileRevision = 0,
    cursor = null;
  const updateCursorElevation = () => {
    if (!cursor || !viewport) {
      ui.elevation.textContent = "--";
      return;
    }
    const x = Math.floor(viewport.left + (cursor.x * pixelRatio) / displayScale()),
      y = Math.floor(viewport.top + (cursor.y * pixelRatio) / displayScale());
    const tile = visibleTiles.get(tileKey(Math.floor(x / tileSize), Math.floor(y / tileSize)));
    ui.elevation.textContent = tile
      ? formatElevation(elevationMeters(tile.elevations[(y - tile.y) * tileSize + x - tile.x]))
      : "Loading";
  };
  const updateViewStats = () => {
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      sumSquares = 0,
      count = 0,
      loaded = 0;
    for (const requested of tileSequence) {
      const tile = visibleTiles.get(requested.key);
      if (!tile) continue;
      loaded++;
      const left = Math.max(tile.x, Math.ceil(viewport.left)) - tile.x,
        top = Math.max(tile.y, Math.ceil(viewport.top)) - tile.y;
      const right =
          Math.min(tile.x + tileSize, Math.ceil(viewport.left + canvas.width / displayScale())) -
          tile.x,
        bottom =
          Math.min(tile.y + tileSize, Math.ceil(viewport.top + canvas.height / displayScale())) -
          tile.y;
      const stats =
        left === 0 && top === 0 && right === tileSize && bottom === tileSize
          ? tile.stats
          : summarizeElevations(tile.elevations, tileSize, { left, top, right, bottom });
      min = Math.min(min, stats.min);
      max = Math.max(max, stats.max);
      sum += stats.sum;
      sumSquares += stats.sumSquares;
      count += stats.count;
    }
    if (!count) {
      [ui["view-min"], ui["view-max"], ui["view-mean"], ui["view-stddev"]].forEach((element) => {
        element.textContent = "Loading";
      });
      return;
    }
    const mean = sum / count,
      variance = Math.max(0, sumSquares / count - mean * mean);
    const suffix = loaded === tileSequence.length ? "" : ` (${loaded}/${tileSequence.length})`;
    ui["view-min"].textContent = formatElevation(min) + suffix;
    ui["view-max"].textContent = formatElevation(max) + suffix;
    ui["view-mean"].textContent = formatElevation(mean) + suffix;
    ui["view-stddev"].textContent = formatElevation(Math.sqrt(variance)) + suffix;
  };
  const queueViewStats = () => {
    clearTimeout(statsTimer);
    statsTimer = setTimeout(() => {
      statsTimer = 0;
      if (state.dragging || pinch || zoomFrame) {
        queueViewStats();
        return;
      }
      updateViewStats();
    }, 180);
  };
  // Endpoints live in world meters, independent of zoom and display pixel density.
  const cursorWorldPoint = () => {
    const mpp = metersPerPixel(modelZoom());
    return {
      x: (viewport.left + (cursor.x * pixelRatio) / displayScale()) * mpp,
      y: (viewport.top + (cursor.y * pixelRatio) / displayScale()) * mpp,
    };
  };
  const measurementLabel = (start, end) => {
    const meters = Math.hypot(end.x - start.x, end.y - start.y);
    const kilometers = meters >= 1000;
    return `${(kilometers ? meters / 1000 : meters).toLocaleString(undefined, {
      maximumFractionDigits: kilometers ? 2 : 1,
    })} ${kilometers ? "km" : "m"}`;
  };
  const drawMeasurement = () => {
    const start = state.measurementStart;
    const end = state.measurementEnd || (start && cursor ? cursorWorldPoint() : null);
    ui.distance.textContent = start && end ? measurementLabel(start, end) : "--";
    ui["clear-measurement"].disabled = !start;
    if (!start) return;
    const mpp = metersPerPixel(modelZoom());
    const screenPoint = (point) => ({
      x: ((point.x / mpp - viewport.left) * displayScale()) / pixelRatio,
      y: ((point.y / mpp - viewport.top) * displayScale()) / pixelRatio,
    });
    const a = screenPoint(start),
      b = end ? screenPoint(end) : a;
    context.save();
    context.scale(pixelRatio, pixelRatio);
    context.lineCap = "round";
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.strokeStyle = "#08151a";
    context.lineWidth = 5;
    context.stroke();
    context.strokeStyle = "#ffe07a";
    context.lineWidth = 2;
    context.setLineDash(state.measurementEnd ? [] : [6, 5]);
    context.stroke();
    context.setLineDash([]);
    for (const point of end ? [a, b] : [a]) {
      context.beginPath();
      context.arc(point.x, point.y, 4, 0, Math.PI * 2);
      context.fillStyle = "#ffe07a";
      context.fill();
      context.strokeStyle = "#08151a";
      context.lineWidth = 2;
      context.stroke();
    }
    if (end) {
      const label = measurementLabel(start, end);
      context.font = "600 12px ui-monospace, monospace";
      const width = context.measureText(label).width + 16;
      const x = clamp((a.x + b.x) / 2 - width / 2, 8, innerWidth - width - 8);
      const y = clamp((a.y + b.y) / 2 - 34, 8, innerHeight - 32);
      context.fillStyle = "rgba(7, 21, 25, 0.92)";
      context.fillRect(x, y, width, 24);
      context.fillStyle = "#ffe07a";
      context.textBaseline = "middle";
      context.fillText(label, x + 8, y + 12);
    }
    context.restore();
  };
  const clearMeasurement = () => {
    state.measurementStart = state.measurementEnd = null;
    drawTiles();
  };
  // Join tiles on integer source pixels before filtering the viewport. Scaling
  // separate tile images leaves partially covered edge pixels at fractional zoom.
  const tileSurface = document.createElement("canvas");
  const tileContext = tileSurface.getContext("2d", { alpha: false });
  let mosaic = null;
  let mosaicBuilds = 0;
  const drawTiles = () => {
    if (!viewport) return;
    const ratio = displayScale();
    const left = Math.floor(viewport.left) - 1;
    const top = Math.floor(viewport.top) - 1;
    const width = Math.ceil(canvas.width / ratio) + 3;
    const height = Math.ceil(canvas.height / ratio) + 3;
    const identity = state.seed + ":" + state.colormap + ":" + modelZoom();
    const neededWidth = Math.ceil((width + 256) / tileSize) * tileSize;
    const neededHeight = Math.ceil((height + 256) / tileSize) * tileSize;
    if (
      !mosaic ||
      mosaic.identity !== identity ||
      tileSurface.width < neededWidth ||
      tileSurface.height < neededHeight
    ) {
      tileSurface.width = neededWidth;
      tileSurface.height = neededHeight;
      mosaic = null;
    }
    const surfaceWidth = tileSurface.width,
      surfaceHeight = tileSurface.height;
    // A native-pixel margin lets input frames pan by cropping the existing mosaic.
    if (
      !mosaic ||
      mosaic.identity !== identity ||
      mosaic.revision !== tileRevision ||
      left < mosaic.left ||
      top < mosaic.top ||
      left + width > mosaic.left + surfaceWidth ||
      top + height > mosaic.top + surfaceHeight
    ) {
      mosaic = { identity, revision: tileRevision, left: left - 128, top: top - 128 };
      tileContext.fillStyle = "#082c50";
      tileContext.fillRect(0, 0, surfaceWidth, surfaceHeight);
      tileContext.imageSmoothingEnabled = false;
      const tiles = [...visibleTiles.values()].filter(
        (tile) => tile.seed === state.seed.toString() && tile.map === state.colormap,
      );
      tiles.sort((a, b) => Math.abs(b.zoom - modelZoom()) - Math.abs(a.zoom - modelZoom()));
      for (const tile of tiles) {
        const scale = 2 ** (modelZoom() - tile.zoom);
        const x = Math.round(tile.x * scale - mosaic.left);
        const y = Math.round(tile.y * scale - mosaic.top);
        const right = Math.round((tile.x + tileSize) * scale - mosaic.left);
        const bottom = Math.round((tile.y + tileSize) * scale - mosaic.top);
        if (right > 0 && bottom > 0 && x < surfaceWidth && y < surfaceHeight)
          tileContext.drawImage(tile.image, x, y, right - x, bottom - y);
      }
      canvas.dataset.mosaicBuilds = ++mosaicBuilds;
    }
    context.imageSmoothingEnabled = ratio !== 1;
    context.drawImage(
      tileSurface,
      left - mosaic.left,
      top - mosaic.top,
      width,
      height,
      (left - viewport.left) * ratio,
      (top - viewport.top) * ratio,
      width * ratio,
      height * ratio,
    );
    drawMeasurement();
  };
  function receiveTile(data, tile, identity) {
    tileRevision++;
    const elevations = new Float32Array(data.elevations);
    ui.model.textContent = data.model;
    showProvider(data.provider);
    canvas.dataset.modelRuns = data.metrics.modelRuns;
    canvas.dataset.cacheHits = data.metrics.cacheHits;
    canvas.dataset.tileModelRuns = data.metrics.tileModelRuns;
    canvas.dataset.inferenceWorkers = data.metrics.inferenceWorkers;
    ui.threads.textContent = `${data.metrics.inferenceWorkers} (${data.metrics.inferenceWorkers} workers x 1)`;
    canvas.dataset.maxParallelRuns = data.metrics.maxParallelRuns;
    canvas.dataset.coalescedBlocks = data.metrics.coalescedBlocks;
    const result = {
      ...tile,
      image: makeImage(data),
      elevations,
      stats: data.elevationStats,
      model: data.model,
      provider: data.provider,
      ...identity,
    };
    // Drawing runs after the controller has stored the returned tile.
    requestAnimationFrame(() => {
      drawTiles();
      updateCursorElevation();
      queueViewStats();
    });
    return result;
  }
  function showProvider(provider) {
    ui.backend.textContent = provider
      ? provider.name === "wasm"
        ? "CPU / WASM"
        : "WebGPU + WASM"
      : "Loading";
    ui.backend.title = provider?.detail || "";
    canvas.dataset.gpuError = provider?.detail || "";
  }
  const failModel = (message) => controller.fail(message);
  const onModelWorkerMessage = ({ data }) => {
    if (data.type === "bundle-changed") {
      TerrainHiGAN.notifyUpdate();
      return;
    }
    if (data.type === "phase") {
      TerrainHiGAN.setRenderPhase(data.message);
      return;
    }
    controller.receive(data);
  };
  const resize = () => {
    pixelRatio = TerrainHiGAN.displayPixelRatio();
    const width = Math.round(innerWidth * pixelRatio),
      height = Math.round(innerHeight * pixelRatio);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvas.style.width = `${innerWidth}px`;
    canvas.style.height = `${innerHeight}px`;
    render();
  };
  const updateUi = () => {
    ui.seed.value = state.seed.toString();
    document.getElementById("globe-link").href =
      `./globe.html#seed=${state.seed}&map=${state.colormap}`;
    ui.resolution.textContent = `${formatDistance(metersPerPixel(state.zoom))} / px`;
    ui.position.textContent = `${Math.round(state.x)}, ${Math.round(state.y)}`;
    const scale = metersPerPixel(state.zoom) * pixelRatio * TerrainHiGAN.scalePixels;
    ui["scale-label"].textContent = formatDistance(scale);
    document
      .querySelectorAll("[data-map]")
      .forEach((button) =>
        button.setAttribute("aria-pressed", String(button.dataset.map === state.colormap)),
      );
  };
  const render = () => {
    if (walkView) return;
    updateUi();
    updateUrl();
    const mpp = metersPerPixel(modelZoom());
    const centerX = state.x / mpp,
      centerY = state.y / mpp;
    viewport = {
      left: centerX - canvas.width / (2 * displayScale()),
      top: centerY - canvas.height / (2 * displayScale()),
    };
    const firstX = Math.floor(viewport.left / tileSize),
      firstY = Math.floor(viewport.top / tileSize),
      lastX = Math.ceil((viewport.left + canvas.width / displayScale()) / tileSize) - 1,
      lastY = Math.ceil((viewport.top + canvas.height / displayScale()) / tileSize) - 1;
    tileSequence = [];
    for (let tileY = firstY; tileY <= lastY; tileY++)
      for (let tileX = firstX; tileX <= lastX; tileX++)
        tileSequence.push({
          tileX,
          tileY,
          x: tileX * tileSize,
          y: tileY * tileSize,
          width: tileSize,
          height: tileSize,
          key: tileKey(tileX, tileY),
        });
    tileSequence.sort(
      (left, right) =>
        Math.hypot(left.x + tileSize / 2 - centerX, left.y + tileSize / 2 - centerY) -
        Math.hypot(right.x + tileSize / 2 - centerX, right.y + tileSize / 2 - centerY),
    );
    const displayed = tileSequence.map((tile) => visibleTiles.get(tile.key)).find(Boolean);
    ui.model.textContent = displayed?.model || "Loading";
    showProvider(displayed?.provider);
    drawTiles();
    queueViewStats();
    updateCursorElevation();
    controller.setView(tileSequence, {
      seed: state.seed.toString(),
      map: state.colormap,
      zoom: modelZoom(),
    });
    updateProgress();
  };
  let renderFrame = 0;
  const queueRender = () => {
    if (renderFrame) return;
    renderFrame = requestAnimationFrame(() => {
      renderFrame = 0;
      render();
    });
  };
  const moveZoom = (delta, clientX, clientY) => {
    const next = clamp(state.zoom + delta, 0, 20);
    if (next === state.zoom) return false;
    const before = metersPerPixel(state.zoom),
      after = metersPerPixel(next);
    state.x += (clientX - innerWidth / 2) * pixelRatio * (before - after);
    state.y += (clientY - innerHeight / 2) * pixelRatio * (before - after);
    state.zoom = next;
    return true;
  };
  let zoomFrame = 0,
    zoomTarget = state.zoom;
  const stopZoom = () => {
    cancelAnimationFrame(zoomFrame);
    zoomFrame = 0;
    zoomTarget = state.zoom;
  };
  const zoomAt = (delta, clientX = innerWidth / 2, clientY = innerHeight / 2) => {
    zoomTarget = clamp((zoomFrame ? zoomTarget : state.zoom) + delta, 0, 20);
    cancelAnimationFrame(zoomFrame);
    let previous = performance.now();
    const step = (now) => {
      const remaining = zoomTarget - state.zoom;
      const amount =
        Math.abs(remaining) < 0.001
          ? remaining
          : remaining * (1 - Math.exp(-(now - previous) / 45));
      previous = now;
      moveZoom(amount, clientX, clientY);
      render();
      zoomFrame = state.zoom === zoomTarget ? 0 : requestAnimationFrame(step);
    };
    zoomFrame = requestAnimationFrame(step);
  };
  const touches = new Map();
  let pinch = null;
  const touchGesture = () => {
    const [a, b] = [...touches.values()];
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      distance: Math.max(1, Math.hypot(b.x - a.x, b.y - a.y)),
    };
  };
  const beginPinch = () => {
    const gesture = touchGesture();
    const ratio = metersPerPixel(state.zoom) * pixelRatio;
    return {
      ...gesture,
      zoom: state.zoom,
      worldX: state.x + (gesture.x - innerWidth / 2) * ratio,
      worldY: state.y + (gesture.y - innerHeight / 2) * ratio,
    };
  };
  canvas.addEventListener("pointerdown", (event) => {
    stopZoom();
    if (event.pointerType === "touch") {
      touches.set(event.pointerId, { x: event.clientX, y: event.clientY, id: event.pointerId });
      canvas.setPointerCapture(event.pointerId);
      if (touches.size >= 2) {
        pinch = beginPinch();
        state.dragging = false;
        state.pointer = null;
        canvas.classList.add("dragging");
        return;
      }
    }
    if (!event.isPrimary || event.button !== 0) return;
    cursor = { x: event.clientX, y: event.clientY };
    updateCursorElevation();
    state.dragging = true;
    state.pointer = { x: event.clientX, y: event.clientY, id: event.pointerId };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (touches.has(event.pointerId)) {
      touches.set(event.pointerId, { x: event.clientX, y: event.clientY, id: event.pointerId });
      if (pinch && touches.size >= 2) {
        const next = touchGesture();
        const target = clamp(pinch.zoom + Math.log2(next.distance / pinch.distance), 0, 20);
        state.zoom = target;
        const ratio = metersPerPixel(state.zoom) * pixelRatio;
        state.x = pinch.worldX - (next.x - innerWidth / 2) * ratio;
        state.y = pinch.worldY - (next.y - innerHeight / 2) * ratio;
        queueRender();
        return;
      }
    }
    if (!event.isPrimary && state.pointer?.id !== event.pointerId) return;
    cursor = { x: event.clientX, y: event.clientY };
    if (state.dragging) {
      if (!canvas.classList.contains("dragging")) {
        if (Math.hypot(event.clientX - state.pointer.x, event.clientY - state.pointer.y) < 4)
          return;
        canvas.classList.add("dragging");
      }
      const ratio = metersPerPixel(state.zoom) * pixelRatio;
      state.x -= (event.clientX - state.pointer.x) * ratio;
      state.y -= (event.clientY - state.pointer.y) * ratio;
      state.pointer.x = event.clientX;
      state.pointer.y = event.clientY;
      queueRender();
    } else {
      updateCursorElevation();
      if (state.measurementStart && !state.measurementEnd) drawTiles();
    }
  });
  const endTouch = (event) => {
    if (!touches.delete(event.pointerId) || !pinch) return false;
    if (touches.size < 2) {
      pinch = null;
      state.pointer = touches.values().next().value || null;
      state.dragging = !!state.pointer;
      canvas.classList.toggle("dragging", state.dragging);
    } else pinch = beginPinch();
    return true;
  };
  canvas.addEventListener("pointerup", (event) => {
    if (endTouch(event)) {
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      return;
    }
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    cursor = { x: event.clientX, y: event.clientY };
    if (!canvas.classList.contains("dragging") && viewport) {
      if (!state.measurementStart || state.measurementEnd) {
        state.measurementStart = cursorWorldPoint();
        state.measurementEnd = null;
      } else state.measurementEnd = cursorWorldPoint();
      drawTiles();
    }
    state.dragging = false;
    state.pointer = null;
    canvas.releasePointerCapture(event.pointerId);
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("lostpointercapture", (event) => {
    if (endTouch(event) || state.pointer?.id !== event.pointerId) return;
    state.dragging = false;
    state.pointer = null;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("pointercancel", (event) => {
    if (endTouch(event)) return;
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    state.dragging = false;
    state.pointer = null;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("pointerleave", () => {
    cursor = null;
    updateCursorElevation();
    drawTiles();
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const delta =
        event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
      zoomAt(clamp(-delta * 0.002, -0.75, 0.75), event.clientX, event.clientY);
    },
    { passive: false },
  );
  document.getElementById("randomize").addEventListener("click", () => {
    stopZoom();
    exitWalk(false);
    state.seed = BigInt.asUintN(
      64,
      BigInt(Date.now()) * 6364136223846793005n + BigInt(Math.floor(Math.random() * 2 ** 32)),
    );
    state.x = state.y = 0;
    clearMeasurement();
    render();
  });
  ui.seed.addEventListener("change", () => {
    try {
      const seed = TerrainHiGAN.parseSeed(ui.seed.value);
      stopZoom();
      exitWalk(false);
      state.seed = seed;
      clearMeasurement();
      render();
    } catch (_) {
      ui.seed.value = state.seed.toString();
    }
  });
  document.querySelectorAll("[data-map]").forEach((button) =>
    button.addEventListener("click", () => {
      state.colormap = button.dataset.map;
      if (walkView) {
        walkView.restyle(state.colormap);
        updateUi();
        return;
      }
      render();
    }),
  );
  ui["clear-measurement"].addEventListener("click", clearMeasurement);
  const shortcuts = ui["shortcuts-dialog"];
  const toggleShortcuts = () => {
    if (shortcuts.open) shortcuts.close();
    else {
      shortcuts.showModal();
      ui["toggle-shortcuts"].setAttribute("aria-expanded", "true");
    }
  };
  ui["toggle-shortcuts"].addEventListener("click", toggleShortcuts);
  document.getElementById("close-shortcuts").addEventListener("click", () => shortcuts.close());
  shortcuts.addEventListener("close", () =>
    ui["toggle-shortcuts"].setAttribute("aria-expanded", "false"),
  );
  shortcuts.addEventListener("click", (event) => {
    if (event.target !== shortcuts) return;
    const bounds = shortcuts.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    )
      shortcuts.close();
  });
  const setMapOnly = (enabled) => document.body.classList.toggle("map-only", enabled);
  addEventListener("keydown", (event) => {
    if (event.target.matches("input")) return;
    if (event.key === "?") {
      event.preventDefault();
      toggleShortcuts();
      return;
    }
    if (shortcuts.open) return;
    if (walkView && !["f", "F"].includes(event.key)) return;
    if (event.key === "Escape" && document.body.classList.contains("map-only")) {
      setMapOnly(false);
      return;
    }
    if (event.key === "Escape") {
      clearMeasurement();
      return;
    }
    if (event.key === "f" || event.key === "F") {
      setMapOnly(!document.body.classList.contains("map-only"));
      return;
    }
    if (event.key === "r" || event.key === "R") document.getElementById("randomize").click();
    if (event.key === "Home") {
      event.preventDefault();
      stopZoom();
      state.x = state.y = 0;
      state.zoom = initialZoom();
      render();
    }
    if (event.key === "+" || event.key === "=") zoomAt(1);
    if (event.key === "-") zoomAt(-1);
  });
  const readModelBundle = async () => {
    const response = await fetch("./assets/models/manifest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Model manifest returned HTTP ${response.status}`);
    const { bundle } = await response.json();
    if (!/^[0-9a-f]{16}$/.test(bundle)) throw new Error("Invalid model manifest");
    return bundle;
  };
  const checkModelBundle = async () => {
    if (bundleCheckPending) return;
    bundleCheckPending = true;
    try {
      if ((await readModelBundle()) !== modelBundle) TerrainHiGAN.notifyUpdate();
    } catch (error) {
      console.warn("Could not check model bundle:", error);
    } finally {
      bundleCheckPending = false;
    }
  };
  loadUrl();
  const exitWalk = (renderMap = true) => {
    if (!walkView) return;
    Object.assign(state, walkView.dispose());
    walkView = null;
    if (renderMap) render();
  };
  document.getElementById("toggle-walk").addEventListener("click", async () => {
    if (walkView) {
      exitWalk();
      return;
    }
    const button = document.getElementById("toggle-walk");
    button.disabled = true;
    try {
      for (const file of [
        "globe-renderer.js",
        "globe-walk.js",
        "globe-export.js",
        "walk-minimap.js",
        "planar-walk.js",
      ])
        await import(`./${file}?bundle=${modelBundle}`);
      controller.setView([], {
        seed: state.seed.toString(),
        map: state.colormap,
        zoom: state.zoom,
      });
      clearMeasurement();
      walkView = new TerrainHiGAN.globe.PlanarWalkView(state, exitWalk);
    } catch (error) {
      ui.status.textContent = error.message;
      console.error(error);
      render();
    } finally {
      button.disabled = false;
    }
  });
  addEventListener("resize", resize);
  (async () => {
    if ((await readModelBundle()) !== modelBundle) {
      TerrainHiGAN.notifyUpdate();
      return;
    }
    ui.bundle.textContent = modelBundle.slice(0, 8);
    ui.bundle.title = `Model bundle ${modelBundle}`;
    const workerOptions = new URLSearchParams(location.search);
    workerOptions.set("mobile", TerrainHiGAN.mobileView ? "1" : "0");
    workerOptions.set("workers", TerrainHiGAN.inferenceWorkerCount(workerOptions));
    workerOptions.set("bundle", modelBundle);
    modelWorker = new Worker(`./model-worker.js?${workerOptions}`);
    modelWorker.onerror = (event) => failModel(event.message || "Worker initialization failed");
    modelWorker.onmessage = onModelWorkerMessage;
    resize();
    setInterval(checkModelBundle, 15000);
    addEventListener("focus", checkModelBundle);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) checkModelBundle();
    });
  })().catch((error) => failModel(error.message || String(error)));
})();
