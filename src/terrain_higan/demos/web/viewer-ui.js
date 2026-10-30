"use strict";
(() => {
  const progress = document.getElementById("render-progress-value");
  const progressLabel = document.getElementById("render-progress-label");
  const progressPanel = document.getElementById("render-progress");
  const updateButton = document.getElementById("reload-viewer");
  let loadedTiles = 0,
    totalTiles = 0,
    renderFailed = false,
    renderError = "",
    phase = "Starting",
    updateAvailable = false;
  TerrainHiGAN.scalePixels = document.querySelector(".scale i").getBoundingClientRect().width;
  const drawProgress = () => {
    const ratio = totalTiles > 0 ? Math.min(1, loadedTiles / totalTiles) : 0;
    const ready = totalTiles > 0 && ratio === 1 && !renderFailed;
    progressPanel.hidden = ready && !updateAvailable;
    progressPanel.title = renderError;
    updateButton.hidden = !updateAvailable && !renderFailed;
    updateButton.textContent = updateAvailable ? "Reload" : "Retry";
    if (loadedTiles > 0 || renderFailed) progress.value = ratio;
    else progress.removeAttribute("value");
    progressLabel.textContent = updateAvailable
      ? "Update available"
      : renderFailed
        ? /out of memory|allocation|bad_alloc/i.test(renderError)
          ? "Out of memory"
          : "Render failed"
        : ready
          ? "Ready"
          : (phase || "Generating") + (loadedTiles > 0 ? ` · ${Math.round(ratio * 100)}%` : "");
    progressPanel.dataset.state = updateAvailable
      ? "update"
      : renderFailed
        ? "failed"
        : ready
          ? "ready"
          : "rendering";
    if (ready) phase = "";
  };
  TerrainHiGAN.setRenderProgress = (loaded, total, failed = false) => {
    loadedTiles = loaded;
    totalTiles = total;
    renderFailed = failed;
    if (!failed) renderError = "";
    drawProgress();
  };
  TerrainHiGAN.setRenderError = (message) => {
    renderError = message;
    renderFailed = true;
    drawProgress();
  };
  TerrainHiGAN.setRenderPhase = (message) => {
    phase = message;
    drawProgress();
  };
  TerrainHiGAN.notifyUpdate = () => {
    updateAvailable = true;
    updateButton.hidden = false;
    drawProgress();
  };
  updateButton.addEventListener("click", () => location.reload());
  const mobile = TerrainHiGAN.isMobileDevice() || matchMedia("(pointer: coarse)").matches;
  document.body.classList.toggle("mobile-ui", mobile);
  TerrainHiGAN.mobileView = mobile;
  const toggle = document.getElementById("toggle-controls");
  toggle.addEventListener("click", () => {
    const expanded = document.body.classList.toggle("controls-expanded");
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.setAttribute("aria-label", expanded ? "Hide controls" : "Show controls");
    toggle.textContent = expanded ? "×" : "☰";
  });
  const fullscreen = document.getElementById("toggle-fullscreen");
  const element = document.documentElement;
  const activeFullscreen = () => document.fullscreenElement || document.webkitFullscreenElement;
  const fullscreenRequest = () =>
    element.requestFullscreen && document.fullscreenEnabled !== false
      ? element.requestFullscreen
      : document.webkitFullscreenEnabled !== false
        ? element.webkitRequestFullscreen
        : null;
  const canFullscreen = !!fullscreenRequest();
  fullscreen.hidden = !canFullscreen;
  const syncFullscreen = () => {
    const active = !!activeFullscreen();
    fullscreen.setAttribute("aria-pressed", String(active));
    const label = active ? "Exit fullscreen" : "Enter fullscreen";
    fullscreen.setAttribute("aria-label", label);
    fullscreen.title = label;
  };
  fullscreen.addEventListener("click", async () => {
    try {
      if (activeFullscreen()) {
        const exit = document.fullscreenElement
          ? document.exitFullscreen
          : document.webkitExitFullscreen;
        await exit.call(document);
      } else if (fullscreenRequest()) await fullscreenRequest().call(element);
      else fullscreen.hidden = true;
    } catch {
      fullscreen.hidden = true;
    }
    syncFullscreen();
  });
  document.addEventListener("fullscreenchange", syncFullscreen);
  document.addEventListener("webkitfullscreenchange", syncFullscreen);
  syncFullscreen();
})();
