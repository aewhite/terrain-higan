"use strict";
(() => {
  // Observe only active rendering/interaction, never an idle gap between frames.
  // Delays are bounded: refinement continues even during a sustained gesture.
  class RefinementScheduler {
    constructor(canvas) {
      this.canvas = canvas;
      this.delayMs = 0;
      this.interactionDelayMs =
        TerrainHiGAN.isMobileDevice() || globalThis.matchMedia?.("(pointer: coarse)").matches
          ? 4
          : 0;
      this.meshBudgetMs = 4;
      this.interactingUntil = 0;
      this.busy = false;
      this.frame = 0;
      this.lastFrame = null;
      this.frameIntervalMs = Infinity;
      this.lastSent = -Infinity;
      const interact = (event) => {
        if (event.type === "pointermove" && !event.buttons && event.pointerType !== "touch") return;
        const now = performance.now();
        const starting = now >= this.interactingUntil;
        this.interactingUntil = now + 150;
        this.meshBudgetMs = 1.5;
        this.delayMs = Math.max(this.interactionDelayMs, this.delayMs);
        if (starting || now - this.lastSent >= 100) this.send();
        this.schedule();
      };
      for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel", "wheel"])
        canvas.addEventListener(type, interact, { passive: true, capture: true });
    }

    setWorker(worker) {
      this.worker = worker;
      this.send();
    }

    setBusy(busy) {
      this.busy = busy;
      if (busy) this.schedule();
    }

    send() {
      this.worker?.postMessage({ type: "frame-budget", delayMs: this.delayMs });
      this.lastSent = performance.now();
      this.canvas.dataset.inferenceDelayMs = this.delayMs;
      this.canvas.dataset.meshBudgetMs = this.meshBudgetMs;
    }

    observe(time) {
      const interval = this.lastFrame === null ? 0 : time - this.lastFrame;
      const interacting = time < this.interactingUntil;
      // Compare with the observed display cadence (including 30/120 Hz phones).
      // Idle gaps never enter this calculation; recover gradually after a hitch.
      if (interval > 0) {
        this.frameIntervalMs = Math.min(this.frameIntervalMs, interval);
        const slow = interval > this.frameIntervalMs * 1.5;
        this.delayMs = Math.max(0, Math.min(32, this.delayMs + (slow ? 4 : -1)));
      }
      if (interacting) this.delayMs = Math.max(this.interactionDelayMs, this.delayMs);
      this.meshBudgetMs = interacting || this.delayMs > 0 ? 1.5 : 4;
      this.lastFrame = time;
      if (!this.busy && !interacting) {
        this.delayMs = 0;
        this.meshBudgetMs = 4;
        this.lastFrame = null;
        this.frameIntervalMs = Infinity;
        this.send();
      } else {
        if (time - this.lastSent >= 100) this.send();
        this.schedule();
      }
    }

    schedule() {
      if (this.frame) return;
      this.frame = requestAnimationFrame((time) => {
        this.frame = 0;
        this.observe(time);
      });
    }
  }
  TerrainHiGAN.RefinementScheduler = RefinementScheduler;
})();
