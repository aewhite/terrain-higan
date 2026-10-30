const { demoRequire } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: true,
      args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    }),
  );
  try {
    for (const view of ["", "globe.html"]) {
      const page = await browser.newPage({
        viewport: { width: 375, height: 812 },
        hasTouch: true,
        userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari",
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      await page.addInitScript(() => {
        globalThis.Worker = class {
          postMessage() {}
          terminate() {}
        };
        Element.prototype.requestFullscreen = undefined;
        Object.defineProperty(document, "fullscreenEnabled", { value: false });
        Object.defineProperty(document, "webkitFullscreenEnabled", { value: false });
      });
      await page.goto(baseUrl + view);
      await page.waitForFunction(() => document.body.classList.contains("mobile-ui"));
      assert(
        !(await page.locator("#toggle-fullscreen").isVisible()),
        "unsupported fullscreen icon stays hidden",
      );
      const manifest = await page.evaluate(async () => {
        const response = await fetch(document.querySelector("link[rel=manifest]").href);
        return { type: response.headers.get("Content-Type"), value: await response.json() };
      });
      assert.equal(manifest.type, "application/manifest+json");
      assert.equal(manifest.value.display, "standalone");
      assert.equal(manifest.value.scope, "./");
      assert.equal(
        await page.locator("meta[name=apple-mobile-web-app-capable]").getAttribute("content"),
        "yes",
      );
      assert.deepEqual(errors, []);
      await page.close();
    }
    for (const mode of ["webkit", "standalone"]) {
      const page = await browser.newPage();
      await page.addInitScript((mode) => {
        globalThis.Worker = class {
          postMessage() {}
        };
        Element.prototype.requestFullscreen = undefined;
        Object.defineProperty(document, "fullscreenEnabled", { value: undefined });
        if (mode === "webkit") {
          Object.defineProperty(document, "webkitFullscreenEnabled", { value: true });
          Object.defineProperty(document, "webkitFullscreenElement", {
            value: null,
            writable: true,
          });
          Element.prototype.webkitRequestFullscreen = function () {
            document.webkitFullscreenElement = this;
            document.dispatchEvent(new Event("webkitfullscreenchange"));
          };
          document.webkitExitFullscreen = () => {
            document.webkitFullscreenElement = null;
            document.dispatchEvent(new Event("webkitfullscreenchange"));
          };
        } else {
          Element.prototype.webkitRequestFullscreen = undefined;
          Object.defineProperty(document, "webkitFullscreenEnabled", { value: false });
          Object.defineProperty(navigator, "standalone", { value: true });
        }
      }, mode);
      await page.goto(baseUrl);
      await page.waitForFunction(() => TerrainHiGAN.setRenderProgress);
      if (mode === "webkit") {
        await page.locator("#toggle-fullscreen").click();
        assert.equal(await page.locator("#toggle-fullscreen").getAttribute("aria-pressed"), "true");
        await page.locator("#toggle-fullscreen").click();
        assert.equal(
          await page.locator("#toggle-fullscreen").getAttribute("aria-pressed"),
          "false",
        );
      } else
        assert(
          !(await page.locator("#toggle-fullscreen").isVisible()),
          "installed app does not offer an unavailable fullscreen toggle",
        );
      await page.close();
    }
    console.log(
      "Fullscreen: unavailable mobile API stays hidden in both views, valid standalone manifest, WebKit API entry/exit, installed-app state",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
