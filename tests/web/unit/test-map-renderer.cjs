const assert = require("node:assert/strict");
const { loadModules } = require("../support/test-support.cjs");

(() => {
  let uploads = 0,
    copies = 0,
    created = 0;
  const makeCanvas = () => {
    created++;
    const context = {
      fillRect() {},
      drawImage(...args) {
        copies++;
        this.lastDraw = args;
      },
      putImageData() {
        uploads++;
      },
    };
    return { width: 256, height: 256, dataset: {}, getContext: () => context };
  };
  const { api, load } = loadModules({
    document: { createElement: makeCanvas },
    ImageData: class {
      constructor(data, width, height) {
        Object.assign(this, { data, width, height });
      }
    },
  });
  load("map-renderer.js");
  const sent = [];
  const controller = new api.TileController({
    send: (request) => sent.push(request),
    onTile: (_data, tile) => tile,
    onStatus() {},
    onError: (message) => assert.fail(message),
  });
  const identity = { seed: "1", map: "realistic", zoom: 0 };
  controller.setView([{ key: "first" }, { key: "outdated" }], identity);
  controller.receive({ ...sent[0], type: "tile", provider: { label: "test" } }, false);
  assert.equal(sent.length, 1, "tile replies leave dispatch to the next display frame");
  controller.setView([{ key: "latest" }], identity);
  assert.equal(
    sent.at(-1).key,
    "latest",
    "navigation selects current work before submitting another tile",
  );
  const canvas = makeCanvas();
  const renderer = new api.MapRenderer(canvas);
  const view = { seed: "1", map: "realistic", zoom: 0, level: 0, left: 0, top: 0, size: 128 };
  const tile = (x, y, zoom = 0) => ({
    key: `${zoom}:${x}:${y}`,
    seed: "1",
    map: "realistic",
    zoom,
    x,
    y,
    pixels: new Uint8Array(128 * 128 * 4).buffer,
  });
  const tiles = new Map();
  for (let y = 0; y < 256; y += 128)
    for (let x = 0; x < 256; x += 128) {
      const value = tile(x, y);
      tiles.set(value.key, value);
    }
  let frames = 0;
  const update = (camera = view, revision = 1) => {
    const before = uploads,
      painted = copies;
    renderer.update(camera, tiles, revision, 1.5);
    assert(uploads - before <= 1, "tile image preparation is bounded per frame");
    assert(copies - painted <= 1, "tile composites are bounded per frame");
    assert(++frames < 1000, "replacement work finishes");
  };
  while (!renderer.ready(view, 1)) update();
  assert.equal(uploads, tiles.size);
  const original = renderer.front;
  const builds = renderer.builds;
  const moved = { ...view, left: 12.25, top: -7.75 };
  const prepared = uploads,
    allocated = created;
  renderer.draw(moved);
  renderer.draw({ ...moved, zoom: 0.5, level: 1, left: moved.left * 2, top: moved.top * 2 });
  assert.equal(uploads, prepared, "input only transforms the committed image");
  assert.equal(created, allocated, "input never allocates tile images or replacement canvases");
  assert.equal(renderer.front, original);
  assert(
    !renderer.intersects(renderer.front, { x: 1, y: 1, width: 0, height: 0 }),
    "zooming out skips cached tiles that collapse below one native pixel",
  );
  update(moved);
  assert.equal(renderer.builds, builds, "small pans reuse the committed mosaic");
  const added = tile(256, 0);
  tiles.set(added.key, added);
  update(view, 2);
  assert.equal(renderer.front, original, "new native tiles patch the existing mosaic");
  assert.equal(renderer.builds, builds, "tile arrival does not redraw every cached tile");
  assert(added.image && !added.pixels, "raw pixels are released after deferred image preparation");

  const zoomed = { ...view, zoom: 1, level: 1 };
  const complete = renderer.front;
  update(zoomed, 2);
  assert.equal(renderer.front, complete, "the old mosaic stays installed during a zoom rebuild");
  renderer.draw(zoomed);
  assert.equal(canvas.getContext().lastDraw[0], complete.surface);
  const latest = tile(0, 0, 1);
  tiles.set(latest.key, latest);
  while (!renderer.ready(zoomed, 3)) update(zoomed, 3);
  assert.notEqual(renderer.front, complete);
  assert(
    renderer.front.applied.has(latest.key),
    "tiles arriving during preparation are eventually displayed",
  );
  const surfaces = new Set([renderer.front.surface, renderer.spare]);
  update({ ...zoomed, left: 300 }, 3);
  assert(renderer.job, "a replacement is still being prepared");
  const installed = renderer.front;
  while (!renderer.ready(zoomed, 3)) update(zoomed, 3);
  assert.equal(
    renderer.front,
    installed,
    "returning to covered terrain cancels obsolete replacement work",
  );
  assert(surfaces.has(renderer.spare), "cancelled jobs reuse the spare canvas");
  tiles.clear();
  const retained = renderer.front;
  const beyond = { ...zoomed, left: 300 };
  while (!renderer.ready(beyond, 4)) update(beyond, 4);
  assert.equal(
    renderer.front.context.lastDraw[0],
    retained.surface,
    "replacement canvases preserve visible pixels after source-cache eviction",
  );
  const staleCopies = copies;
  renderer.draw({ ...view, seed: "2" });
  assert.equal(copies, staleCopies, "a changed seed never displays old terrain");
  console.log(
    "2D retained-image transforms, bounded preparation, incremental tiles, replacement/cancellation and seam-safe mosaics passed",
  );
})();
