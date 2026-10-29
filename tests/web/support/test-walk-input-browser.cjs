const assert = require("node:assert/strict");

module.exports = async function checkWalkingInput(page, selector) {
  await page.evaluate(() => {
    const original = TerrainHiGAN.globe.GlobeWalk.prototype.tick;
    TerrainHiGAN.globe.GlobeWalk.prototype.tick = function (...args) {
      window.walkInstance = this;
      return original.apply(this, args);
    };
  });
  const canvas = page.locator(selector);
  const { width, height } = page.viewportSize();
  const stopped = () =>
    page.waitForFunction(
      (selector) => Number(document.querySelector(selector).dataset.walkSpeed) === 0,
      selector,
    );
  const moving = () =>
    page.waitForFunction(
      (selector) => Number(document.querySelector(selector).dataset.walkSpeed) > 0,
      selector,
    );
  const pose = () =>
    page.evaluate(() => [
      ...walkInstance.renderer.camera.position.toArray(),
      ...walkInstance.heading.toArray(),
    ]);
  await page.keyboard.press("f");
  await page.mouse.move(width / 2, height / 2);
  await page.keyboard.down("w");
  await moving();
  await page.keyboard.up("w");
  await stopped();
  const resting = await pose();
  await page.waitForTimeout(150);
  assert.deepEqual(await pose(), resting, "releasing keys stops walking without a pause command");
  assert.equal(
    await page.evaluate(() => walkInstance.frame),
    0,
    "idle input has no animation loop",
  );
  await canvas.click({ position: { x: width / 2, y: height / 2 } });
  await page.waitForFunction(
    (selector) => document.querySelector(selector).dataset.walkMouseBound === "false",
    selector,
  );
  const free = await pose();
  await page.mouse.move(width * 0.7, height * 0.4);
  await page.waitForTimeout(150);
  assert.deepEqual(await pose(), free, "unbound mouse cannot steer the camera");
  for (const key of ["ArrowUp", "ArrowRight", "s", "a"]) {
    const before = await pose();
    await page.keyboard.down(key);
    await moving();
    await page.waitForFunction(
      (position) =>
        walkInstance.renderer.camera.position.distanceTo({
          x: position[0],
          y: position[1],
          z: position[2],
        }) *
          TerrainHiGAN.globe.RADIUS >
        0.1,
      before,
    );
    await page.keyboard.up(key);
    await stopped();
    const after = await pose();
    assert.notDeepEqual(
      after.slice(0, 3),
      before.slice(0, 3),
      `${key} moves with the mouse unbound`,
    );
  }
  await canvas.click({ position: { x: width / 2, y: height / 2 } });
  await page.waitForFunction(
    (selector) => document.querySelector(selector).dataset.walkMouseBound === "true",
    selector,
  );
  const bound = await pose();
  await page.waitForTimeout(100);
  assert.deepEqual(await pose(), bound, "rebinding starts neutral without a camera jump");
  await page.mouse.move(width * 0.7, height * 0.4);
  await page.waitForFunction(
    (heading) =>
      Math.hypot(
        ...walkInstance.heading.toArray().map((value, index) => value - heading[index + 3]),
      ) > 0.05,
    bound,
  );
  await page.keyboard.press("Escape");
  await page.waitForFunction(
    (selector) => document.querySelector(selector).dataset.walkPaused === "true",
    selector,
  );
  await page.mouse.move(width / 2, height / 2);
  await page.keyboard.down("w");
  await moving();
  await page.keyboard.press("Escape");
  await page.keyboard.down("w");
  await page.waitForTimeout(150);
  assert.equal(await page.evaluate(() => walkInstance.paused), true);
  await page.waitForFunction(
    (selector) => document.querySelector(selector).dataset.walkPaused === "true",
    selector,
  );
  assert.equal(
    await canvas.getAttribute("data-walk-paused"),
    "true",
    "held-key repeats cannot undo Escape pause",
  );
  assert.equal(await page.evaluate(() => walkInstance.keys.size), 0);
  await page.keyboard.up("w");
  await page.keyboard.down("ArrowDown");
  await moving();
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await page.keyboard.up("ArrowDown");
  await stopped();
  assert.equal(await page.evaluate(() => walkInstance.keys.size), 0, "blur clears held movement");
  await page.keyboard.press("f");
  await page.locator("#walk-speed").focus();
  const editing = await pose();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("w");
  await page.waitForTimeout(100);
  assert.deepEqual(await pose(), editing, "form edits do not move the camera");
  assert.equal(await page.evaluate(() => walkInstance.keys.size), 0);
  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press("f");
};
