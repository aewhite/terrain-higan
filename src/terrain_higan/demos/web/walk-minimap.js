import * as THREE from "three";

const { RADIUS, WALK_RADIUS } = TerrainHiGAN.globe;

class WalkMinimap {
  constructor(walk) {
    this.walk = walk;
    this.element = document.createElement("aside");
    this.element.className = "walk-minimap panel";
    this.element.hidden = true;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = 224;
    this.canvas.setAttribute(
      "aria-label",
      "North-up terrain minimap with position and field of view",
    );
    this.caption = document.createElement("div");
    this.element.append(this.canvas, this.caption);
    document.body.append(this.element);
  }

  build() {
    const walk = this.walk;
    walk.minimapFrame = walk.mapFrame();
    this.background = document.createElement("canvas");
    this.background.width = this.background.height = 224;
    const context = this.background.getContext("2d");
    const image = context.createImageData(224, 224);
    for (let row = 0; row < 224; row++)
      for (let column = 0; column < 224; column++) {
        const east = ((column + 0.5 - 112) / 104) * WALK_RADIUS;
        const south = ((row + 0.5 - 112) / 104) * WALK_RADIUS;
        if (Math.hypot(east, south) > WALK_RADIUS) continue;
        const height = walk.mapHeight(walk.mapPosition(east, south));
        if (height === null) continue;
        const offset = (row * 224 + column) * 4;
        TerrainHiGAN.writeRealisticColor(height, image.data, offset);
        image.data[offset + 3] = 255;
      }
    context.putImageData(image, 0, 0);
    this.selection = walk.selection;
    this.caption.textContent = "20 km walking radius · N ↑";
  }

  draw() {
    const walk = this.walk;
    this.element.hidden = !walk.active;
    if (!walk.active) return;
    const context = this.canvas.getContext("2d");
    context.clearRect(0, 0, 224, 224);
    if (!walk.prepared) {
      this.selection = null;
      this.caption.textContent = "Preparing 20 km map…";
      return;
    }
    if (this.selection !== walk.selection) this.build();
    context.drawImage(this.background, 0, 0);
    const { origin, east, north } = walk.minimapFrame;
    const position = walk.renderer.camera.position.clone().sub(origin);
    const scale = (104 * RADIUS) / WALK_RADIUS;
    const playerX = 112 + position.dot(east) * scale;
    const playerY = 112 - position.dot(north) * scale;
    const camera = walk.renderer.camera;
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    const heading = Math.atan2(-direction.dot(north), direction.dot(east));
    const fov = 2 * Math.atan(Math.tan((camera.fov * Math.PI) / 360) * camera.aspect);
    context.save();
    context.beginPath();
    context.arc(112, 112, 104, 0, Math.PI * 2);
    context.clip();
    context.beginPath();
    context.moveTo(playerX, playerY);
    context.arc(playerX, playerY, 224, heading - fov / 2, heading + fov / 2);
    context.closePath();
    context.fillStyle = "rgba(255, 224, 122, 0.24)";
    context.strokeStyle = "#ffe07a";
    context.lineWidth = 1.5;
    context.fill();
    context.stroke();
    context.beginPath();
    context.arc(playerX, playerY, 4, 0, Math.PI * 2);
    context.fillStyle = "#fff";
    context.fill();
    context.strokeStyle = "#142b30";
    context.stroke();
    context.restore();
    context.beginPath();
    context.arc(112, 112, 104, 0, Math.PI * 2);
    context.strokeStyle = "#91d8ba";
    context.stroke();
    Object.assign(this.canvas.dataset, { heading, fov, playerX, playerY });
  }

  dispose() {
    this.element.remove();
  }
}

TerrainHiGAN.globe.WalkMinimap = WalkMinimap;
