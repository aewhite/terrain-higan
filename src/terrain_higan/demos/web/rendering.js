"use strict";
globalThis.TerrainHiGAN ||= {};
(() => {
  const { clamp, elevationMeters, metersPerPixel } = TerrainHiGAN;
  const relief = {
    landGamma: 0.7,
    terrainStart: 0.25,
    landMaxElevation: 9000,
    hillshadeGamma: 0.85,
    ambientLight: 0.35,
    directionalLight: 0.65,
    azimuthDegrees: 315,
    altitudeDegrees: 45,
    realisticShadowStrength: 0.18,
    realisticShadowMinIntensity: 0.88,
    oceanReferenceDepth: 10000,
    coastColor: [0.68, 0.88, 1],
    deepOceanColor: [0, 0.1, 0.45],
  };
  // Samples from Matplotlib's terrain colormap. Relief only uses positions .25 through 1.
  const terrainStops = [
    [0, 0.2, 0.2, 0.6],
    [0.15, 0, 0.6, 1],
    [0.25, 0, 0.8, 0.4],
    [0.5, 1, 1, 0.6],
    [0.75, 0.5, 0.36, 0.33],
    [1, 1, 1, 1],
  ];
  const realisticOceanStops = [
    [-11000, 29, 56, 129],
    [-4000, 42, 81, 164],
    [-1000, 47, 93, 177],
    [-200, 51, 106, 187],
    [-50, 57, 126, 194],
    [-10, 69, 151, 202],
    [-0.001, 89, 172, 207],
  ];
  const realisticLandStops = [
    [0, 201, 189, 123],
    [100, 108, 152, 80],
    [700, 84, 117, 72],
    [1400, 149, 128, 100],
    [2000, 133, 133, 128],
    [3000, 158, 146, 127],
    [4000, 183, 173, 159],
    [4850, 211, 209, 204],
    [5000, 244, 246, 247],
    [9000, 244, 246, 247],
  ];

  function interpolate(low, high, fraction) {
    return low + (high - low) * fraction;
  }
  function contourInterval(zoom) {
    if (zoom <= 10) return 1000;
    if (zoom <= 14) return 250;
    if (zoom <= 18) return 50;
    return 10;
  }

  function singleScaleHillshade(elevation, width, height, sampleSpacingMeters) {
    const shade = new Float32Array(elevation.length);
    const azimuth = (relief.azimuthDegrees * Math.PI) / 180,
      altitude = (relief.altitudeDegrees * Math.PI) / 180;
    const lightX = Math.sin(azimuth) * Math.cos(altitude);
    // Canvas rows increase southward, so northward illumination has a negative Y component.
    const lightY = -Math.cos(azimuth) * Math.cos(altitude),
      lightZ = Math.sin(altitude);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        const left = elevation[y * width + Math.max(0, x - 1)],
          right = elevation[y * width + Math.min(width - 1, x + 1)];
        const above = elevation[Math.max(0, y - 1) * width + x],
          below = elevation[Math.min(height - 1, y + 1) * width + x];
        const gradientX = (right - left) / (2 * sampleSpacingMeters),
          gradientY = (below - above) / (2 * sampleSpacingMeters);
        const illumination =
          (-gradientX * lightX - gradientY * lightY + lightZ) / Math.hypot(gradientX, gradientY, 1);
        shade[index] = clamp((illumination + 1) * 0.5, 0, 1);
      }
    return shade;
  }

  function computeHillshade(landElevation, width, height, sampleSpacingMeters) {
    const shade = singleScaleHillshade(landElevation, width, height, sampleSpacingMeters);
    for (let index = 0; index < shade.length; index++) shade[index] **= relief.hillshadeGamma;
    return shade;
  }

  function terrainColor(position, pixels, index) {
    let right = terrainStops.length - 1;
    for (let stop = 1; stop < terrainStops.length; stop++)
      if (position <= terrainStops[stop][0]) {
        right = stop;
        break;
      }
    const low = terrainStops[right - 1],
      high = terrainStops[right];
    const fraction = clamp((position - low[0]) / (high[0] - low[0]), 0, 1);
    pixels[index] = interpolate(low[1], high[1], fraction) * 255;
    pixels[index + 1] = interpolate(low[2], high[2], fraction) * 255;
    pixels[index + 2] = interpolate(low[3], high[3], fraction) * 255;
  }

  function oceanColor(elevation, pixels, index) {
    const fraction = clamp(-elevation / relief.oceanReferenceDepth, 0, 1) ** relief.landGamma;
    pixels[index] = interpolate(relief.coastColor[0], relief.deepOceanColor[0], fraction) * 255;
    pixels[index + 1] = interpolate(relief.coastColor[1], relief.deepOceanColor[1], fraction) * 255;
    pixels[index + 2] = interpolate(relief.coastColor[2], relief.deepOceanColor[2], fraction) * 255;
  }

  function renderElevationRelief(values, width, height, padding, sampleSpacingMeters) {
    const landElevation = new Float32Array(values.length);
    for (let index = 0; index < values.length; index++) {
      const elevation = elevationMeters(values[index]);
      landElevation[index] = Number.isFinite(elevation) ? Math.max(0, elevation) : 0;
    }
    const hillshade = computeHillshade(landElevation, width, height, sampleSpacingMeters);
    const outputWidth = width - padding * 2,
      outputHeight = height - padding * 2;
    const pixels = new Uint8ClampedArray(outputWidth * outputHeight * 4);
    for (let y = padding; y < height - padding; y++)
      for (let x = padding; x < width - padding; x++) {
        const sourceIndex = y * width + x,
          pixelIndex = ((y - padding) * outputWidth + x - padding) * 4;
        const elevation = elevationMeters(values[sourceIndex]);
        if (!Number.isFinite(elevation)) {
          pixels[pixelIndex + 3] = 255;
          continue;
        }
        if (elevation < 0) oceanColor(elevation, pixels, pixelIndex);
        else {
          const position =
            relief.terrainStart +
            clamp((Math.max(0, elevation) / relief.landMaxElevation) ** relief.landGamma, 0, 1) *
              (1 - relief.terrainStart);
          terrainColor(position, pixels, pixelIndex);
          const intensity = relief.ambientLight + relief.directionalLight * hillshade[sourceIndex];
          pixels[pixelIndex] *= intensity;
          pixels[pixelIndex + 1] *= intensity;
          pixels[pixelIndex + 2] *= intensity;
        }
        pixels[pixelIndex + 3] = 255;
      }
    return pixels;
  }

  function writeRealisticColor(elevation, pixels, index) {
    const stops = elevation < 0 ? realisticOceanStops : realisticLandStops;
    let right = stops.findIndex(([height]) => elevation <= height);
    if (right < 0) right = stops.length - 1;
    if (right === 0) {
      pixels[index] = stops[0][1];
      pixels[index + 1] = stops[0][2];
      pixels[index + 2] = stops[0][3];
      return;
    }
    const low = stops[right - 1],
      high = stops[right],
      fraction = clamp((elevation - low[0]) / (high[0] - low[0]), 0, 1);
    pixels[index] = interpolate(low[1], high[1], fraction) | 0;
    pixels[index + 1] = interpolate(low[2], high[2], fraction) | 0;
    pixels[index + 2] = interpolate(low[3], high[3], fraction) | 0;
  }

  function renderRealisticRelief(values, width, height, padding, sampleSpacingMeters) {
    const landElevation = new Float32Array(values.length);
    for (let index = 0; index < values.length; index++) {
      const elevation = elevationMeters(values[index]);
      landElevation[index] = Number.isFinite(elevation) ? Math.max(0, elevation) : 0;
    }
    const shadow = singleScaleHillshade(landElevation, width, height, sampleSpacingMeters);
    const flatHillshade = (Math.sin((relief.altitudeDegrees * Math.PI) / 180) + 1) * 0.5;
    const outputWidth = width - padding * 2,
      outputHeight = height - padding * 2;
    const pixels = new Uint8ClampedArray(outputWidth * outputHeight * 4);
    for (let y = padding; y < height - padding; y++)
      for (let x = padding; x < width - padding; x++) {
        const sourceIndex = y * width + x,
          pixelIndex = ((y - padding) * outputWidth + x - padding) * 4;
        const elevation = elevationMeters(values[sourceIndex]);
        if (Number.isFinite(elevation)) {
          writeRealisticColor(elevation, pixels, pixelIndex);
          if (elevation >= 0) {
            const intensity = Math.max(
              relief.realisticShadowMinIntensity,
              1 + relief.realisticShadowStrength * Math.min(shadow[sourceIndex] - flatHillshade, 0),
            );
            pixels[pixelIndex] *= intensity;
            pixels[pixelIndex + 1] *= intensity;
            pixels[pixelIndex + 2] *= intensity;
          }
        }
        pixels[pixelIndex + 3] = 255;
      }
    return pixels;
  }

  function contourBaseColor(elevation) {
    if (elevation < 0) return [202, 226, 232];
    const shade = clamp(elevation / 9000, 0, 1);
    return [(239 - shade * 20) | 0, (232 - shade * 18) | 0, (211 - shade * 14) | 0];
  }

  function renderContours(values, width, height, padding, zoom) {
    const interval = contourInterval(zoom),
      majorInterval = interval * 5;
    const outputWidth = width - padding * 2,
      outputHeight = height - padding * 2;
    const pixels = new Uint8ClampedArray(outputWidth * outputHeight * 4);
    const contourLevel = (elevation, step) => Math.floor(elevation / step);
    for (let y = padding; y < height - padding; y++)
      for (let x = padding; x < width - padding; x++) {
        const index = y * width + x,
          elevation = elevationMeters(values[index]);
        const right = elevationMeters(values[index + 1]),
          below = elevationMeters(values[index + width]);
        const crosses = (step) =>
          contourLevel(elevation, step) !== contourLevel(right, step) ||
          contourLevel(elevation, step) !== contourLevel(below, step);
        const rgb = crosses(majorInterval)
          ? [113, 78, 45]
          : crosses(interval)
            ? elevation < 0
              ? [47, 100, 128]
              : [165, 126, 84]
            : contourBaseColor(elevation);
        const pixel = ((y - padding) * outputWidth + x - padding) * 4;
        pixels[pixel] = rgb[0];
        pixels[pixel + 1] = rgb[1];
        pixels[pixel + 2] = rgb[2];
        pixels[pixel + 3] = 255;
      }
    return pixels;
  }

  function renderRegion(region, map, padding) {
    const { values, width, height, zoom } = region;
    if (map === "contour") return renderContours(values, width, height, padding, zoom);
    const sampleSpacingMeters = metersPerPixel(zoom);
    if (map === "elevation")
      return renderElevationRelief(values, width, height, padding, sampleSpacingMeters);
    if (map === "realistic")
      return renderRealisticRelief(values, width, height, padding, sampleSpacingMeters);
    throw new RangeError("Unknown colormap: " + map);
  }
  function renderingPadding() {
    return 1;
  }

  Object.assign(TerrainHiGAN, {
    renderRegion,
    renderingPadding,
    writeRealisticColor,
    renderRealisticRelief,
    renderElevationRelief,
    computeHillshade,
    contourInterval,
    contourBaseColor,
    renderContours,
    terrainColor,
    oceanColor,
  });
})();
