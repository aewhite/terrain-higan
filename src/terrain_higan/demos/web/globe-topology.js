"use strict";
globalThis.TerrainHiGAN ||= {};
(() => {
  const RADIUS = 6371000;
  const BASE_NSIDE = 16;
  const FACE_COUNT = 6;
  const MAX_GLOBE_LEVEL = 20;
  const MIN_ALTITUDE = 2;
  const MAX_ALTITUDE = RADIUS * 5;
  const GLOBE_VERSION = "s2-quadratic-v1";
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const sideAt = (level) => BASE_NSIDE * 2 ** level;
  const spacingAt = (level) => (RADIUS * Math.sqrt((4 * Math.PI) / FACE_COUNT)) / sideAt(level);
  const dot = (first, second) => first.reduce((sum, value, axis) => sum + value * second[axis], 0);
  const normalize = (vector) => {
    const length = Math.hypot(...vector);
    return vector.map((value) => value / length);
  };

  function faceVector(face, east, north) {
    const horizontal = stToUv(east),
      vertical = stToUv(north);
    const normal = 1 / Math.sqrt(1 + horizontal * horizontal + vertical * vertical);
    const eastComponent = horizontal * normal,
      northComponent = vertical * normal;
    switch (face) {
      case 0:
        return [normal, northComponent, -eastComponent];
      case 1:
        return [-eastComponent, northComponent, -normal];
      case 2:
        return [-eastComponent, normal, northComponent];
      case 3:
        return [-normal, -eastComponent, northComponent];
      case 4:
        return [northComponent, -eastComponent, normal];
      case 5:
        return [northComponent, -normal, -eastComponent];
    }
  }

  const stToUv = (value) =>
    value >= 0.5 ? (4 * value * value - 1) / 3 : (1 - 4 * (1 - value) * (1 - value)) / 3;
  const uvToSt = (value) =>
    value >= 0 ? 0.5 * Math.sqrt(1 + 3 * value) : 1 - 0.5 * Math.sqrt(1 - 3 * value);

  function vectorFace(vector) {
    const [first, third, negativeSecond] = vector;
    const second = -negativeSecond;
    const axis =
      Math.abs(first) >= Math.abs(second) && Math.abs(first) >= Math.abs(third)
        ? 0
        : Math.abs(second) >= Math.abs(third)
          ? 1
          : 2;
    const face = axis + ([first, second, third][axis] < 0 ? 3 : 0);
    let horizontal, vertical;
    switch (face) {
      case 0:
        horizontal = second / first;
        vertical = third / first;
        break;
      case 1:
        horizontal = -first / second;
        vertical = third / second;
        break;
      case 2:
        horizontal = -first / third;
        vertical = -second / third;
        break;
      case 3:
        horizontal = third / first;
        vertical = second / first;
        break;
      case 4:
        horizontal = third / second;
        vertical = -first / second;
        break;
      case 5:
        horizontal = -second / third;
        vertical = -first / third;
        break;
    }
    return { face, east: uvToSt(horizontal), north: uvToSt(vertical) };
  }

  const edgePoint = (edge, along) =>
    edge === 0 ? [0, along] : edge === 1 ? [1, along] : edge === 2 ? [along, 0] : [along, 1];
  const edges = Array.from({ length: FACE_COUNT }, (_, face) =>
    Array.from({ length: 4 }, (_, edge) => {
      const start = faceVector(face, ...edgePoint(edge, 0.25));
      const end = faceVector(face, ...edgePoint(edge, 0.75));
      for (let neighbor = 0; neighbor < FACE_COUNT; neighbor++) {
        if (neighbor === face) continue;
        for (let other = 0; other < 4; other++) {
          const first = faceVector(neighbor, ...edgePoint(other, 0.25));
          const last = faceVector(neighbor, ...edgePoint(other, 0.75));
          if (dot(start, first) > 1 - 1e-12 && dot(end, last) > 1 - 1e-12)
            return { face: neighbor, edge: other, reverse: false };
          if (dot(start, last) > 1 - 1e-12 && dot(end, first) > 1 - 1e-12)
            return { face: neighbor, edge: other, reverse: true };
        }
      }
      throw new Error(`Missing S2 neighbor ${face}:${edge}`);
    }),
  );

  function acrossEdge(face, east, north, edge, side) {
    const neighbor = edges[face][edge];
    const depth =
      edge === 0 ? -east - 1 : edge === 1 ? east - side : edge === 2 ? -north - 1 : north - side;
    let along = edge < 2 ? north : east;
    if (neighbor.reverse) along = side - 1 - along;
    return {
      face: neighbor.face,
      east: neighbor.edge === 0 ? depth : neighbor.edge === 1 ? side - 1 - depth : along,
      north: neighbor.edge === 2 ? depth : neighbor.edge === 3 ? side - 1 - depth : along,
    };
  }

  function canonicalPixel(face, east, north, side) {
    const minimum = -0.5,
      maximum = side - 0.5;
    while (east < minimum || east > maximum || north < minimum || north > maximum) {
      if ((east < minimum || east > maximum) && (north < minimum || north > maximum)) {
        const point = vectorFace(faceVector(face, (east + 0.5) / side, (north + 0.5) / side));
        return {
          face: point.face,
          east: Math.min(side - 1, Math.floor(point.east * side)),
          north: Math.min(side - 1, Math.floor(point.north * side)),
        };
      }
      const edge = east < minimum ? 0 : east > maximum ? 1 : north < minimum ? 2 : 3;
      ({ face, east, north } = acrossEdge(face, east, north, edge, side));
    }
    return { face, east, north };
  }

  function greatCircle(first, second) {
    const cross = [
      first[1] * second[2] - first[2] * second[1],
      first[2] * second[0] - first[0] * second[2],
      first[0] * second[1] - first[1] * second[0],
    ];
    return RADIUS * Math.atan2(Math.hypot(...cross), clamp(dot(first, second), -1, 1));
  }

  function horizonAngle(height) {
    const elevation = Math.max(0, height);
    return Math.atan2(Math.sqrt(elevation * (2 * RADIUS + elevation)), RADIUS);
  }

  function horizonDistance(observerHeight, targetHeight = 0) {
    return RADIUS * (horizonAngle(observerHeight) + horizonAngle(targetHeight));
  }

  function arcPoints(first, second, count = 96) {
    const angle = greatCircle(first, second) / RADIUS;
    let tangent = second.map((value, axis) => value - dot(first, second) * first[axis]);
    if (Math.hypot(...tangent) < 1e-10) {
      const reference = Math.abs(first[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      tangent = reference.map((value, axis) => value - dot(first, reference) * first[axis]);
    }
    tangent = normalize(tangent);
    return Array.from({ length: count + 1 }, (_, index) => {
      const phase = (angle * index) / count;
      return first.map((value, axis) => value * Math.cos(phase) + tangent[axis] * Math.sin(phase));
    });
  }

  Object.assign(TerrainHiGAN, {
    globe: {
      RADIUS,
      BASE_NSIDE,
      FACE_COUNT,
      MAX_GLOBE_LEVEL,
      MIN_ALTITUDE,
      MAX_ALTITUDE,
      GLOBE_VERSION,
      sideAt,
      spacingAt,
      dot,
      normalize,
      faceVector,
      vectorFace,
      edges,
      acrossEdge,
      canonicalPixel,
      greatCircle,
      horizonAngle,
      horizonDistance,
      arcPoints,
    },
  });
})();
