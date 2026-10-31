"use strict";
(() => {
  const { faceVector, vectorFace, canonicalPixel, acrossEdge, edges, sideAt, MAX_GLOBE_LEVEL } =
    TerrainHiGAN.globe;
  const meshSide = sideAt(MAX_GLOBE_LEVEL);
  const edgePoint = (patch, edge, along) => {
    const side = 2 ** patch.depth;
    return [
      (patch.tileX + (edge === 0 ? 0 : edge === 1 ? 1 : along)) / side,
      (patch.tileY + (edge === 2 ? 0 : edge === 3 ? 1 : along)) / side,
    ];
  };
  const cornerKey = (face, east, north) => {
    const point = vectorFace(faceVector(face, east, north));
    return `${point.face}:${Math.round(point.east * meshSide)}:${Math.round(point.north * meshSide)}`;
  };
  const before = (first, second) =>
    first.depth < second.depth || (first.depth === second.depth && first.key < second.key);

  function edgeValue(binding, along, result = new Float64Array(6)) {
    const parameter = binding.start + ((binding.end - binding.start) * along) / 64;
    const { parameters, vertices } = binding.line;
    let low = 0,
      high = parameters.length - 1;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (parameters[middle] <= parameter) low = middle;
      else high = middle;
    }
    const fraction = (parameter - parameters[low]) / (parameters[high] - parameters[low]);
    for (let axis = 0; axis < 6; axis++)
      result[axis] =
        vertices[low * 6 + axis] * (1 - fraction) + vertices[high * 6 + axis] * fraction;
    return result;
  }

  function* stitchPatchSteps(patches, previous = {}, vertex, topology = {}) {
    const pixel = topology.canonicalPixel || canonicalPixel;
    const corner = topology.cornerKey || cornerKey;
    const lookup = new Map(),
      prefixes = new Set(),
      records = [],
      parents = [];
    for (const [index, patch] of patches.entries()) {
      lookup.set(patch.key, index);
      for (let depth = 0; depth < patch.depth; depth++) {
        const scale = 2 ** (patch.depth - depth);
        prefixes.add(
          `${patch.face}:${depth}:${Math.floor(patch.tileX / scale)}:${Math.floor(patch.tileY / scale)}`,
        );
      }
      for (let edge = 0; edge < 4; edge++) {
        parents.push(records.length);
        records.push({ patch, edge });
      }
      yield;
    }
    const root = (index) => {
      while (parents[index] !== index) {
        parents[index] = parents[parents[index]];
        index = parents[index];
      }
      return index;
    };
    const join = (first, second) => {
      parents[root(first)] = root(second);
    };
    for (let index = 0; index < records.length; index++) {
      const { patch, edge } = records[index];
      const neighbor = pixel(
        patch.face,
        patch.tileX + (edge === 0 ? -1 : edge === 1 ? 1 : 0),
        patch.tileY + (edge === 2 ? -1 : edge === 3 ? 1 : 0),
        2 ** patch.depth,
      );
      const opposite = neighbor.face === patch.face ? edge ^ 1 : edges[patch.face][edge].edge;
      let covered = false;
      for (let depth = patch.depth; depth >= 0; depth--) {
        const scale = 2 ** (patch.depth - depth);
        const found = lookup.get(
          `${neighbor.face}:${depth}:${Math.floor(neighbor.east / scale)}:${Math.floor(neighbor.north / scale)}`,
        );
        if (found === undefined) continue;
        join(index, found * 4 + opposite);
        covered = true;
        break;
      }
      if (covered) {
        yield;
        continue;
      }
      const visit = (depth, east, north) => {
        const key = `${neighbor.face}:${depth}:${east}:${north}`;
        const found = lookup.get(key);
        if (found !== undefined) {
          join(index, found * 4 + opposite);
          return;
        }
        if (!prefixes.has(key)) return;
        for (let along = 0; along < 2; along++)
          visit(
            depth + 1,
            east * 2 + (opposite === 0 ? 0 : opposite === 1 ? 1 : along),
            north * 2 + (opposite === 2 ? 0 : opposite === 3 ? 1 : along),
          );
      };
      visit(patch.depth, neighbor.east, neighbor.north);
      yield;
    }
    const groups = new Map(),
      corners = new Map();
    for (const [index, record] of records.entries()) {
      const group = root(index);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(record);
      for (const along of [0, 1]) {
        const [east, north] = edgePoint(record.patch, record.edge, along);
        const key = corner(record.patch.face, east, north);
        const source = record.patch.source;
        const existing = corners.get(key);
        if (existing && !before(source, existing.source)) continue;
        corners.set(key, { source, east, north });
      }
      yield;
    }
    for (const members of groups.values()) {
      const source = members.reduce((best, member) =>
        before(member.patch.source, best.patch.source) ? member : best,
      ).patch.source;
      for (const { patch, edge } of members)
        for (const along of [0, 1]) {
          let [east, north] = edgePoint(patch, edge, along);
          const key = corner(patch.face, east, north);
          if (!before(source, corners.get(key).source)) continue;
          if (patch.face !== source.face) {
            const point = acrossEdge(patch.face, east - 0.5, north - 0.5, edge, 1);
            east = point.east + 0.5;
            north = point.north + 0.5;
          }
          corners.set(key, { source, east, north });
        }
      yield;
    }
    for (const [key, corner] of corners) {
      const { source, east, north } = corner;
      const cached = previous.corners?.get(key);
      corner.value =
        cached?.source === source
          ? cached.value
          : vertex(
              source,
              (east * 2 ** source.depth - source.tileX) * 64,
              (north * 2 ** source.depth - source.tileY) * 64,
            );
      yield;
    }
    const cornerCache = new Map(corners),
      edgeCache = new Map();
    const components = [...groups.values()].sort(
      (first, second) =>
        Math.min(...first.map(({ patch }) => patch.depth + Math.log2(patch.segments))) -
        Math.min(...second.map(({ patch }) => patch.depth + Math.log2(patch.segments))),
    );
    const bindings = new Map(patches.map((patch) => [patch.key, []]));
    for (const members of components) {
      const owner = members.reduce((best, member) =>
        before(member.patch, best.patch) ? member : best,
      );
      const field = members.reduce((best, member) =>
        before(member.patch.source, best.patch.source) ? member : best,
      ).patch.source;
      const intervals = members.map((member) => {
        const parameters = [0, 1].map((along) => {
          let [east, north] = edgePoint(member.patch, member.edge, along);
          if (member.patch.face !== owner.patch.face) {
            const point = acrossEdge(member.patch.face, east - 0.5, north - 0.5, member.edge, 1);
            east = point.east + 0.5;
            north = point.north + 0.5;
          }
          return (
            (owner.edge < 2 ? north : east) * 2 ** owner.patch.depth -
            (owner.edge < 2 ? owner.patch.tileY : owner.patch.tileX)
          );
        });
        return { ...member, start: parameters[0], end: parameters[1] };
      });
      const parameters = [];
      for (let index = 0; index <= owner.patch.segments; index++) {
        const parameter = index / owner.patch.segments;
        if (
          intervals.every(({ patch, start, end }) => {
            const fraction = (parameter - start) / (end - start);
            return (
              fraction < -1e-10 ||
              fraction > 1 + 1e-10 ||
              Math.abs(fraction * patch.segments - Math.round(fraction * patch.segments)) < 1e-8
            );
          })
        )
          parameters.push(parameter);
        yield;
      }
      const anchorCorners = parameters.map((parameter) => {
        const [east, north] = edgePoint(owner.patch, owner.edge, parameter);
        return corners.get(corner(owner.patch.face, east, north))?.value;
      });
      const key = `${owner.patch.key}:${owner.edge}`;
      const cached = previous.edges?.get(key);
      const same =
        cached?.source === field &&
        parameters.length === cached.line.parameters.length &&
        parameters.every(
          (parameter, index) =>
            parameter === cached.line.parameters[index] &&
            (anchorCorners[index]
              ? anchorCorners[index].every(
                  (value, axis) => value === cached.anchorCorners[index]?.[axis],
                )
              : !cached.anchorCorners[index]),
        );
      let entry = cached;
      if (!same) {
        const vertices = new Float64Array(parameters.length * 6);
        for (const [index, parameter] of parameters.entries()) {
          if (anchorCorners[index]) {
            vertices.set(anchorCorners[index], index * 6);
            yield;
            continue;
          }
          let [east, north] = edgePoint(owner.patch, owner.edge, parameter);
          if (owner.patch.face !== field.face) {
            const point = acrossEdge(owner.patch.face, east - 0.5, north - 0.5, owner.edge, 1);
            east = point.east + 0.5;
            north = point.north + 0.5;
          }
          vertices.set(
            vertex(
              field,
              (east * 2 ** field.depth - field.tileX) * 64,
              (north * 2 ** field.depth - field.tileY) * 64,
            ),
            index * 6,
          );
          yield;
        }
        entry = {
          source: field,
          anchorCorners,
          line: { parameters: Float64Array.from(parameters), vertices },
        };
      }
      edgeCache.set(key, entry);
      for (const interval of intervals) {
        const binding = { line: entry.line, start: interval.start, end: interval.end };
        bindings.get(interval.patch.key)[interval.edge] = binding;
        for (const along of [0, 1]) {
          const [east, north] = edgePoint(interval.patch, interval.edge, along);
          const existing = corners.get(corner(interval.patch.face, east, north));
          corners.set(corner(interval.patch.face, east, north), {
            ...existing,
            value: edgeValue(binding, along * 64),
          });
        }
      }
      yield;
    }
    return {
      patches: patches.map((patch) => ({ ...patch, edgeBindings: bindings.get(patch.key) })),
      cache: { edges: edgeCache, corners: cornerCache },
    };
  }
  function stitchPatches(...args) {
    const steps = stitchPatchSteps(...args);
    let result;
    do result = steps.next();
    while (!result.done);
    return result.value;
  }
  Object.assign(TerrainHiGAN.globe, { stitchPatches, stitchPatchSteps, edgeValue });
})();
