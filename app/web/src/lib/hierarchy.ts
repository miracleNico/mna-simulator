/**
 * Pure hierarchy logic for the schematic editor: entity ports, SUBCKT pin
 * layout, navigation, validation, and the make / duplicate / dissolve
 * transforms. Nothing here touches React state; ``App.tsx`` and the
 * components call these functions and store the results.
 *
 * Model recap (see ``schematic.ts``): a workspace is a list of levels; every
 * level except the root is an *entity* that a ``SUBCKT`` instance can use
 * (``subcircuitId``). An entity's ports are its ``NODE`` markers: a marker
 * named ``in`` becomes the ``port_in`` junction in the payload, and the
 * backend bridges the instance pin ``in`` to it (``_flatten_subcircuits``).
 * The instance's ``pins`` list is a copy of the entity's ports and is kept in
 * step by ``syncEntityPorts``.
 */

import {
  CanvasComponent,
  CanvasWire,
  GND_BOX,
  PinSide,
  SchematicLevel,
  SubcktPinLayout,
  WireEndpoint,
  componentBounds,
  defaultPinSide,
  getPinCoordinates,
  getPinsForType,
  isPinSide,
  normalizeNodeName,
  resolveSubcktLayout,
  snap,
  subcktPinSide
} from "./schematic";

export { defaultPinSide } from "./schematic";

/** User-facing error raised by the hierarchy transforms. */
export class HierarchyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HierarchyError";
  }
}

/** Factory for new element ids; tests pass a deterministic one. */
export type IdFactory = (prefix: string) => string;

const randomId: IdFactory = (prefix) => `${prefix}-${Math.random().toString(36).slice(2, 9)}`;

const PORT_PREFIX = "port_";

class UnionFind {
  private parent = new Map<string, string>();
  add(key: string): void {
    if (!this.parent.has(key)) this.parent.set(key, key);
  }
  find(key: string): string {
    this.add(key);
    let root = key;
    while (this.parent.get(root) !== root) root = this.parent.get(root)!;
    let cur = key;
    while (cur !== root) {
      const next = this.parent.get(cur)!;
      this.parent.set(cur, root);
      cur = next;
    }
    return root;
  }
  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(ra, rb);
  }
}

const pinKey = (componentId: string, pin: string): string => `cp:${componentId}:${pin}`;
const ptKey = (x: number, y: number): string => `pt:${Math.round(x)}:${Math.round(y)}`;
const nameKey = (name: string): string => `nm:${name}`;
const endpointKey = (ep: WireEndpoint): string => (ep.kind === "pin" ? pinKey(ep.componentId, ep.pin) : ptKey(ep.x, ep.y));

function cloneEndpoint(ep: WireEndpoint): WireEndpoint {
  return ep.kind === "pin" ? { kind: "pin", componentId: ep.componentId, pin: ep.pin } : { kind: "point", x: ep.x, y: ep.y };
}

function cloneComponent(c: CanvasComponent): CanvasComponent {
  const { pinLayout: _derived, ...rest } = c;
  return {
    ...rest,
    pins: c.pins ? [...c.pins] : undefined,
    metadata: c.metadata ? { ...c.metadata } : undefined
  };
}

function sameSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every((item) => set.has(item));
}

/* ---- Ports ------------------------------------------------------------ */

/** Port name a NODE marker exports (``port_<name>`` in the payload). */
export function nodeMarkerName(marker: CanvasComponent): string {
  return normalizeNodeName(marker.name || marker.value || marker.id);
}

/** First NODE marker of every port name, in component order. */
export function entityPortMarkers(level: SchematicLevel): Map<string, CanvasComponent> {
  const markers = new Map<string, CanvasComponent>();
  for (const component of level.components) {
    if (component.type !== "NODE") continue;
    const name = nodeMarkerName(component);
    if (!markers.has(name)) markers.set(name, component);
  }
  return markers;
}

/**
 * Ports an entity exports, mirroring the backend's ``_exported_port_names``:
 * every NODE marker name (in component order) plus explicit ``port_*`` level
 * junctions. This is the single source of truth for instance pins.
 */
export function entityPorts(level: SchematicLevel): string[] {
  const ports = Array.from(entityPortMarkers(level).keys());
  const seen = new Set(ports);
  for (const junction of level.junctions ?? []) {
    if (!junction.id.toLowerCase().startsWith(PORT_PREFIX)) continue;
    const name = junction.id.slice(PORT_PREFIX.length);
    if (!seen.has(name)) {
      seen.add(name);
      ports.push(name);
    }
  }
  return ports;
}

/** Side a NODE marker's pin is drawn on: ``metadata.side`` or the default by name. */
export function nodeMarkerSide(marker: CanvasComponent): PinSide {
  const side = marker.metadata?.side;
  return isPinSide(side) ? side : defaultPinSide(nodeMarkerName(marker));
}

/**
 * Pin layout of an instance of ``entity``: each pin goes to the side of the
 * entity's first NODE marker with that name and is ordered by the marker's
 * y (left/right) or x (top/bottom) coordinate. Pins the entity does not
 * export go to their default side after the others.
 */
export function entityPinLayout(entity: SchematicLevel, pins: string[]): SubcktPinLayout {
  const markers = entityPortMarkers(entity);
  const junctions = new Map<string, { x: number; y: number }>();
  for (const junction of entity.junctions ?? []) {
    if (junction.id.toLowerCase().startsWith(PORT_PREFIX)) {
      junctions.set(junction.id.slice(PORT_PREFIX.length), { x: junction.x, y: junction.y });
    }
  }
  type Entry = { pin: string; side: PinSide; x: number; y: number; known: boolean; index: number };
  const entries: Entry[] = pins.map((pin, index) => {
    const marker = markers.get(pin);
    if (marker) return { pin, side: nodeMarkerSide(marker), x: marker.x, y: marker.y, known: true, index };
    const junction = junctions.get(pin);
    if (junction) return { pin, side: defaultPinSide(pin), x: junction.x, y: junction.y, known: true, index };
    return { pin, side: defaultPinSide(pin), x: 0, y: 0, known: false, index };
  });
  const layout: SubcktPinLayout = { left: [], right: [], top: [], bottom: [] };
  for (const side of ["left", "right", "top", "bottom"] as PinSide[]) {
    const vertical = side === "left" || side === "right";
    layout[side] = entries
      .filter((entry) => entry.side === side)
      .sort((a, b) => {
        if (a.known !== b.known) return a.known ? -1 : 1;
        const primary = vertical ? a.y - b.y : a.x - b.x;
        if (primary !== 0) return primary;
        const secondary = vertical ? a.x - b.x : a.y - b.y;
        if (secondary !== 0) return secondary;
        return a.index - b.index;
      })
      .map((entry) => entry.pin);
  }
  return layout;
}

/** Copy of a SUBCKT instance with ``pinLayout`` derived from its entity (others unchanged). */
export function withEntityPinLayout(component: CanvasComponent, levels: SchematicLevel[]): CanvasComponent {
  if (component.type !== "SUBCKT" || !component.subcircuitId) return component;
  const entity = levels.find((level) => level.id === component.subcircuitId);
  if (!entity) return component;
  return { ...component, pinLayout: entityPinLayout(entity, component.pins ?? []) };
}

/** Components of a level ready for display: SUBCKT instances get their entity's pin layout. */
export function decorateSubcktInstances(components: CanvasComponent[], levels: SchematicLevel[]): CanvasComponent[] {
  let changed = false;
  const out = components.map((component) => {
    const decorated = withEntityPinLayout(component, levels);
    if (decorated !== component) changed = true;
    return decorated;
  });
  return changed ? out : components;
}

/* ---- Navigation --------------------------------------------------------- */

/** Path from the root of the hierarchy tree down to ``levelId`` (inclusive). */
export function breadcrumbPath(levels: SchematicLevel[], levelId: string): SchematicLevel[] {
  const byId = new Map(levels.map((level) => [level.id, level]));
  const path: SchematicLevel[] = [];
  const seen = new Set<string>();
  let cur = byId.get(levelId);
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    path.unshift(cur);
    cur = cur.parentId ? byId.get(cur.parentId) : undefined;
  }
  return path;
}

/** Parent of a level in the hierarchy tree (``null`` for a root or an unknown id). */
export function parentLevelId(levels: SchematicLevel[], levelId: string): string | null {
  const level = levels.find((l) => l.id === levelId);
  if (!level?.parentId) return null;
  return levels.some((l) => l.id === level.parentId) ? level.parentId : null;
}

/* ---- Recursion and validation ------------------------------------------ */

/**
 * True when ``entityId`` is ``levelId`` or contains it, directly or through
 * nested SUBCKT instances. Instancing such an entity on ``levelId`` would
 * make the hierarchy recursive.
 */
export function entityContainsLevel(levels: SchematicLevel[], entityId: string, levelId: string): boolean {
  const byId = new Map(levels.map((level) => [level.id, level]));
  const stack = [entityId];
  const seen = new Set<string>();
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (id === levelId) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    const level = byId.get(id);
    if (!level) continue;
    for (const component of level.components) {
      if (component.type === "SUBCKT" && component.subcircuitId) stack.push(component.subcircuitId);
    }
  }
  return false;
}

/** Instance path segment, mirroring the backend's ``_logical_segment``. */
export function logicalSegment(component: CanvasComponent): string {
  const raw = (component.name || component.id).trim();
  const segment = raw.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
  return segment || component.id;
}

function instanceLabel(component: CanvasComponent): string {
  return component.name || component.id;
}

/**
 * First problem with a SUBCKT instance placed on ``hostLevelId``, worded like
 * the backend's messages, or ``null`` when it is fine. ``logicalPrefix`` is
 * the instance path used in port names (``X1.`` at the top level).
 */
export function validateSubcktInstance(
  levels: SchematicLevel[],
  hostLevelId: string,
  instance: CanvasComponent,
  logicalPrefix = `${logicalSegment(instance)}.`
): string | null {
  if (instance.type !== "SUBCKT") return null;
  const label = instanceLabel(instance);
  if (!instance.subcircuitId) {
    return `SUBCKT instance '${label}' is not linked to an entity: choose one in its properties.`;
  }
  const entity = levels.find((level) => level.id === instance.subcircuitId);
  if (!entity || entity.parentId === null) {
    return `SUBCKT instance '${label}' references missing subcircuit '${instance.subcircuitId}'.`;
  }
  if (entityContainsLevel(levels, entity.id, hostLevelId)) {
    const host = levels.find((level) => level.id === hostLevelId);
    return entity.id === hostLevelId
      ? `SUBCKT instance '${label}' is recursive: entity '${entity.title}' cannot contain an instance of itself.`
      : `SUBCKT instance '${label}' is recursive: entity '${entity.title}' contains '${host?.title ?? hostLevelId}'.`;
  }
  const expected = new Set(instance.pins ?? []);
  if (expected.size === 0) {
    return `SUBCKT instance '${label}' requires pins: entity '${entity.title}' has no NODE ports.`;
  }
  const actual = new Set(entityPorts(entity));
  const missing = Array.from(expected).filter((pin) => !actual.has(pin)).sort();
  const extra = Array.from(actual).filter((pin) => !expected.has(pin)).sort();
  if (missing.length === 0 && extra.length === 0) return null;
  const parts: string[] = [];
  if (missing.length > 0) {
    const logical = missing.map((pin) => `${logicalPrefix}${pin}`).join(", ");
    const local = missing.map((pin) => `${PORT_PREFIX}${pin}`).join(", ");
    parts.push(`missing child ports ${logical} (local junctions: ${local})`);
  }
  if (extra.length > 0) {
    parts.push(`unexpected child ports ${extra.map((pin) => `${logicalPrefix}${pin}`).join(", ")}`);
  }
  return `SUBCKT instance '${label}' port mismatch: ${parts.join("; ")}`;
}

/**
 * First hierarchy problem in what a run of ``topLevelId`` would simulate:
 * every SUBCKT instance reachable from it, depth first, with full logical
 * paths. Mirrors the backend checks so Run can refuse before sending.
 */
export function firstHierarchyIssue(levels: SchematicLevel[], topLevelId: string): string | null {
  const byId = new Map(levels.map((level) => [level.id, level]));
  const visit = (levelId: string, prefix: string, stack: string[]): string | null => {
    const level = byId.get(levelId);
    if (!level) return null;
    const names = new Set<string>();
    for (const component of level.components) {
      if (component.type !== "SUBCKT") continue;
      const segment = logicalSegment(component);
      if (names.has(segment)) {
        return `Duplicate SUBCKT logical name '${segment}' under '${prefix ? prefix.slice(0, -1) : "root"}'.`;
      }
      names.add(segment);
      const childPrefix = `${prefix}${segment}.`;
      const issue = validateSubcktInstance(levels, levelId, component, childPrefix);
      if (issue) return issue;
      const entityId = component.subcircuitId!;
      if (stack.includes(entityId)) {
        return `SUBCKT instance '${instanceLabel(component)}' is recursive.`;
      }
      const nested = visit(entityId, childPrefix, [...stack, entityId]);
      if (nested) return nested;
    }
    return null;
  };
  return visit(topLevelId, "", [topLevelId]);
}

export type SubcktDecoration = { title: string | null; issue: string | null };

/** Entity title and validation message for every SUBCKT instance on a level. */
export function subcktDecorations(levels: SchematicLevel[], levelId: string): Map<string, SubcktDecoration> {
  const out = new Map<string, SubcktDecoration>();
  const level = levels.find((l) => l.id === levelId);
  if (!level) return out;
  for (const component of level.components) {
    if (component.type !== "SUBCKT") continue;
    const entity = component.subcircuitId ? levels.find((l) => l.id === component.subcircuitId) : undefined;
    out.set(component.id, {
      title: entity && entity.parentId !== null ? entity.title : null,
      issue: validateSubcktInstance(levels, levelId, component)
    });
  }
  return out;
}

/* ---- Keeping instances in step with their entity ----------------------- */

export type PortSyncReport = {
  entityId: string;
  renamed: Array<[string, string]>;
  added: string[];
  removed: string[];
  /** Wire ends that sat on a removed pin and were turned into free points. */
  detached: number;
  /** Number of instances whose pins were updated. */
  instances: number;
};

/**
 * Propagate a change of an entity's ports (``before`` → ``after`` is one
 * edit of that entity) to every instance: pins follow the new port list,
 * wires on renamed ports move with the rename (a NODE marker keeps its id
 * while its name changes), wires on removed ports are detached into free
 * points at the old pin position. Returns ``after`` unchanged when the port
 * set did not change.
 */
export function syncEntityPorts(
  before: SchematicLevel[],
  after: SchematicLevel[],
  entityId: string
): { levels: SchematicLevel[]; report: PortSyncReport | null } {
  const oldEntity = before.find((level) => level.id === entityId);
  const newEntity = after.find((level) => level.id === entityId);
  if (!oldEntity || !newEntity || newEntity.parentId === null) return { levels: after, report: null };
  const oldPorts = entityPorts(oldEntity);
  const newPorts = entityPorts(newEntity);
  if (sameSet(oldPorts, newPorts)) return { levels: after, report: null };

  const oldSet = new Set(oldPorts);
  const newSet = new Set(newPorts);
  const oldNames = new Map<string, string>();
  for (const c of oldEntity.components) if (c.type === "NODE") oldNames.set(c.id, nodeMarkerName(c));
  const renamed = new Map<string, string>();
  for (const c of newEntity.components) {
    if (c.type !== "NODE") continue;
    const oldName = oldNames.get(c.id);
    const newName = nodeMarkerName(c);
    // A marker that changed its name renames the port when the old name is
    // gone afterwards. If the new name already existed the two ports merge,
    // and the wires follow the marker into the merged port.
    if (oldName && oldName !== newName && !newSet.has(oldName) && !renamed.has(oldName)) {
      renamed.set(oldName, newName);
    }
  }
  const renameTargets = new Set(renamed.values());
  const added = newPorts.filter((pin) => !oldSet.has(pin) && !renameTargets.has(pin));
  const removed = oldPorts.filter((pin) => !newSet.has(pin) && !renamed.has(pin));

  let detached = 0;
  let instances = 0;
  const levels = after.map((level) => {
    let changed = false;
    let wires = level.wires;
    const components = level.components.map((component) => {
      if (component.type !== "SUBCKT" || component.subcircuitId !== entityId) return component;
      instances += 1;
      changed = true;
      const geometry = withEntityPinLayout(component, before);
      const fix = (ep: WireEndpoint): WireEndpoint => {
        if (ep.kind !== "pin" || ep.componentId !== component.id) return ep;
        const target = renamed.get(ep.pin) ?? ep.pin;
        if (newSet.has(target)) return target === ep.pin ? ep : { kind: "pin", componentId: ep.componentId, pin: target };
        detached += 1;
        const p = getPinCoordinates(geometry, ep.pin);
        return { kind: "point", x: snap(p.x), y: snap(p.y) };
      };
      wires = wires.map((wire) => {
        const start = fix(wire.start);
        const end = fix(wire.end);
        return start === wire.start && end === wire.end ? wire : { ...wire, start, end };
      });
      return { ...component, pins: [...newPorts] };
    });
    let next = changed ? { ...level, components, wires } : level;
    if (level.id === entityId) next = { ...next, pins: [...newPorts] };
    return next;
  });
  return {
    levels,
    report: { entityId, renamed: Array.from(renamed.entries()), added, removed, detached, instances }
  };
}

/** One-line status text for a port change. */
export function describePortSync(report: PortSyncReport, entityTitle: string): string {
  const parts: string[] = [];
  if (report.renamed.length) parts.push(`renamed ${report.renamed.map(([a, b]) => `${a} → ${b}`).join(", ")}`);
  if (report.added.length) parts.push(`added ${report.added.join(", ")}`);
  if (report.removed.length) parts.push(`removed ${report.removed.join(", ")}`);
  let text = `Ports of '${entityTitle}': ${parts.join("; ") || "updated"}`;
  text += ` · ${report.instances} instance${report.instances === 1 ? "" : "s"} updated`;
  if (report.detached > 0) {
    text += ` · ${report.detached} wire${report.detached === 1 ? "" : "s"} detached from deleted pin${report.removed.length === 1 ? "" : "s"}`;
  }
  return `${text}.`;
}

/**
 * Move wire ends on a SUBCKT instance from ``oldPins`` to ``newPins`` after
 * its pin list was edited (unlinked instances) or relinked: a pin whose
 * name survives keeps its wires, otherwise the pin at the same position in
 * the list (a rename), otherwise a normalised-name match. Ends with no
 * match become free points at ``locate(pin)`` (the old pin position) when
 * ``locate`` is given, otherwise they keep their old pin name.
 */
export function remapInstancePinEndpoints(
  wires: CanvasWire[],
  componentId: string,
  oldPins: string[],
  newPins: string[],
  locate?: (pin: string) => { x: number; y: number }
): CanvasWire[] {
  const valid = new Set(newPins);
  const resolve = (pin: string): string => {
    if (valid.has(pin)) return pin;
    const index = oldPins.indexOf(pin);
    if (index >= 0 && index < newPins.length && !oldPins.includes(newPins[index])) return newPins[index];
    const cleaned = normalizeNodeName(pin);
    const normalized = newPins.find((candidate) => normalizeNodeName(candidate) === cleaned);
    if (normalized) return normalized;
    if (index >= 0 && index < newPins.length) return newPins[index];
    return pin;
  };
  const fix = (ep: WireEndpoint): WireEndpoint => {
    if (ep.kind !== "pin" || ep.componentId !== componentId) return ep;
    const next = resolve(ep.pin);
    if (!valid.has(next) && locate) {
      const p = locate(ep.pin);
      return { kind: "point", x: snap(p.x), y: snap(p.y) };
    }
    return next === ep.pin ? ep : { kind: "pin", componentId, pin: next };
  };
  return wires.map((wire) => {
    const start = fix(wire.start);
    const end = fix(wire.end);
    return start === wire.start && end === wire.end ? wire : { ...wire, start, end };
  });
}

/* ---- Naming helpers ------------------------------------------------------ */

/** Smallest ``<prefix><n>`` not used as a component name on the level (case-insensitive). */
export function uniqueInstanceName(components: CanvasComponent[], prefix = "X"): string {
  const used = new Set(components.map((c) => c.name.toLowerCase()));
  for (let n = 1; ; n++) {
    const name = `${prefix}${n}`;
    if (!used.has(name.toLowerCase())) return name;
  }
}

function uniqueTitle(levels: SchematicLevel[], base: string): string {
  const used = new Set(levels.map((level) => level.title));
  if (!used.has(base)) return base;
  for (let n = 2; ; n++) {
    const candidate = `${base} ${n}`;
    if (!used.has(candidate)) return candidate;
  }
}

/** Next free "Subckt N" title. */
export function nextSubcktTitle(levels: SchematicLevel[]): string {
  const used = new Set(levels.map((level) => level.title));
  for (let n = levels.filter((level) => level.parentId !== null).length + 1; ; n++) {
    const title = `Subckt ${n}`;
    if (!used.has(title)) return title;
  }
}

/** Fields a pending SUBCKT placement needs to come out linked to ``entityId``. */
export function instanceTemplate(levels: SchematicLevel[], entityId: string): Partial<CanvasComponent> | null {
  const entity = levels.find((level) => level.id === entityId);
  if (!entity || entity.parentId === null) return null;
  return { subcircuitId: entity.id, pins: entityPorts(entity) };
}

/* ---- Make subcircuit from selection ------------------------------------- */

export type MakeSubcircuitOptions = {
  newId?: IdFactory;
  /** Title of the new entity (default: the next free "Subckt N"). */
  title?: string;
};

export type MakeSubcircuitResult = {
  levels: SchematicLevel[];
  newLevelId: string;
  instanceId: string;
  /** Pins of the new instance (= ports of the new entity). */
  ports: string[];
  /** Ports from net labels inside the selection whose net had no outside connection (the parent keeps the label on the pin). */
  internalPorts: string[];
};

type PinRef = { component: CanvasComponent; pin: string; key: string };

type NetInfo = {
  insidePins: PinRef[];
  outsidePins: PinRef[];
  ground: boolean;
  insideNames: string[];
  outsideNames: string[];
};

type Fragment = {
  pins: PinRef[];
  points: Array<{ x: number; y: number; key: string }>;
  hasGround: boolean;
  names: Set<string>;
  /** Outside wire ends that sat on a selected component's pin. */
  cuts: Array<{ wireId: string; end: "start" | "end"; pin: PinRef }>;
};

/**
 * Turn the selected components (and the wires among them) of the active
 * level into a new child entity and replace them by one SUBCKT instance.
 *
 * Every net with a connection inside and outside the selection becomes a
 * port: inside, a NODE marker at the left or right edge of the moved group
 * (named after the net's NODE label when it has one, else ``p1``, ``p2``,
 * ... in left-then-right, top-to-bottom order); outside, the stubs are
 * wired to the matching instance pin. Ground stays global: GND parts in the
 * selection move inside, and each side that loses its ground connection
 * gets a GND symbol of its own. NODE labels that move inside are ports of
 * the new entity (every NODE in an entity is); the parent gets each such
 * label back on its pin unless it still has one, so net names and the
 * parent entity's own ports survive. The result simulates exactly like the
 * original (same nets, same devices).
 */
export function makeSubcircuitFromSelection(
  levels: SchematicLevel[],
  activeLevelId: string,
  selectedIds: Iterable<string>,
  options: MakeSubcircuitOptions = {}
): MakeSubcircuitResult {
  const newId = options.newId ?? randomId;
  const selected = new Set(selectedIds);
  const parent = levels.find((level) => level.id === activeLevelId);
  if (!parent) throw new HierarchyError("The active level no longer exists.");

  // Geometry uses the displayed pin positions of nested SUBCKT instances.
  const geometry = new Map(decorateSubcktInstances(parent.components, levels).map((c) => [c.id, c]));
  const insideIds = new Set(parent.components.filter((c) => selected.has(c.id)).map((c) => c.id));
  const insideComponents = parent.components.filter((c) => insideIds.has(c.id));
  if (!insideComponents.some((c) => c.type !== "NODE" && c.type !== "LABEL" && c.type !== "GND")) {
    throw new HierarchyError("Select at least one part (not only net labels or ground) to make a subcircuit.");
  }
  const pinXY = (ref: PinRef) => getPinCoordinates(geometry.get(ref.component.id) ?? ref.component, ref.pin);

  // 1. Nets of the parent level, as the editor and the backend form them.
  const uf = new UnionFind();
  for (const c of parent.components) for (const pin of getPinsForType(c.type, c)) uf.add(pinKey(c.id, pin));
  for (const c of parent.components) if (c.type === "NODE") uf.union(pinKey(c.id, "n"), nameKey(nodeMarkerName(c)));
  for (const w of parent.wires) uf.union(endpointKey(w.start), endpointKey(w.end));

  const nets = new Map<string, NetInfo>();
  const allNames = new Set<string>();
  for (const c of parent.components) {
    for (const pin of getPinsForType(c.type, c)) {
      const key = pinKey(c.id, pin);
      const root = uf.find(key);
      let info = nets.get(root);
      if (!info) {
        info = { insidePins: [], outsidePins: [], ground: false, insideNames: [], outsideNames: [] };
        nets.set(root, info);
      }
      const inside = insideIds.has(c.id);
      (inside ? info.insidePins : info.outsidePins).push({ component: c, pin, key });
      if (c.type === "GND") info.ground = true;
      if (c.type === "NODE") {
        const name = nodeMarkerName(c);
        allNames.add(name);
        const names = inside ? info.insideNames : info.outsideNames;
        if (!names.includes(name)) names.push(name);
      }
    }
  }
  const netOfKey = (key: string): NetInfo | undefined => nets.get(uf.find(key));
  const kindOf = (info: NetInfo | undefined): "internal" | "external" | "split" =>
    !info || info.insidePins.length === 0 ? "external" : info.outsidePins.length === 0 ? "internal" : "split";

  // 2. Wires: those of nets entirely inside move in, those of nets entirely
  // outside stay; on boundary nets the selected wires (and wires between two
  // selected parts) move in unless they touch an outside pin.
  const isInsidePin = (ep: WireEndpoint) => ep.kind === "pin" && insideIds.has(ep.componentId);
  const isOutsidePin = (ep: WireEndpoint) => ep.kind === "pin" && !insideIds.has(ep.componentId);
  const insideWires: CanvasWire[] = [];
  const outsideWires: CanvasWire[] = [];
  for (const w of parent.wires) {
    const kind = kindOf(netOfKey(endpointKey(w.start)));
    const goesInside =
      kind === "internal"
        ? !isOutsidePin(w.start) && !isOutsidePin(w.end)
        : kind === "split"
          ? (isInsidePin(w.start) && isInsidePin(w.end)) ||
            (selected.has(w.id) && !isOutsidePin(w.start) && !isOutsidePin(w.end))
          : false;
    (goesInside ? insideWires : outsideWires).push(w);
  }
  const insidePointKeys = new Set<string>();
  for (const w of insideWires) for (const ep of [w.start, w.end]) if (ep.kind === "point") insidePointKeys.add(ptKey(ep.x, ep.y));
  const sharedPoints = new Set<string>();
  for (const w of outsideWires) {
    for (const ep of [w.start, w.end]) {
      if (ep.kind === "point" && insidePointKeys.has(ptKey(ep.x, ep.y))) sharedPoints.add(ptKey(ep.x, ep.y));
    }
  }

  // Bounds and centre of the moved group.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const grow = (x: number, y: number) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  };
  for (const c of insideComponents) {
    const b = componentBounds(geometry.get(c.id) ?? c);
    grow(b.x, b.y);
    grow(b.x + b.w, b.y + b.h);
  }
  for (const w of insideWires) for (const ep of [w.start, w.end]) if (ep.kind === "point") grow(ep.x, ep.y);
  const cx = snap((minX + maxX) / 2);
  const cy = snap((minY + maxY) / 2);

  // 3. Fragments: the pieces each side of a boundary net falls into once the
  // other side is gone. Each piece must be tied to the port again.
  const fragmentsOf = (info: NetInfo, side: "inside" | "outside"): Fragment[] => {
    const local = new UnionFind();
    const pins = side === "inside" ? info.insidePins : info.outsidePins;
    const wires = side === "inside" ? insideWires : outsideWires;
    const netWires = wires.filter((w) => netOfKey(endpointKey(w.start)) === info);
    for (const ref of pins) {
      local.add(ref.key);
      if (ref.component.type === "NODE") local.union(ref.key, nameKey(nodeMarkerName(ref.component)));
    }
    const cuts: Array<{ wireId: string; end: "start" | "end"; pin: PinRef; anchor: string }> = [];
    const pointsByKey = new Map<string, { x: number; y: number; key: string }>();
    for (const w of netWires) {
      const ends: Array<["start" | "end", WireEndpoint]> = [["start", w.start], ["end", w.end]];
      const kept: string[] = [];
      for (const [end, ep] of ends) {
        if (side === "outside" && isInsidePin(ep)) {
          const ref = info.insidePins.find((r) => r.key === endpointKey(ep));
          const other = end === "start" ? w.end : w.start;
          if (ref) cuts.push({ wireId: w.id, end, pin: ref, anchor: endpointKey(other) });
          continue;
        }
        const key = endpointKey(ep);
        local.add(key);
        kept.push(key);
        if (ep.kind === "point") pointsByKey.set(key, { x: ep.x, y: ep.y, key });
      }
      if (kept.length === 2) local.union(kept[0], kept[1]);
    }
    const byRoot = new Map<string, Fragment>();
    const fragment = (key: string): Fragment => {
      const root = local.find(key);
      let f = byRoot.get(root);
      if (!f) {
        f = { pins: [], points: [], hasGround: false, names: new Set(), cuts: [] };
        byRoot.set(root, f);
      }
      return f;
    };
    for (const ref of pins) {
      const f = fragment(ref.key);
      f.pins.push(ref);
      if (ref.component.type === "GND") f.hasGround = true;
      if (ref.component.type === "NODE") f.names.add(nodeMarkerName(ref.component));
    }
    for (const point of pointsByKey.values()) fragment(point.key).points.push(point);
    for (const cut of cuts) fragment(cut.anchor).cuts.push({ wireId: cut.wireId, end: cut.end, pin: cut.pin });
    return Array.from(byRoot.values());
  };

  // 4. Ports: one per boundary net that is not ground.
  type Port = {
    info: NetInfo;
    side: "left" | "right";
    refX: number;
    refY: number;
    name: string;
    names: string[];
    hinted: boolean;
    marker?: CanvasComponent;
  };
  const splitNets = Array.from(nets.values()).filter((info) => kindOf(info) === "split");
  const ports: Port[] = splitNets
    .filter((info) => !info.ground)
    .map((info) => {
      let sx = 0;
      let sy = 0;
      for (const ref of info.outsidePins) {
        const p = pinXY(ref);
        sx += p.x;
        sy += p.y;
      }
      const refX = sx / info.outsidePins.length;
      const refY = sy / info.outsidePins.length;
      return { info, side: refX < cx ? "left" : "right", refX, refY, name: "", names: [], hinted: false } as Port;
    });
  const sideOrder = (side: "left" | "right") =>
    ports.filter((p) => p.side === side).sort((a, b) => a.refY - b.refY || a.refX - b.refX);
  const orderedPorts = [...sideOrder("left"), ...sideOrder("right")];
  let generated = 1;
  for (const port of orderedPorts) {
    if (port.info.insideNames.length > 0) {
      port.names = [...port.info.insideNames];
      port.hinted = true;
    } else if (port.info.outsideNames.length > 0) {
      port.names = [port.info.outsideNames[0]];
      port.hinted = true;
    } else {
      while (allNames.has(`p${generated}`)) generated++;
      const name = `p${generated}`;
      allNames.add(name);
      port.names = [name];
    }
    port.name = port.names[0];
  }

  // 5. Build the child entity.
  const newLevelId = newId("level");
  const instanceId = newId("subckt");
  const edgeMarkers: CanvasComponent[] = [];
  const childExtras: CanvasComponent[] = [];
  const childWires: CanvasWire[] = insideWires.map((w) => ({ ...w, start: cloneEndpoint(w.start), end: cloneEndpoint(w.end) }));
  const newWire = (start: WireEndpoint, end: WireEndpoint): CanvasWire => ({ id: newId("w"), start, end });
  const leftX = snap(minX - 60);
  const rightX = snap(maxX + 60);
  for (const side of ["left", "right"] as const) {
    const list = orderedPorts.filter((p) => p.side === side);
    list.forEach((port, i) => {
      const marker: CanvasComponent = {
        id: newId("node"),
        type: "NODE",
        name: port.name,
        value: "node",
        x: side === "left" ? leftX : rightX,
        y: snap(cy - (list.length - 1) * 20 + i * 40),
        rotation: 0,
        mirrored: side === "left" ? true : undefined,
        metadata: { side }
      };
      port.marker = marker;
      edgeMarkers.push(marker);
    });
  }
  const insideAttach = (f: Fragment, crossing: Set<string>): { ep: WireEndpoint; x: number; y: number } => {
    const crossingPin = f.pins.find((ref) => crossing.has(ref.key));
    const pinEp = (ref: PinRef) => {
      const p = pinXY(ref);
      return { ep: { kind: "pin", componentId: ref.component.id, pin: ref.pin } as WireEndpoint, x: p.x, y: p.y };
    };
    if (crossingPin) return pinEp(crossingPin);
    const shared = f.points.find((p) => sharedPoints.has(p.key));
    if (shared) return { ep: { kind: "point", x: shared.x, y: shared.y }, x: shared.x, y: shared.y };
    const device = f.pins.find((ref) => ref.component.type !== "NODE") ?? f.pins[0];
    if (device) return pinEp(device);
    const point = f.points[0];
    return { ep: { kind: "point", x: point.x, y: point.y }, x: point.x, y: point.y };
  };
  const crossingPins = new Set<string>();
  for (const w of outsideWires) for (const ep of [w.start, w.end]) if (isInsidePin(ep)) crossingPins.add(endpointKey(ep));
  const groundBelow = (x: number, y: number, list: CanvasComponent[], wires: CanvasWire[], from: WireEndpoint) => {
    const gnd: CanvasComponent = {
      id: newId("gnd"),
      type: "GND",
      name: uniqueInstanceName([...parent.components, ...list], "GND"),
      value: "0",
      x: snap(x),
      y: snap(y) + 20 + GND_BOX / 2,
      rotation: 0
    };
    list.push(gnd);
    wires.push(newWire(from, { kind: "pin", componentId: gnd.id, pin: "g" }));
  };
  for (const port of orderedPorts) {
    for (const f of fragmentsOf(port.info, "inside")) {
      if (f.names.size > 0) continue; // tied by name: every inside label of this net is a port
      const attach = insideAttach(f, crossingPins);
      childWires.push(newWire(attach.ep, { kind: "pin", componentId: port.marker!.id, pin: "n" }));
    }
  }
  for (const info of splitNets.filter((n) => n.ground)) {
    for (const f of fragmentsOf(info, "inside")) {
      if (f.hasGround) continue;
      const attach = insideAttach(f, crossingPins);
      groundBelow(attach.x, attach.y, childExtras, childWires, attach.ep);
    }
  }
  let childComponents: CanvasComponent[] = [
    ...edgeMarkers,
    ...insideComponents.map(cloneComponent),
    ...childExtras
  ];

  // Move the group so the entity starts near the top-left of its canvas.
  let cMinX = Infinity;
  let cMinY = Infinity;
  for (const c of childComponents) {
    const b = componentBounds(geometry.get(c.id) ?? c);
    cMinX = Math.min(cMinX, b.x);
    cMinY = Math.min(cMinY, b.y);
  }
  for (const w of childWires) {
    for (const ep of [w.start, w.end]) {
      if (ep.kind === "point") {
        cMinX = Math.min(cMinX, ep.x);
        cMinY = Math.min(cMinY, ep.y);
      }
    }
  }
  const tx = Number.isFinite(cMinX) ? snap(120 - cMinX) : 0;
  const ty = Number.isFinite(cMinY) ? snap(100 - cMinY) : 0;
  childComponents = childComponents.map((c) => ({ ...c, x: c.x + tx, y: c.y + ty }));
  const shift = (ep: WireEndpoint): WireEndpoint => (ep.kind === "point" ? { kind: "point", x: ep.x + tx, y: ep.y + ty } : ep);
  const movedChildWires = childWires.map((w) => ({ ...w, start: shift(w.start), end: shift(w.end) }));

  const title = options.title ?? nextSubcktTitle(levels);
  const childLevel: SchematicLevel = {
    id: newLevelId,
    title,
    components: childComponents,
    wires: movedChildWires,
    pins: [],
    parentId: activeLevelId
  };
  const instancePins = entityPorts(childLevel);
  childLevel.pins = [...instancePins];
  if (instancePins.length === 0) {
    throw new HierarchyError("The selection has no connection to the rest of the schematic, so there is nothing to make a port of.");
  }

  // 6. The instance on the parent, wired to the outside stubs.
  const outsideComponents = parent.components.filter((c) => !insideIds.has(c.id));
  const instance: CanvasComponent = {
    id: instanceId,
    type: "SUBCKT",
    name: uniqueInstanceName(parent.components),
    value: "",
    subcircuitId: newLevelId,
    pins: [...instancePins],
    x: cx,
    y: cy,
    rotation: 0
  };
  const instanceLayout = resolveSubcktLayout(instancePins, entityPinLayout(childLevel, instancePins));
  const placed = { ...instance, pinLayout: instanceLayout };
  const instancePin = (pin: string): WireEndpoint => ({ kind: "pin", componentId: instanceId, pin });
  const parentExtras: CanvasComponent[] = [];
  const parentNewWires: CanvasWire[] = [];
  const reanchor = new Map<string, WireEndpoint>(); // `${wireId}:${end}` → new endpoint
  // Labels to keep on the parent, as a NODE marker wired to the matching pin.
  const labelOnParent = new Set<string>();

  for (const port of orderedPorts) {
    let byName = false;
    for (const f of fragmentsOf(port.info, "outside")) {
      if (f.cuts.length > 0) {
        for (const cut of f.cuts) reanchor.set(`${cut.wireId}:${cut.end}`, instancePin(port.name));
        continue;
      }
      const shared = f.points.find((p) => sharedPoints.has(p.key));
      if (shared) {
        parentNewWires.push(newWire(instancePin(port.name), { kind: "point", x: shared.x, y: shared.y }));
      } else if (port.hinted && f.names.has(port.name)) {
        byName = true;
      } else if (f.pins.length > 0) {
        const ref = f.pins[0];
        parentNewWires.push(newWire(instancePin(port.name), { kind: "pin", componentId: ref.component.id, pin: ref.pin }));
      }
    }
    // A piece tied to the net only by its label needs that label at the pin.
    if (byName) labelOnParent.add(port.name);
    // Further labels of the same net inside the selection are ports too; they
    // are the same net, so tie their pins to the main one.
    for (const extra of port.names.slice(1)) {
      parentNewWires.push(newWire(instancePin(extra), instancePin(port.name)));
    }
  }
  for (const info of splitNets.filter((n) => n.ground)) {
    for (const f of fragmentsOf(info, "outside")) {
      for (const cut of f.cuts) {
        const p = pinXY(cut.pin);
        const gnd: CanvasComponent = {
          id: newId("gnd"),
          type: "GND",
          name: uniqueInstanceName([...parent.components, ...parentExtras], "GND"),
          value: "0",
          x: snap(p.x),
          y: snap(p.y) + GND_BOX / 2,
          rotation: 0
        };
        parentExtras.push(gnd);
        reanchor.set(`${cut.wireId}:${cut.end}`, { kind: "pin", componentId: gnd.id, pin: "g" });
      }
      const shared = f.points.filter((p) => sharedPoints.has(p.key));
      for (const point of shared) {
        groundBelow(point.x, point.y, parentExtras, parentNewWires, { kind: "point", x: point.x, y: point.y });
      }
      if (!f.hasGround && f.cuts.length === 0 && shared.length === 0 && f.pins.length > 0) {
        const p = pinXY(f.pins[0]);
        groundBelow(p.x, p.y, parentExtras, parentNewWires, {
          kind: "pin",
          componentId: f.pins[0].component.id,
          pin: f.pins[0].pin
        });
      }
    }
  }

  // Every label that moved inside is a port of the new entity. Keep a label
  // of the same name on the parent at that pin unless the parent still has
  // one, so net names survive (probes keep working) and, when the parent is
  // itself an entity, its own ports stay as they were.
  const remainingNames = new Set(outsideComponents.filter((c) => c.type === "NODE").map(nodeMarkerName));
  for (const c of insideComponents) {
    if (c.type !== "NODE") continue;
    const name = nodeMarkerName(c);
    if (!remainingNames.has(name)) labelOnParent.add(name);
  }
  for (const name of instancePins.filter((pin) => labelOnParent.has(pin))) {
    const p = getPinCoordinates(placed, name);
    const side = subcktPinSide(instanceLayout, name) ?? "right";
    const dx = side === "left" ? -40 : side === "right" ? 40 : 0;
    const dy = side === "top" ? -40 : side === "bottom" ? 40 : 0;
    const marker: CanvasComponent = {
      id: newId("node"),
      type: "NODE",
      name,
      value: "node",
      x: snap(p.x + dx),
      y: snap(p.y + dy),
      rotation: 0,
      mirrored: side === "left" ? true : undefined
    };
    parentExtras.push(marker);
    parentNewWires.push(newWire(instancePin(name), { kind: "pin", componentId: marker.id, pin: "n" }));
  }

  const parentWires: CanvasWire[] = outsideWires.map((w) => {
    const fixEnd = (end: "start" | "end", ep: WireEndpoint): WireEndpoint => {
      const replacement = reanchor.get(`${w.id}:${end}`);
      if (replacement) return replacement;
      if (isInsidePin(ep) && ep.kind === "pin") {
        // Safety net: never leave a wire end on a part that moved away.
        const moved = geometry.get(ep.componentId);
        const p = moved ? getPinCoordinates(moved, ep.pin) : { x: 0, y: 0 };
        return { kind: "point", x: snap(p.x), y: snap(p.y) };
      }
      return ep;
    };
    const start = fixEnd("start", w.start);
    const end = fixEnd("end", w.end);
    return start === w.start && end === w.end ? w : { ...w, start, end };
  });

  const newParent: SchematicLevel = {
    ...parent,
    components: [...outsideComponents, instance, ...parentExtras],
    wires: [...parentWires, ...parentNewWires]
  };

  // Entities only used by instances that moved inside now live under the new entity.
  const movedInstances = insideComponents.filter((c) => c.type === "SUBCKT" && c.subcircuitId);
  const reparent = new Set<string>();
  for (const inst of movedInstances) {
    const entityId = inst.subcircuitId!;
    const entity = levels.find((level) => level.id === entityId);
    if (!entity || entity.parentId !== activeLevelId) continue;
    const usedElsewhere = levels.some((level) =>
      level.components.some(
        (c) => c.type === "SUBCKT" && c.subcircuitId === entityId && !(level.id === activeLevelId && insideIds.has(c.id))
      )
    );
    if (!usedElsewhere) reparent.add(entityId);
  }

  const nextLevels = levels.map((level) => {
    if (level.id === activeLevelId) return newParent;
    if (reparent.has(level.id)) return { ...level, parentId: newLevelId };
    return level;
  });
  nextLevels.push(childLevel);
  const connected = new Set(orderedPorts.flatMap((p) => p.names));
  return {
    levels: nextLevels,
    newLevelId,
    instanceId,
    ports: instancePins,
    internalPorts: instancePins.filter((pin) => !connected.has(pin))
  };
}

/* ---- Duplicate entity ---------------------------------------------------- */

/**
 * Deep copy of an entity and of every entity defined below it in the tree,
 * with new ids. Instances inside the copy that used a copied entity are
 * relinked to the copy; links to entities outside the subtree are kept.
 */
export function duplicateEntity(
  levels: SchematicLevel[],
  entityId: string,
  options: { newId?: IdFactory } = {}
): { levels: SchematicLevel[]; newLevelId: string } {
  const newId = options.newId ?? randomId;
  const entity = levels.find((level) => level.id === entityId);
  if (!entity || entity.parentId === null) throw new HierarchyError("Only a child entity can be duplicated.");
  const subtree: SchematicLevel[] = [entity];
  for (let i = 0; i < subtree.length; i++) {
    for (const level of levels) if (level.parentId === subtree[i].id && !subtree.includes(level)) subtree.push(level);
  }
  const levelIds = new Map(subtree.map((level) => [level.id, newId("level")]));
  const working = [...levels];
  const copies: SchematicLevel[] = subtree.map((level) => {
    const componentIds = new Map(level.components.map((c) => [c.id, newId(c.type.toLowerCase())]));
    const remap = (ep: WireEndpoint): WireEndpoint =>
      ep.kind === "pin"
        ? { kind: "pin", componentId: componentIds.get(ep.componentId) ?? ep.componentId, pin: ep.pin }
        : { kind: "point", x: ep.x, y: ep.y };
    const copy: SchematicLevel = {
      ...level,
      id: levelIds.get(level.id)!,
      title: uniqueTitle(working, `${level.title} copy`),
      parentId: level.id === entity.id ? entity.parentId : levelIds.get(level.parentId!) ?? level.parentId,
      components: level.components.map((c) => {
        const copied = cloneComponent(c);
        copied.id = componentIds.get(c.id)!;
        if (c.type === "SUBCKT" && c.subcircuitId && levelIds.has(c.subcircuitId)) {
          copied.subcircuitId = levelIds.get(c.subcircuitId);
        }
        return copied;
      }),
      wires: level.wires.map((w) => ({ id: newId("w"), start: remap(w.start), end: remap(w.end) })),
      pins: [...level.pins],
      junctions: level.junctions?.map((j) => ({ ...j }))
    };
    working.push(copy);
    return copy;
  });
  return { levels: [...levels, ...copies], newLevelId: copies[0].id };
}

/* ---- Dissolve subcircuit -------------------------------------------------- */

/**
 * Inline an instance's entity into the level in place of the instance: the
 * entity's parts get new ids and keep their relative layout centred on the
 * instance; every port marker (every NODE marker of the entity is a port)
 * becomes a plain wire corner that both the inner wires and the outer stubs
 * meet at, so no inner label can short against a net of the parent. The
 * entity itself stays (other instances may use it).
 */
export function dissolveSubcircuit(
  levels: SchematicLevel[],
  levelId: string,
  instanceId: string,
  options: { newId?: IdFactory } = {}
): { levels: SchematicLevel[]; addedIds: string[] } {
  const newId = options.newId ?? randomId;
  const host = levels.find((level) => level.id === levelId);
  const instance = host?.components.find((c) => c.id === instanceId);
  if (!host || !instance || instance.type !== "SUBCKT") throw new HierarchyError("Select one SUBCKT instance to dissolve.");
  const issue = validateSubcktInstance(levels, levelId, instance);
  if (issue) throw new HierarchyError(issue);
  const entity = levels.find((level) => level.id === instance.subcircuitId)!;
  const ports = new Set(entityPorts(entity));
  const placedInstance = withEntityPinLayout(instance, levels);

  // Geometry of the entity (bounds centre goes to the instance position).
  const entityGeometry = decorateSubcktInstances(entity.components, levels);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const c of entityGeometry) {
    const b = componentBounds(c);
    minX = Math.min(minX, b.x);
    minY = Math.min(minY, b.y);
    maxX = Math.max(maxX, b.x + b.w);
    maxY = Math.max(maxY, b.y + b.h);
  }
  if (!Number.isFinite(minX)) {
    minX = maxX = instance.x;
    minY = maxY = instance.y;
  }
  let dx = snap(instance.x - (minX + maxX) / 2);
  let dy = snap(instance.y - (minY + maxY) / 2);

  // Avoid landing an inner free point on a free point of the host (points
  // with equal coordinates are the same net).
  const hostPoints = new Set<string>();
  for (const w of host.wires) for (const ep of [w.start, w.end]) if (ep.kind === "point") hostPoints.add(ptKey(ep.x, ep.y));
  const innerPoints: Array<{ x: number; y: number }> = [];
  for (const w of entity.wires) for (const ep of [w.start, w.end]) if (ep.kind === "point") innerPoints.push(ep);
  for (const c of entity.components) if (c.type === "NODE" && ports.has(nodeMarkerName(c))) innerPoints.push({ x: c.x, y: c.y });
  for (const j of entity.junctions ?? []) innerPoints.push({ x: j.x, y: j.y });
  for (let attempt = 0; attempt < 50 && innerPoints.some((p) => hostPoints.has(ptKey(p.x + dx, p.y + dy))); attempt++) {
    dx += 10;
    dy += 10;
  }

  const prefix = logicalSegment(instance);
  const ids = new Map<string, string>();
  const added: CanvasComponent[] = [];
  const portPoint = new Map<string, { x: number; y: number }>(); // port → first marker position
  const extraPortPoints: Array<{ name: string; x: number; y: number }> = [];
  const usedNames = host.components.filter((c) => c.id !== instanceId);
  for (const c of entity.components) {
    if (c.type === "NODE") {
      const name = nodeMarkerName(c);
      if (ports.has(name)) {
        const point = { x: c.x + dx, y: c.y + dy };
        if (!portPoint.has(name)) portPoint.set(name, point);
        else extraPortPoints.push({ name, ...point });
        continue;
      }
    }
    const copy = cloneComponent(c);
    copy.id = newId(c.type.toLowerCase());
    copy.x = c.x + dx;
    copy.y = c.y + dy;
    if (c.type === "SUBCKT" && usedNames.some((u) => u.name.toLowerCase() === c.name.toLowerCase())) {
      copy.name = uniqueInstanceName([...usedNames, ...added], `${prefix}_${c.name}_`);
    }
    ids.set(c.id, copy.id);
    added.push(copy);
  }
  for (const j of entity.junctions ?? []) {
    if (!j.id.toLowerCase().startsWith(PORT_PREFIX)) continue;
    const name = j.id.slice(PORT_PREFIX.length);
    if (!portPoint.has(name)) portPoint.set(name, { x: snap(j.x) + dx, y: snap(j.y) + dy });
  }

  const markerIds = new Set(entity.components.filter((c) => c.type === "NODE" && ports.has(nodeMarkerName(c))).map((c) => c.id));
  const markerById = new Map(entity.components.map((c) => [c.id, c]));
  const innerEndpoint = (ep: WireEndpoint): WireEndpoint => {
    if (ep.kind === "point") return { kind: "point", x: ep.x + dx, y: ep.y + dy };
    if (markerIds.has(ep.componentId)) {
      const marker = markerById.get(ep.componentId)!;
      return { kind: "point", x: marker.x + dx, y: marker.y + dy };
    }
    return { kind: "pin", componentId: ids.get(ep.componentId) ?? ep.componentId, pin: ep.pin };
  };
  const newWires: CanvasWire[] = entity.wires.map((w) => ({ id: newId("w"), start: innerEndpoint(w.start), end: innerEndpoint(w.end) }));
  // Same-named port markers were tied by name; tie them with wires now.
  for (const extra of extraPortPoints) {
    const first = portPoint.get(extra.name)!;
    newWires.push({ id: newId("w"), start: { kind: "point", x: extra.x, y: extra.y }, end: { kind: "point", x: first.x, y: first.y } });
  }
  const hostWires = host.wires.map((w) => {
    const fix = (ep: WireEndpoint): WireEndpoint => {
      if (ep.kind !== "pin" || ep.componentId !== instanceId) return ep;
      const target = portPoint.get(ep.pin);
      if (target) return { kind: "point", x: target.x, y: target.y };
      const p = getPinCoordinates(placedInstance, ep.pin);
      return { kind: "point", x: snap(p.x), y: snap(p.y) };
    };
    const start = fix(w.start);
    const end = fix(w.end);
    return start === w.start && end === w.end ? w : { ...w, start, end };
  });
  const nextHost: SchematicLevel = {
    ...host,
    components: [...host.components.filter((c) => c.id !== instanceId), ...added],
    wires: [...hostWires, ...newWires]
  };
  return {
    levels: levels.map((level) => (level.id === levelId ? nextHost : level)),
    addedIds: [...added.map((c) => c.id), ...newWires.map((w) => w.id)]
  };
}
