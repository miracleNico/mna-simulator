/**
 * Frontend data model for the schematic editor.
 *
 * Kept in lock-step with backend `SchematicDocument` in
 * `mna_simulation/api/contracts.py`. Only the fields exchanged with the API
 * must match exactly; UI-only fields (position, rotation, etc.) are local.
 */

export type DeviceType =
  | "R"
  | "C"
  | "L"
  | "V"
  | "I"
  | "D"
  | "GND"
  | "VCVS"
  | "VCCS"
  | "CCCS"
  | "CCVS"
  | "QNPN"
  | "QPNP"
  | "NMOS"
  | "PMOS"
  | "LABEL"
  | "NODE"
  | "SUBCKT";

export type BasicAnalysisMode = "op" | "tran" | "ac" | "hb" | "dyn";
export type KrylovMethod = "auto" | "arnoldi_gmres" | "conjugate_residual" | "conjugate_gradient";
export type MorMethod = "auto" | "linear_krylov" | "tpwl";
/** Worker threads requested for a run: "auto" or a positive integer. */
export type ThreadSetting = "auto" | number;

export type Rotation = 0 | 90 | 180 | 270;

export type CanvasComponent = {
  id: string;
  type: DeviceType;
  name: string;
  value: string;
  subtype?: string;
  value2?: string;
  value3?: string;
  ctrlNode1?: string;
  ctrlNode2?: string;
  ctrlSource?: string;
  subcircuitId?: string;
  pins?: string[];
  metadata?: Record<string, string>;
  x: number;
  y: number;
  rotation: Rotation;
  mirrored?: boolean;
};

/** Wire endpoint may be anchored to a component pin or float as a free point. */
export type WireEndpoint =
  | { kind: "pin"; componentId: string; pin: string }
  | { kind: "point"; x: number; y: number };

export type CanvasWire = {
  id: string;
  start: WireEndpoint;
  end: WireEndpoint;
};

export type AnalysisState = {
  mode: BasicAnalysisMode;
  tStop: string;
  tStep: string;
  fStart: string;
  fStop: string;
  points: number;
  harmonics: number;
  /** Optional .hb time-domain reconstruction window, e.g. "200m"; empty = spectrum. */
  hbTimeWindow: string;
  /** For .dyn: simulation-seconds per wall-second. */
  dynSpeed: string;
  /** Rolling sim-time window for live .dyn scope (e.g. "5m" = 5 ms). */
  dynWindow: string;
  /** Selected nodes for .tran result display (empty = all). */
  probeNodes: string[];
  /** If true, backend auto-selects a Krylov linear solve path. */
  krylov: boolean;
  /** Auto = backend uses 50% of the actual generated matrix dimension. */
  krylovRankMode: "auto" | "manual";
  /** Manual Krylov rank / iteration budget. Must be a positive integer. */
  krylovRank: number;
  /** Auto or explicit Krylov algorithm route. */
  krylovMethod: KrylovMethod;
  /** If true, backend reduces the model before selected-output solves. */
  mor: boolean;
  /** Auto routes .ac/linear .tran to linear Krylov MOR and nonlinear .tran to TPWL. */
  morMethod: MorMethod;
  /** Auto uses the backend input/output heuristic; manual sends the integer order. */
  morOrderMode: "auto" | "manual";
  /** Manual reduced order. Must be a positive integer. */
  morOrder: number;
  /** Output labels kept by MOR, independent of plot display filtering. */
  morOutputNodes: string[];
  /** Threads requested per run; the server grants at most what its budget has free. */
  numThreads: ThreadSetting;
};

/**
 * Named junction inside a hierarchy level. When a level is instantiated as
 * a SUBCKT on a parent canvas, junctions whose id matches ``port_<pinname>``
 * become the electrical bridge between outer pins and inner wires — see
 * ``_flatten_subcircuits`` in the Python pipeline.
 */
export type LevelJunction = {
  id: string;
  x: number;
  y: number;
};

export type SchematicLevel = {
  id: string;
  title: string;
  components: CanvasComponent[];
  wires: CanvasWire[];
  /** Declared external pin names for this level (used when instanced as SUBCKT). */
  pins: string[];
  /** Optional named junctions; usually ``port_<pin>`` markers for SUBCKT use. */
  junctions?: LevelJunction[];
  /** Parent level id for the hierarchy tree (null = root). */
  parentId: string | null;
};

export type SchematicPreset = {
  id: string;
  title: string;
  description: string;
  components: CanvasComponent[];
  wires: CanvasWire[];
  analysis: AnalysisState;
  /** Additional hierarchy levels beyond the root, e.g. for a multi-stage amp demo. */
  extraLevels?: Array<Omit<SchematicLevel, "parentId"> & { parentId: string | null }>;
};

/* --- Geometry / symbol layout --- */

export const GRID = 10;
export const COMPONENT_BOX = 72; // square hit-area for most two-terminal parts
export const GND_BOX = 48;

export type PinOffset = { dx: number; dy: number };

export const DEVICE_PIN_OFFSETS: Record<DeviceType, Record<string, PinOffset>> = {
  R: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  C: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  L: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  V: { p: { dx: 0, dy: -COMPONENT_BOX / 2 }, n: { dx: 0, dy: COMPONENT_BOX / 2 } },
  I: { p: { dx: 0, dy: -COMPONENT_BOX / 2 }, n: { dx: 0, dy: COMPONENT_BOX / 2 } },
  D: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  GND: { g: { dx: 0, dy: -GND_BOX / 2 } },
  VCVS: {
    p: { dx: COMPONENT_BOX / 2, dy: -COMPONENT_BOX / 4 },
    n: { dx: COMPONENT_BOX / 2, dy: COMPONENT_BOX / 4 },
    cp: { dx: -COMPONENT_BOX / 2, dy: -COMPONENT_BOX / 4 },
    cn: { dx: -COMPONENT_BOX / 2, dy: COMPONENT_BOX / 4 }
  },
  VCCS: {
    p: { dx: COMPONENT_BOX / 2, dy: -COMPONENT_BOX / 4 },
    n: { dx: COMPONENT_BOX / 2, dy: COMPONENT_BOX / 4 },
    cp: { dx: -COMPONENT_BOX / 2, dy: -COMPONENT_BOX / 4 },
    cn: { dx: -COMPONENT_BOX / 2, dy: COMPONENT_BOX / 4 }
  },
  CCCS: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  CCVS: { p: { dx: -COMPONENT_BOX / 2, dy: 0 }, n: { dx: COMPONENT_BOX / 2, dy: 0 } },
  QNPN: {
    c: { dx: 0, dy: -COMPONENT_BOX / 2 },
    b: { dx: -COMPONENT_BOX / 2, dy: 0 },
    e: { dx: 0, dy: COMPONENT_BOX / 2 }
  },
  QPNP: {
    c: { dx: 0, dy: COMPONENT_BOX / 2 },
    b: { dx: -COMPONENT_BOX / 2, dy: 0 },
    e: { dx: 0, dy: -COMPONENT_BOX / 2 }
  },
  NMOS: {
    d: { dx: 0, dy: -COMPONENT_BOX / 2 },
    g: { dx: -COMPONENT_BOX / 2, dy: 0 },
    s: { dx: 0, dy: COMPONENT_BOX / 2 }
  },
  PMOS: {
    d: { dx: 0, dy: COMPONENT_BOX / 2 },
    g: { dx: -COMPONENT_BOX / 2, dy: 0 },
    s: { dx: 0, dy: -COMPONENT_BOX / 2 }
  },
  LABEL: {},
  NODE: { n: { dx: 0, dy: 0 } },
  SUBCKT: {}
};

const DEFAULT_VALUES: Record<DeviceType, string> = {
  R: "1k",
  C: "1u",
  L: "1m",
  V: "5",
  I: "1m",
  D: "1e-15",
  GND: "0",
  VCVS: "1",
  VCCS: "1",
  CCCS: "1",
  CCVS: "1",
  QNPN: "1e-15", // Level-1 IS (A)
  QPNP: "1e-15",
  NMOS: "1m", // Level-1 beta
  PMOS: "1m",
  LABEL: "node",
  NODE: "node",
  SUBCKT: "1"
};

export const DEVICE_LABELS: Record<DeviceType, string> = {
  R: "Resistor",
  C: "Capacitor",
  L: "Inductor",
  V: "V Source",
  I: "I Source",
  D: "Diode",
  GND: "Ground",
  VCVS: "VCVS (E)",
  VCCS: "VCCS (G)",
  CCCS: "CCCS (F)",
  CCVS: "CCVS (H)",
  QNPN: "BJT (NPN)",
  QPNP: "BJT (PNP)",
  NMOS: "NMOS",
  PMOS: "PMOS",
  LABEL: "Label",
  NODE: "Node",
  SUBCKT: "Subckt"
};

export function snap(value: number, grid = GRID): number {
  return Math.round(value / grid) * grid;
}

export function getPinsForType(type: DeviceType, component?: CanvasComponent): string[] {
  if (type === "SUBCKT") {
    return component?.pins ?? [];
  }
  return Object.keys(DEVICE_PIN_OFFSETS[type] ?? {});
}

export function rotatePinOffset(offset: PinOffset, rotation: Rotation): PinOffset {
  switch (rotation) {
    case 90:
      return { dx: -offset.dy, dy: offset.dx };
    case 180:
      return { dx: -offset.dx, dy: -offset.dy };
    case 270:
      return { dx: offset.dy, dy: -offset.dx };
    default:
      return offset;
  }
}

function transformPinOffset(offset: PinOffset, component: CanvasComponent): PinOffset {
  const mirrored = component.mirrored ? { dx: -offset.dx, dy: offset.dy } : offset;
  return rotatePinOffset(mirrored, component.rotation);
}

export function getPinCoordinates(component: CanvasComponent, pin: string): { x: number; y: number } {
  if (component.type === "SUBCKT") {
    // Distribute pins around a rounded rectangle.
    const off = subcktPinLocalOffset(component.pins ?? [], pin);
    const rotated = transformPinOffset(off, component);
    return { x: component.x + rotated.dx, y: component.y + rotated.dy };
  }
  const raw = DEVICE_PIN_OFFSETS[component.type]?.[pin];
  if (!raw) {
    return { x: component.x, y: component.y };
  }
  const rotated = transformPinOffset(raw, component);
  return { x: component.x + rotated.dx, y: component.y + rotated.dy };
}

function resolveComponentPin(component: CanvasComponent, pin: string): string {
  const pins = getPinsForType(component.type, component);
  if (pins.includes(pin)) return pin;
  if (component.type !== "SUBCKT") return pin;

  const cleaned = normalizeNodeName(pin);
  const exactNormalized = pins.find((candidate) => normalizeNodeName(candidate) === cleaned);
  if (exactNormalized) return exactNormalized;

  const trimmed = cleaned.replace(/_+$/, "");
  if (!trimmed) return pin;
  return (
    pins.find((candidate) => {
      const normalized = normalizeNodeName(candidate);
      return normalized === trimmed || normalized.startsWith(trimmed) || cleaned.startsWith(normalized);
    }) ?? pin
  );
}

export function componentBounds(component: CanvasComponent): { x: number; y: number; w: number; h: number } {
  if (component.type === "GND") {
    return { x: component.x - GND_BOX / 2, y: component.y - GND_BOX / 2, w: GND_BOX, h: GND_BOX };
  }
  if (component.type === "LABEL") {
    return { x: component.x - 38, y: component.y - 12, w: 76, h: 24 };
  }
  if (component.type === "NODE") {
    return { x: component.x - 28, y: component.y - 18, w: 56, h: 36 };
  }
  if (component.type === "SUBCKT") {
    return {
      x: component.x - COMPONENT_BOX,
      y: component.y - COMPONENT_BOX,
      w: COMPONENT_BOX * 2,
      h: COMPONENT_BOX * 2
    };
  }
  return { x: component.x - COMPONENT_BOX / 2, y: component.y - COMPONENT_BOX / 2, w: COMPONENT_BOX, h: COMPONENT_BOX };
}

export function createDefaultComponent(
  type: DeviceType,
  index: number,
  x: number,
  y: number
): CanvasComponent {
  const namePrefix =
    type === "GND" ? "GND" :
    type === "QNPN" || type === "QPNP" ? "Q" :
    type === "NMOS" || type === "PMOS" ? "M" :
    type === "LABEL" ? "label" :
    type === "NODE" ? "node" :
    type;
  return {
    id: `${type.toLowerCase()}-${index}-${Math.random().toString(36).slice(2, 7)}`,
    type,
    name: `${namePrefix}${index}`,
    value: DEFAULT_VALUES[type],
    subtype: type === "V" || type === "I" ? "DC" : undefined,
    value2:
      type === "QNPN" || type === "QPNP" ? "150" :
      type === "NMOS" || type === "PMOS" ? "0.7" :
      undefined,
    value3:
      type === "QNPN" || type === "QPNP" ? "3" :
      type === "NMOS" || type === "PMOS" ? "0.02" :
      undefined,
    metadata:
      type === "QNPN" || type === "QPNP"
        ? { model: "level1", vaf: "100", var: "25", cje: "4p", cjc: "2p", rb: "50", re: "0.5", rc: "5" }
        : type === "NMOS" || type === "PMOS"
          ? { model: "level1", cgs: "2p", cgd: "1p" }
          : undefined,
    pins: type === "SUBCKT" ? ["in_1", "out_1"] : undefined,
    subcircuitId: type === "SUBCKT" ? "" : undefined,
    x: snap(x),
    y: snap(y),
    rotation: 0
  };
}

export function pinEndpoint(componentId: string, pin: string): WireEndpoint {
  return { kind: "pin", componentId, pin };
}
export function pointEndpoint(x: number, y: number): WireEndpoint {
  return { kind: "point", x: snap(x), y: snap(y) };
}

function endpointKey(ep: WireEndpoint): string {
  return ep.kind === "pin"
    ? `pin:${ep.componentId}:${ep.pin}`
    : `pt:${Math.round(ep.x)}:${Math.round(ep.y)}`;
}

export function wireKey(wire: CanvasWire): string {
  const ordered = [endpointKey(wire.start), endpointKey(wire.end)].sort();
  return `${ordered[0]}|${ordered[1]}`;
}

export function endpointXY(
  ep: WireEndpoint,
  componentsById: Map<string, CanvasComponent>
): { x: number; y: number } | null {
  if (ep.kind === "point") return { x: ep.x, y: ep.y };
  const c = componentsById.get(ep.componentId);
  if (!c) return null;
  return getPinCoordinates(c, ep.pin);
}

/** Endpoint key used for de-duplicating free wire points into shared junctions. */
function pointKey(x: number, y: number): string {
  return `${Math.round(x)}:${Math.round(y)}`;
}

/* ---- Frontend net-name computation ----------------------------------- */

class UnionFind {
  private parent = new Map<string, string>();
  add(k: string): void {
    if (!this.parent.has(k)) this.parent.set(k, k);
  }
  find(k: string): string {
    let p = this.parent.get(k) ?? k;
    if (p !== k) {
      p = this.find(p);
      this.parent.set(k, p);
    }
    return p;
  }
  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(ra, rb);
  }
}

/**
 * Assign a net name to every pin/free-point endpoint on the canvas.
 *
 * Mirrors the backend's union-find net formation so the user can see the same
 * net labels the simulator will compute. GND component pins anchor a special
 * "0" net. The result maps every wire-endpoint key (pin or coordinate) to its
 * net name, plus a per-wire net map for fast canvas drawing.
 */
export type NetAssignment = {
  /** ``cp:<componentId>:<pin>`` or ``pt:x:y`` → net name */
  byKey: Map<string, string>;
  /** wire id → net name (or "?" when an endpoint is unresolved) */
  byWire: Map<string, string>;
};

export function computeNetAssignments(
  components: CanvasComponent[],
  wires: CanvasWire[]
): NetAssignment {
  const uf = new UnionFind();
  const allKeys: string[] = [];

  function pinKey(componentId: string, pin: string): string {
    return `cp:${componentId}:${pin}`;
  }
  function ptKey(x: number, y: number): string {
    return `pt:${Math.round(x)}:${Math.round(y)}`;
  }
  function epKey(ep: WireEndpoint): string {
    return ep.kind === "pin" ? pinKey(ep.componentId, ep.pin) : ptKey(ep.x, ep.y);
  }

  // Seed every component pin.
  for (const c of components) {
    for (const pin of getPinsForType(c.type, c)) {
      const k = pinKey(c.id, pin);
      uf.add(k);
      allKeys.push(k);
    }
  }
  const nodeRoots = new Map<string, string>();
  for (const c of components) {
    if (c.type !== "NODE") continue;
    const k = pinKey(c.id, "n");
    const nodeName = normalizeNodeName(c.name || c.value || c.id);
    const first = nodeRoots.get(nodeName);
    if (first) uf.union(first, k);
    else nodeRoots.set(nodeName, k);
  }
  // Seed wire endpoints (free points get their own keys) and union the pair.
  for (const w of wires) {
    const a = epKey(w.start);
    const b = epKey(w.end);
    uf.add(a);
    uf.add(b);
    if (!allKeys.includes(a)) allKeys.push(a);
    if (!allKeys.includes(b)) allKeys.push(b);
    uf.union(a, b);
  }

  // Identify ground roots.
  const groundRoots = new Set<string>();
  for (const c of components) {
    if (c.type === "GND") {
      const gk = pinKey(c.id, "g");
      uf.add(gk);
      groundRoots.add(uf.find(gk));
    }
  }

  const rootToNet = new Map<string, string>();
  for (const c of components) {
    if (c.type !== "NODE") continue;
    const r = uf.find(pinKey(c.id, "n"));
    if (!rootToNet.has(r)) rootToNet.set(r, normalizeNodeName(c.name || c.value || c.id));
  }
  let counter = 1;
  for (const k of allKeys) {
    const r = uf.find(k);
    if (rootToNet.has(r)) continue;
    if (groundRoots.has(r)) rootToNet.set(r, "0");
    else rootToNet.set(r, `n${counter++}`);
  }

  const byKey = new Map<string, string>();
  for (const k of allKeys) {
    byKey.set(k, rootToNet.get(uf.find(k)) ?? "?");
  }
  const byWire = new Map<string, string>();
  for (const w of wires) {
    const k = epKey(w.start);
    byWire.set(w.id, byKey.get(k) ?? "?");
  }
  return { byKey, byWire };
}

export function normalizeNodeName(raw: string): string {
  const cleaned = (raw || "node")
    .trim()
    .replace(/^port_/i, "")
    .replace(/[^A-Za-z0-9_]/g, "_");
  return cleaned || "node";
}

/* ---- Transistor small-signal expansion -------------------------------- */

type ExpandedComponent = {
  id: string;
  type: DeviceType;
  name: string;
  value: string | null;
  subtype: string | null;
  value2: string | null;
  value3: string | null;
  ctrl_node1: string | null;
  ctrl_node2: string | null;
  ctrl_source: string | null;
  subcircuit_id: string | null;
  pins: string[] | null;
  position: { x: number; y: number };
  metadata: Record<string, string>;
};

type ExpandedWire = {
  id: string;
  start:
    | { kind: "component_pin"; component_id: string; pin: string }
    | { kind: "junction"; junction_id: string };
  end:
    | { kind: "component_pin"; component_id: string; pin: string }
    | { kind: "junction"; junction_id: string };
};

type ExpandedJunction = { id: string; position?: { x: number; y: number } };

function expandSchematic(
  components: CanvasComponent[],
  wires: CanvasWire[],
  namedJunctions: LevelJunction[] = []
): {
  components: ExpandedComponent[];
  wires: ExpandedWire[];
  junctions: ExpandedJunction[];
} {
  const outComponents: ExpandedComponent[] = [];
  const outWires: ExpandedWire[] = [];
  const junctionMap = new Map<string, ExpandedJunction>();
  // Map from original {componentId, pin} -> rewritten endpoint (junction or pin).
  const pinRewrite = new Map<string, ExpandedWire["start"]>();
  const componentsById = new Map(components.map((c) => [c.id, c]));

  function ensureJunction(id: string, x: number, y: number): void {
    if (!junctionMap.has(id)) junctionMap.set(id, { id, position: { x, y } });
  }
  function nodeJunctionId(name: string): string {
    return `port_${normalizeNodeName(name)}`;
  }

  // Pre-register named junctions (e.g. ``port_in``) so any wire whose point
  // endpoint sits on top of one resolves to that explicit name rather than an
  // auto-generated ``jp_x_y``. This is what makes the SUBCKT pipeline work —
  // the backend's flatten step looks for inner junctions named ``port_<pin>``.
  const namedByCoord = new Map<string, string>();
  for (const j of namedJunctions) {
    const sx = snap(j.x);
    const sy = snap(j.y);
    namedByCoord.set(`${sx}:${sy}`, j.id);
    ensureJunction(j.id, sx, sy);
  }

  function pinSlotKey(componentId: string, pin: string): string {
    return `${componentId}:${pin}`;
  }

  // 1. Pass through components verbatim. Transistor small-signal models are
  // backend-owned so AC parasitics and ro are stamped consistently.
  for (const c of components) {
    if (c.type === "LABEL") continue;
    if (c.type === "NODE") {
      const pos = getPinCoordinates(c, "n");
      const jid = nodeJunctionId(c.name || c.value || c.id);
      ensureJunction(jid, pos.x, pos.y);
      pinRewrite.set(pinSlotKey(c.id, "n"), { kind: "junction", junction_id: jid });
      continue;
    }
    outComponents.push(serializeBasicComponent(c));
  }

  // 2. Translate user wires; rewrite marker endpoints into named junctions,
  //    and free point endpoints into shared coordinate-keyed junctions.
  const pointJunctionByKey = new Map<string, string>();
  function pointJunctionId(x: number, y: number): string {
    const sx = snap(x);
    const sy = snap(y);
    const namedId = namedByCoord.get(`${sx}:${sy}`);
    if (namedId) return namedId;
    const key = pointKey(x, y);
    let id = pointJunctionByKey.get(key);
    if (!id) {
      id = `jp_${key.replace(/[^0-9a-z_]/gi, "_")}`;
      pointJunctionByKey.set(key, id);
      ensureJunction(id, x, y);
    }
    return id;
  }

  function translateEndpoint(ep: WireEndpoint): ExpandedWire["start"] | null {
    if (ep.kind === "point") {
      return { kind: "junction", junction_id: pointJunctionId(ep.x, ep.y) };
    }
    const rewrite = pinRewrite.get(`${ep.componentId}:${ep.pin}`);
    if (rewrite) return rewrite;
    const component = componentsById.get(ep.componentId);
    if (!component) return null;
    return { kind: "component_pin", component_id: ep.componentId, pin: resolveComponentPin(component, ep.pin) };
  }

  for (const w of wires) {
    const s = translateEndpoint(w.start);
    const e = translateEndpoint(w.end);
    if (!s || !e) continue;
    outWires.push({ id: w.id, start: s, end: e });
  }

  return {
    components: outComponents,
    wires: outWires,
    junctions: Array.from(junctionMap.values())
  };
}

function serializeBasicComponent(c: CanvasComponent): ExpandedComponent {
  return {
    id: c.id,
    type: c.type as ExpandedComponent["type"],
    name: c.name,
    value: c.type === "GND" ? null : c.value,
    subtype: c.type === "V" || c.type === "I" ? c.subtype ?? "DC" : null,
    value2: c.value2 ?? null,
    value3: c.value3 ?? null,
    ctrl_node1: c.ctrlNode1 ?? null,
    ctrl_node2: c.ctrlNode2 ?? null,
    ctrl_source: c.ctrlSource ?? null,
    subcircuit_id: c.subcircuitId ?? null,
    pins: c.pins ?? null,
    position: { x: c.x, y: c.y },
    metadata: c.metadata ?? {}
  };
}

export function buildSchematicPayload(
  components: CanvasComponent[],
  wires: CanvasWire[],
  analysis: AnalysisState,
  levels: SchematicLevel[] = [],
  rootJunctions: LevelJunction[] = []
): Record<string, unknown> {
  const analysisParams: Record<string, unknown> = {};
  if (analysis.mode === "tran" || analysis.mode === "dyn") {
    analysisParams.t_stop = analysis.tStop;
    analysisParams.t_step = analysis.tStep;
  } else if (analysis.mode === "ac") {
    analysisParams.f_start = analysis.fStart;
    analysisParams.f_stop = analysis.fStop;
    analysisParams.points = analysis.points;
  } else if (analysis.mode === "hb") {
    analysisParams.harmonics = analysis.harmonics;
    if (analysis.hbTimeWindow.trim()) {
      analysisParams.time_window = analysis.hbTimeWindow.trim();
    }
  }
  if (analysis.probeNodes.length > 0) {
    analysisParams.probe_nodes = analysis.probeNodes;
  }
  if (analysis.mor) {
    analysisParams.use_mor = true;
    analysisParams.mor_method = analysis.morMethod;
    analysisParams.mor_order =
      analysis.morOrderMode === "auto"
        ? "auto"
        : Math.max(1, Math.floor(Number(analysis.morOrder) || 1));
    analysisParams.mor_output_nodes = analysis.morOutputNodes;
    analysisParams.mor_validate = true;
  }

  // Pass the active level's own named junctions (if any) so SUBCKT instances
  // at the root level can also use port-style junction names if desired.
  const rootLevel = levels.find((l) => l.parentId === null);
  const expanded = expandSchematic(
    components,
    wires,
    rootJunctions.length > 0 ? rootJunctions : rootLevel?.junctions ?? []
  );

  const subcircuits: Record<string, unknown> = {};
  for (const level of levels) {
    if (level.parentId === null) continue;
    const sub = expandSchematic(level.components, level.wires, level.junctions ?? []);
    subcircuits[level.id] = {
      components: sub.components,
      wires: sub.wires,
      junctions: sub.junctions
    };
  }

  return {
    title: "Schematic",
    components: expanded.components,
    wires: expanded.wires,
    junctions: expanded.junctions,
    analysis: {
      mode: analysis.mode === "dyn" ? "dyn" : analysis.mode,
      params: analysisParams
    },
    subcircuits
  };
}

/* ---- Editor helpers shared by the canvas, menus, and keyboard handlers --- */

/** Local (un-rotated, un-mirrored) pin offset for a SUBCKT pin slot. */
export function subcktPinLocalOffset(pins: string[], pin: string): PinOffset {
  const index = Math.max(0, pins.indexOf(pin));
  const n = Math.max(1, pins.length);
  const side = Math.ceil(n / 4);
  const halfW = COMPONENT_BOX;
  const halfH = COMPONENT_BOX;
  const slot = index % side;
  const spacing = (COMPONENT_BOX * 2) / (side + 1);
  const ringIndex = Math.floor(index / side);
  if (ringIndex === 0) return { dx: -halfW, dy: -halfH + spacing * (slot + 1) };
  if (ringIndex === 1) return { dx: halfW, dy: -halfH + spacing * (slot + 1) };
  if (ringIndex === 2) return { dx: -halfW + spacing * (slot + 1), dy: -halfH };
  return { dx: -halfW + spacing * (slot + 1), dy: halfH };
}

function coordKey(x: number, y: number): string {
  return `${Math.round(x)}:${Math.round(y)}`;
}

/**
 * Connection census of the schematic: how many wire ends and pins meet at
 * every coordinate, plus the set of pins that have at least one wire.
 *
 * LTspice draws a solid junction dot wherever three or more things meet
 * (a T or a cross), and shows open squares on pins that are not wired. We
 * reproduce both from this census.
 */
export type ConnectionCensus = {
  /** Coordinates where >= 3 connections meet, in world units. */
  junctions: { x: number; y: number }[];
  /** ``componentId:pin`` keys that have at least one wire attached. */
  wiredPins: Set<string>;
};

export function computeConnectionCensus(
  components: CanvasComponent[],
  wires: CanvasWire[]
): ConnectionCensus {
  const componentsById = new Map(components.map((c) => [c.id, c]));
  const countAt = new Map<string, { x: number; y: number; n: number }>();
  const bump = (x: number, y: number) => {
    const key = coordKey(x, y);
    const entry = countAt.get(key);
    if (entry) entry.n += 1;
    else countAt.set(key, { x, y, n: 1 });
  };
  const wiredPins = new Set<string>();
  for (const w of wires) {
    for (const ep of [w.start, w.end]) {
      if (ep.kind === "pin") {
        wiredPins.add(`${ep.componentId}:${ep.pin}`);
        const c = componentsById.get(ep.componentId);
        if (!c) continue;
        const p = getPinCoordinates(c, ep.pin);
        bump(p.x, p.y);
      } else {
        bump(ep.x, ep.y);
      }
    }
  }
  // A pin itself counts as one connection at its coordinate, so a pin with
  // two wires forms a T (3 connections) and gets a dot.
  for (const c of components) {
    if (c.type === "NODE" || c.type === "LABEL") continue;
    for (const pin of getPinsForType(c.type, c)) {
      const p = getPinCoordinates(c, pin);
      const key = coordKey(p.x, p.y);
      if (countAt.has(key)) bump(p.x, p.y);
    }
  }
  const junctions: { x: number; y: number }[] = [];
  for (const entry of countAt.values()) {
    if (entry.n >= 3) junctions.push({ x: entry.x, y: entry.y });
  }
  return { junctions, wiredPins };
}

/** Axis-aligned bounds of every component and wire endpoint (world units). */
export function schematicBounds(
  components: CanvasComponent[],
  wires: CanvasWire[]
): { x: number; y: number; w: number; h: number } | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const grow = (x: number, y: number) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  for (const c of components) {
    const b = componentBounds(c);
    grow(b.x, b.y);
    grow(b.x + b.w, b.y + b.h + 24);
  }
  const byId = new Map(components.map((c) => [c.id, c]));
  for (const w of wires) {
    for (const ep of [w.start, w.end]) {
      const p = endpointXY(ep, byId);
      if (p) grow(p.x, p.y);
    }
  }
  if (!Number.isFinite(minX)) return null;
  return { x: minX, y: minY, w: Math.max(1, maxX - minX), h: Math.max(1, maxY - minY) };
}

/** Clipboard snapshot: selected components plus wires fully inside the selection. */
export type ClipboardData = {
  components: CanvasComponent[];
  wires: CanvasWire[];
};

export function extractClipboard(
  components: CanvasComponent[],
  wires: CanvasWire[],
  selectedIds: Set<string>
): ClipboardData | null {
  if (selectedIds.size === 0) return null;
  const comps = components.filter((c) => selectedIds.has(c.id));
  const compSet = new Set(comps.map((c) => c.id));
  const picked = wires.filter((w) => {
    if (selectedIds.has(w.id)) return true;
    const startOk = w.start.kind === "point" || compSet.has(w.start.componentId);
    const endOk = w.end.kind === "point" || compSet.has(w.end.componentId);
    return startOk && endOk;
  });
  if (comps.length === 0 && picked.length === 0) return null;
  return {
    components: comps.map((c) => ({
      ...c,
      pins: c.pins ? [...c.pins] : undefined,
      metadata: c.metadata ? { ...c.metadata } : undefined
    })),
    wires: picked.map((w) => ({
      ...w,
      start: w.start.kind === "pin" ? { ...w.start } : { ...w.start },
      end: w.end.kind === "pin" ? { ...w.end } : { ...w.end }
    }))
  };
}

/** Split ``wire`` at a point lying on one of its two orthogonal segments. */
export function splitWireAtPoint(
  wire: CanvasWire,
  point: { x: number; y: number },
  componentsById: Map<string, CanvasComponent>
): CanvasWire[] | null {
  const a = endpointXY(wire.start, componentsById);
  const b = endpointXY(wire.end, componentsById);
  if (!a || !b) return null;
  const sx = snap(point.x);
  const sy = snap(point.y);
  if ((sx === a.x && sy === a.y) || (sx === b.x && sy === b.y)) return null;
  const joint = pointEndpoint(sx, sy);
  const mk = (start: WireEndpoint, end: WireEndpoint): CanvasWire => ({
    id: `w-${Math.random().toString(36).slice(2, 8)}`,
    start,
    end
  });
  return [mk(wire.start, joint), mk(joint, wire.end)];
}

/* ---- Workspace file format (File > Save / Open) ------------------------ */

export const WORKSPACE_FILE_VERSION = 1;

export type WorkspaceFile = {
  format: "mna-simulator-workspace";
  version: number;
  savedAt: string;
  levels: SchematicLevel[];
  activeLevelId: string;
  analysis: AnalysisState;
  netlistMode: "schematic" | "netlist";
  netlistText: string;
};

export function serializeWorkspace(data: Omit<WorkspaceFile, "format" | "version" | "savedAt">): string {
  const file: WorkspaceFile = {
    format: "mna-simulator-workspace",
    version: WORKSPACE_FILE_VERSION,
    savedAt: new Date().toISOString(),
    ...data
  };
  return JSON.stringify(file, null, 2);
}

export function parseWorkspace(text: string): WorkspaceFile {
  const raw = JSON.parse(text) as Partial<WorkspaceFile>;
  if (!raw || raw.format !== "mna-simulator-workspace" || !Array.isArray(raw.levels)) {
    throw new Error("Not an MNA Simulator workspace file.");
  }
  for (const level of raw.levels) {
    if (!level || typeof level.id !== "string" || !Array.isArray(level.components) || !Array.isArray(level.wires)) {
      throw new Error("Workspace file has a malformed schematic level.");
    }
    level.pins = Array.isArray(level.pins) ? level.pins : [];
    level.parentId = typeof level.parentId === "string" ? level.parentId : null;
  }
  const rootExists = raw.levels.some((l) => l.parentId === null);
  if (!rootExists) throw new Error("Workspace file has no top-level schematic.");
  return {
    format: "mna-simulator-workspace",
    version: Number(raw.version ?? 1),
    savedAt: String(raw.savedAt ?? ""),
    levels: raw.levels,
    activeLevelId:
      typeof raw.activeLevelId === "string" && raw.levels.some((l) => l.id === raw.activeLevelId)
        ? raw.activeLevelId
        : raw.levels.find((l) => l.parentId === null)!.id,
    analysis: (raw.analysis ?? {}) as AnalysisState,
    netlistMode: raw.netlistMode === "netlist" ? "netlist" : "schematic",
    netlistText: typeof raw.netlistText === "string" ? raw.netlistText : ""
  };
}
