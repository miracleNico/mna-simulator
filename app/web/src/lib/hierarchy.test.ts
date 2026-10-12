import { describe, expect, it } from "vitest";

import { DEMO_MENU_GROUPS } from "./demoPresets";
import {
  HierarchyError,
  IdFactory,
  breadcrumbPath,
  decorateSubcktInstances,
  defaultPinSide,
  describePortSync,
  dissolveSubcircuit,
  duplicateEntity,
  entityContainsLevel,
  entityPinLayout,
  entityPorts,
  firstHierarchyIssue,
  instanceTemplate,
  makeSubcircuitFromSelection,
  parentLevelId,
  remapInstancePinEndpoints,
  syncEntityPorts,
  uniqueInstanceName,
  validateSubcktInstance
} from "./hierarchy";
import {
  AnalysisState,
  CanvasComponent,
  CanvasWire,
  SchematicLevel,
  buildSchematicPayload,
  componentBounds,
  getPinCoordinates,
  getPinsForType,
  parseWorkspace,
  subcktPinLocalOffset,
  WORKSPACE_FILE_VERSION
} from "./schematic";

/* ---- helpers ------------------------------------------------------------ */

const analysis: AnalysisState = {
  mode: "op",
  tStop: "2m",
  tStep: "10u",
  fStart: "1",
  fStop: "10000",
  points: 100,
  harmonics: 8,
  hbTimeWindow: "",
  dynSpeed: "1m",
  dynWindow: "5m",
  probeNodes: [],
  krylov: false,
  krylovRankMode: "auto",
  krylovRank: 80,
  krylovMethod: "auto",
  mor: false,
  morMethod: "auto",
  morOrderMode: "auto",
  morOrder: 40,
  morOutputNodes: [],
  numThreads: "auto"
};

function counterIds(): IdFactory {
  let n = 0;
  return (prefix) => `${prefix}-t${++n}`;
}

function part(
  id: string,
  type: CanvasComponent["type"],
  x: number,
  y: number,
  extra: Partial<CanvasComponent> = {}
): CanvasComponent {
  return { id, type, name: id.toUpperCase(), value: type === "GND" ? "0" : "1k", x, y, rotation: 0, ...extra };
}

function node(id: string, name: string, x: number, y: number, extra: Partial<CanvasComponent> = {}): CanvasComponent {
  return { id, type: "NODE", name, value: "node", x, y, rotation: 0, ...extra };
}

function pw(id: string, a: string, ap: string, b: string, bp: string): CanvasWire {
  return { id, start: { kind: "pin", componentId: a, pin: ap }, end: { kind: "pin", componentId: b, pin: bp } };
}

function root(components: CanvasComponent[], wires: CanvasWire[]): SchematicLevel {
  return { id: "root", title: "Top level", components, wires, pins: [], parentId: null };
}

type PayloadEndpoint =
  | { kind: "component_pin"; component_id: string; pin: string }
  | { kind: "junction"; junction_id: string };
type PayloadDoc = {
  components: Array<{ id: string; type: string; name: string; value: string | null; pins: string[] | null; subcircuit_id: string | null }>;
  wires: Array<{ id: string; start: PayloadEndpoint; end: PayloadEndpoint }>;
  junctions: Array<{ id: string }>;
  subcircuits?: Record<string, PayloadDoc>;
};

const PIN_NAMES: Record<string, string[]> = {
  R: ["p", "n"], C: ["p", "n"], L: ["p", "n"], V: ["p", "n"], I: ["p", "n"], D: ["p", "n"],
  VCVS: ["p", "n", "cp", "cn"], VCCS: ["p", "n", "cp", "cn"], CCCS: ["p", "n"], CCVS: ["p", "n"],
  QNPN: ["c", "b", "e"], QPNP: ["c", "b", "e"], NMOS: ["d", "g", "s"], PMOS: ["d", "g", "s"]
};

function segment(c: { id: string; name: string }): string {
  return (c.name || c.id).trim().replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "") || c.id;
}

/**
 * Flatten a payload exactly like ``schematic_pipeline._flatten_subcircuits``
 * and return its nets as a canonical, prefix-free description: one entry per
 * net listing the device terminals on it (by original component id), ground
 * nets marked with ``0|``. Two payloads with equal results simulate alike.
 */
function netPartition(payload: Record<string, unknown>): { nets: string[]; devices: string[] } {
  const subcircuits = (payload.subcircuits ?? {}) as Record<string, PayloadDoc>;
  const links: Array<[string, string]> = [];
  const grounds: string[] = [];
  const terminals: string[] = [];
  const devices: string[] = [];
  const visit = (doc: PayloadDoc, prefix: string) => {
    const subckts = new Map(doc.components.filter((c) => c.type === "SUBCKT").map((c) => [c.id, c]));
    const junction = (id: string) =>
      !prefix ? id : id.toLowerCase().startsWith("port_") ? `${prefix}${id.slice(5)}` : `${prefix}${id}`;
    const key = (ep: PayloadEndpoint) => {
      if (ep.kind === "junction") return `jn:${junction(ep.junction_id)}`;
      const sub = subckts.get(ep.component_id);
      if (sub) return `jn:${prefix}${segment(sub)}.${ep.pin}`;
      return `cp:${prefix}${ep.component_id}:${ep.pin}`;
    };
    for (const c of doc.components) {
      if (c.type === "SUBCKT") {
        expect(subcircuits[c.subcircuit_id ?? ""], `subcircuit of ${c.name}`).toBeDefined();
        visit(subcircuits[c.subcircuit_id!], `${prefix}${segment(c)}.`);
      } else if (c.type === "GND") {
        grounds.push(`cp:${prefix}${c.id}:g`);
      } else {
        devices.push(`${c.id}:${c.type}:${c.value}`);
        for (const pin of PIN_NAMES[c.type]) terminals.push(`cp:${prefix}${c.id}:${pin}`);
      }
    }
    for (const w of doc.wires) links.push([key(w.start), key(w.end)]);
  };
  visit(payload as unknown as PayloadDoc, "");
  const parent = new Map<string, string>();
  const find = (k: string): string => {
    if (!parent.has(k)) parent.set(k, k);
    const p = parent.get(k)!;
    if (p === k) return k;
    const r = find(p);
    parent.set(k, r);
    return r;
  };
  for (const [a, b] of links) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  }
  const groundRoots = new Set(grounds.map(find));
  const byRoot = new Map<string, string[]>();
  for (const t of terminals) {
    const r = find(t);
    const list = byRoot.get(r) ?? [];
    list.push(t.replace(/^cp:(?:[^:]*\.)?/, ""));
    byRoot.set(r, list);
  }
  const nets = Array.from(byRoot.entries()).map(([r, list]) => `${groundRoots.has(r) ? "0|" : ""}${list.sort().join(",")}`);
  return { nets: nets.sort(), devices: devices.sort() };
}

/** Same partition with component ids replaced by component names. */
function netsByName(partition: { nets: string[] }, components: CanvasComponent[]): string[] {
  const nameOf = new Map(components.map((c) => [c.id, c.name]));
  return partition.nets
    .map((net) => {
      const ground = net.startsWith("0|");
      const terminals = (ground ? net.slice(2) : net).split(",").map((t) => {
        const cut = t.lastIndexOf(":");
        return `${nameOf.get(t.slice(0, cut)) ?? t.slice(0, cut)}:${t.slice(cut + 1)}`;
      });
      return `${ground ? "0|" : ""}${terminals.sort().join(",")}`;
    })
    .sort();
}

function payloadOf(levels: SchematicLevel[], mode: AnalysisState["mode"] = "op", overrides: Partial<AnalysisState> = {}) {
  const top = levels.find((l) => l.parentId === null)!;
  return buildSchematicPayload(top.components, top.wires, { ...analysis, mode, ...overrides }, levels, top.junctions ?? []);
}

/** Custom circuit exercising a shared wire corner, a crossing wire, a label used on both sides,
 *  a ground connection that crosses the boundary, and an internal net label. */
function mixedCircuit(): SchematicLevel {
  const components: CanvasComponent[] = [
    part("v1", "V", 100, 200, { value: "5", subtype: "DC" }),
    part("g1", "GND", 100, 300),
    part("r1", "R", 200, 120, { value: "1k" }),
    part("r2", "R", 320, 120, { value: "2k" }),
    part("r3", "R", 400, 200, { value: "3k", rotation: 90 }),
    part("g2", "GND", 400, 300),
    part("c1", "C", 520, 120, { value: "1u" }),
    part("g3", "GND", 600, 200),
    node("tap-in", "tap", 400, 100),
    node("tap-out", "tap", 700, 100),
    part("r4", "R", 700, 200, { value: "4k", rotation: 90 }),
    part("g4", "GND", 700, 300),
    part("r5", "R", 320, 260, { value: "5k" }),
    node("inner", "inner", 240, 260)
  ];
  const wires: CanvasWire[] = [
    pw("w1", "v1", "n", "g1", "g"),
    pw("w2", "v1", "p", "r1", "p"),
    { id: "w3", start: { kind: "pin", componentId: "r1", pin: "n" }, end: { kind: "point", x: 260, y: 120 } },
    { id: "w4", start: { kind: "point", x: 260, y: 120 }, end: { kind: "pin", componentId: "r2", pin: "p" } },
    pw("w5", "r2", "n", "r3", "p"),
    pw("w6", "r3", "p", "tap-in", "n"),
    pw("w7", "r3", "n", "g2", "g"),
    pw("w8", "r2", "n", "c1", "p"),
    pw("w9", "c1", "n", "g3", "g"),
    pw("w10", "tap-out", "n", "r4", "p"),
    pw("w11", "r4", "n", "g4", "g"),
    pw("w12", "inner", "n", "r5", "p"),
    pw("w13", "r5", "n", "r3", "p")
  ];
  return root(components, wires);
}

const MIXED_SELECTION = ["r2", "r3", "r5", "tap-in", "inner", "w4"];
const CE_SELECTION = ["q-1", "r-c", "r-e", "r-btop", "r-bbot", "node-base", "node-vcc-rc", "node-vcc-bias", "gnd-bias", "gnd-e"];

function ceLevels(): SchematicLevel[] {
  const preset = DEMO_MENU_GROUPS.flatMap((g) => g.presets).find((p) => p.id === "ce-amplifier")!;
  return [root(preset.components.map((c) => ({ ...c })), preset.wires.map((w) => ({ ...w })))];
}

function entity(id: string, components: CanvasComponent[], wires: CanvasWire[] = [], parentId = "root"): SchematicLevel {
  return { id, title: id.toUpperCase(), components, wires, pins: [], parentId };
}

/* ---- ports and pin layout ------------------------------------------------ */

describe("entity ports", () => {
  it("lists NODE marker names in component order plus explicit port junctions", () => {
    const e = entity("e", [node("a", "in", 0, 0), part("r", "R", 50, 0), node("b", "out", 100, 0), node("c", "in", 0, 40)]);
    e.junctions = [{ id: "port_aux", x: 0, y: 80 }, { id: "j_free", x: 0, y: 90 }];
    expect(entityPorts(e)).toEqual(["in", "out", "aux"]);
  });

  it("puts in* names on the left by default and everything else on the right", () => {
    expect(defaultPinSide("in")).toBe("left");
    expect(defaultPinSide("INN")).toBe("left");
    expect(defaultPinSide("vin")).toBe("right");
    expect(defaultPinSide("out")).toBe("right");
  });

  it("orders pins by marker coordinate on the side the marker asks for", () => {
    const e = entity("e", [
      node("m1", "in_b", 0, 200),
      node("m2", "in_a", 0, 100),
      node("m3", "vcc", 50, 0, { metadata: { side: "top" } }),
      node("m4", "gnd_ref", 20, 300, { metadata: { side: "bottom" } }),
      node("m5", "aux", 10, 300, { metadata: { side: "bottom" } }),
      node("m6", "out", 200, 150)
    ]);
    const layout = entityPinLayout(e, ["out", "in_b", "in_a", "vcc", "gnd_ref", "aux", "extra"]);
    expect(layout).toEqual({
      left: ["in_a", "in_b"],
      right: ["out", "extra"],
      top: ["vcc"],
      bottom: ["aux", "gnd_ref"]
    });
  });

  it("places pins on the drawn body edge and keeps them there under rotation and mirroring", () => {
    const layout = { left: ["a", "b"], right: ["c"], top: [], bottom: [] };
    expect(subcktPinLocalOffset(["a", "b", "c"], "a", layout)).toEqual({ dx: -70, dy: -10 });
    expect(subcktPinLocalOffset(["a", "b", "c"], "b", layout)).toEqual({ dx: -70, dy: 10 });
    expect(subcktPinLocalOffset(["a", "b", "c"], "c", layout)).toEqual({ dx: 70, dy: 0 });
    const inst: CanvasComponent = { ...part("x", "SUBCKT", 300, 200), pins: ["a", "b", "c"], pinLayout: layout };
    expect(getPinCoordinates(inst, "c")).toEqual({ x: 370, y: 200 });
    for (const rotation of [0, 90, 180, 270] as const) {
      for (const mirrored of [false, true]) {
        const c = { ...inst, rotation, mirrored };
        const b = componentBounds(c);
        for (const pin of c.pins!) {
          const p = getPinCoordinates(c, pin);
          const onEdge =
            ((Math.abs(p.x - b.x) < 1e-9 || Math.abs(p.x - (b.x + b.w)) < 1e-9) && p.y >= b.y && p.y <= b.y + b.h) ||
            ((Math.abs(p.y - b.y) < 1e-9 || Math.abs(p.y - (b.y + b.h)) < 1e-9) && p.x >= b.x && p.x <= b.x + b.w);
          expect(onEdge, `${pin} at rotation ${rotation} mirrored ${mirrored}`).toBe(true);
        }
      }
    }
  });

  it("grows the body with the longest side", () => {
    const pins = ["in1", "in2", "in3", "in4", "in5", "in6"];
    const inst: CanvasComponent = { ...part("x", "SUBCKT", 0, 0), pins };
    const b = componentBounds(inst);
    expect(b.h).toBe(140);
    expect(getPinCoordinates(inst, "in1")).toEqual({ x: -70, y: -50 });
    expect(getPinCoordinates(inst, "in6")).toEqual({ x: -70, y: 50 });
  });

  it("decorates linked instances with the entity layout only", () => {
    const e = entity("e", [node("m", "out", 0, 0, { metadata: { side: "top" } })]);
    const linked: CanvasComponent = { ...part("x1", "SUBCKT", 0, 0), subcircuitId: "e", pins: ["out"] };
    const unlinked: CanvasComponent = { ...part("x2", "SUBCKT", 0, 0), subcircuitId: "", pins: ["out"] };
    const [a, b] = decorateSubcktInstances([linked, unlinked], [root([], []), e]);
    expect(a.pinLayout?.top).toEqual(["out"]);
    expect(b).toBe(unlinked);
  });
});

/* ---- navigation, recursion and validation -------------------------------- */

describe("navigation and validation", () => {
  const levels: SchematicLevel[] = [
    root([{ ...part("x1", "SUBCKT", 0, 0), name: "X1", subcircuitId: "a", pins: ["in"] }], []),
    entity("a", [node("ai", "in", 0, 0), { ...part("x2", "SUBCKT", 0, 0), name: "X2", subcircuitId: "b", pins: ["p"] }]),
    entity("b", [node("bp", "p", 0, 0)], [], "a")
  ];

  it("builds the breadcrumb from the tree", () => {
    expect(breadcrumbPath(levels, "b").map((l) => l.id)).toEqual(["root", "a", "b"]);
    expect(breadcrumbPath(levels, "root").map((l) => l.id)).toEqual(["root"]);
    expect(parentLevelId(levels, "b")).toBe("a");
    expect(parentLevelId(levels, "root")).toBeNull();
  });

  it("detects entities that contain a level directly or through nesting", () => {
    expect(entityContainsLevel(levels, "a", "b")).toBe(true);
    expect(entityContainsLevel(levels, "b", "a")).toBe(false);
    expect(entityContainsLevel(levels, "a", "a")).toBe(true);
  });

  it("reports unlinked, recursive and mismatched instances like the backend", () => {
    const unlinked = { ...part("u", "SUBCKT", 0, 0), name: "X9", subcircuitId: "", pins: ["a"] };
    expect(validateSubcktInstance(levels, "root", unlinked)).toMatch(/^SUBCKT instance 'X9' is not linked/);
    const recursive = { ...part("r", "SUBCKT", 0, 0), name: "X3", subcircuitId: "a", pins: ["in"] };
    expect(validateSubcktInstance(levels, "b", recursive)).toMatch(/^SUBCKT instance 'X3' is recursive/);
    const mismatch = { ...part("m", "SUBCKT", 0, 0), name: "X1", subcircuitId: "a", pins: ["in", "out"] };
    expect(validateSubcktInstance(levels, "root", mismatch)).toBe(
      "SUBCKT instance 'X1' port mismatch: missing child ports X1.out (local junctions: port_out)"
    );
    const extra = { ...part("m", "SUBCKT", 0, 0), name: "X1", subcircuitId: "a", pins: [] as string[] };
    expect(validateSubcktInstance(levels, "root", extra)).toMatch(/requires pins/);
    expect(validateSubcktInstance(levels, "root", levels[0].components[0])).toBeNull();
  });

  it("finds the first problem along logical paths for Run", () => {
    expect(firstHierarchyIssue(levels, "root")).toBeNull();
    const broken = levels.map((l) =>
      l.id === "a" ? { ...l, components: l.components.map((c) => (c.id === "x2" ? { ...c, pins: ["q"] } : c)) } : l
    );
    expect(firstHierarchyIssue(broken, "root")).toBe(
      "SUBCKT instance 'X2' port mismatch: missing child ports X1.X2.q (local junctions: port_q); unexpected child ports X1.X2.p"
    );
    const dup = [
      { ...levels[0], components: [...levels[0].components, { ...levels[0].components[0], id: "x1b" }] },
      ...levels.slice(1)
    ];
    expect(firstHierarchyIssue(dup, "root")).toBe("Duplicate SUBCKT logical name 'X1' under 'root'.");
    const cyclic = levels.map((l) =>
      l.id === "b" ? { ...l, components: [...l.components, { ...part("x5", "SUBCKT", 0, 0), name: "X5", subcircuitId: "a", pins: ["in"] }] } : l
    );
    expect(firstHierarchyIssue(cyclic, "root")).toMatch(/recursive/);
  });

  it("gives a template that places a linked instance", () => {
    expect(instanceTemplate(levels, "a")).toEqual({ subcircuitId: "a", pins: ["in"] });
    expect(instanceTemplate(levels, "root")).toBeNull();
    expect(uniqueInstanceName(levels[0].components)).toBe("X2");
  });
});

/* ---- port edits propagate to instances ------------------------------------ */

describe("syncEntityPorts", () => {
  function base(): SchematicLevel[] {
    return [
      root(
        [
          { ...part("x1", "SUBCKT", 300, 200), name: "X1", subcircuitId: "e", pins: ["in", "out"] },
          part("r1", "R", 100, 200),
          part("r2", "R", 500, 200)
        ],
        [pw("wa", "r1", "n", "x1", "in"), pw("wb", "x1", "out", "r2", "p")]
      ),
      entity("e", [node("mi", "in", 0, 0), node("mo", "out", 200, 0), part("rr", "R", 100, 0)])
    ];
  }

  it("follows a rename and keeps the wire on the renamed pin", () => {
    const before = base();
    const after = before.map((l) =>
      l.id === "e" ? { ...l, components: l.components.map((c) => (c.id === "mo" ? { ...c, name: "vout" } : c)) } : l
    );
    const { levels, report } = syncEntityPorts(before, after, "e");
    expect(report?.renamed).toEqual([["out", "vout"]]);
    const top = levels[0];
    expect(top.components[0].pins).toEqual(["in", "vout"]);
    expect(top.wires[1].start).toEqual({ kind: "pin", componentId: "x1", pin: "vout" });
    expect(levels[1].pins).toEqual(["in", "vout"]);
  });

  it("adds new ports and detaches wires of deleted ones", () => {
    const before = base();
    const after = before.map((l) =>
      l.id === "e" ? { ...l, components: [...l.components.filter((c) => c.id !== "mo"), node("mx", "aux", 0, 80)] } : l
    );
    const { levels, report } = syncEntityPorts(before, after, "e");
    expect(report).toMatchObject({ added: ["aux"], removed: ["out"], detached: 1, instances: 1 });
    const top = levels[0];
    expect(top.components[0].pins).toEqual(["in", "aux"]);
    const detached = top.wires.find((w) => w.id === "wb")!;
    expect(detached.start.kind).toBe("point");
    expect(validateSubcktInstance(levels, "root", top.components[0])).toBeNull();
  });

  it("does nothing when the port set is unchanged (moving or re-siding a marker)", () => {
    const before = base();
    const after = before.map((l) =>
      l.id === "e" ? { ...l, components: l.components.map((c) => (c.id === "mo" ? { ...c, y: 90, metadata: { side: "top" } } : c)) } : l
    );
    const result = syncEntityPorts(before, after, "e");
    expect(result.report).toBeNull();
    expect(result.levels).toBe(after);
  });

  it("remaps pins by name first, then by position", () => {
    const wires = [pw("w", "r", "p", "x", "a"), pw("v", "r", "n", "x", "b")];
    expect(remapInstancePinEndpoints(wires, "x", ["a", "b"], ["b", "a"]).map((w) => (w.end as { pin: string }).pin)).toEqual(["a", "b"]);
    expect(remapInstancePinEndpoints(wires, "x", ["a", "b"], ["z", "b"]).map((w) => (w.end as { pin: string }).pin)).toEqual(["z", "b"]);
    const detached = remapInstancePinEndpoints(wires, "x", ["a", "b"], ["a"], () => ({ x: 12, y: 34 }));
    expect(detached[0].end).toEqual({ kind: "pin", componentId: "x", pin: "a" });
    expect(detached[1].end).toEqual({ kind: "point", x: 10, y: 30 });
  });

  it("describes the change for the status line", () => {
    expect(
      describePortSync({ entityId: "e", renamed: [["out", "vout"]], added: ["aux"], removed: ["p2"], detached: 2, instances: 3 }, "Amp")
    ).toBe("Ports of 'Amp': renamed out → vout; added aux; removed p2 · 3 instances updated · 2 wires detached from deleted pin.");
  });
});

/* ---- make subcircuit ------------------------------------------------------ */

describe("makeSubcircuitFromSelection", () => {
  it("keeps every net of the mixed circuit and builds the expected ports", () => {
    const levels = [mixedCircuit()];
    const before = netPartition(payloadOf(levels));
    const result = makeSubcircuitFromSelection(levels, "root", MIXED_SELECTION, { newId: counterIds() });
    const after = netPartition(payloadOf(result.levels));
    expect(after).toEqual(before);

    expect(result.ports).toEqual(["p1", "tap", "inner"]);
    expect(result.internalPorts).toEqual(["inner"]);
    const child = result.levels.find((l) => l.id === result.newLevelId)!;
    expect(child.parentId).toBe("root");
    expect(child.title).toBe("Subckt 1");
    expect(child.pins).toEqual(["p1", "tap", "inner"]);
    const top = result.levels.find((l) => l.id === "root")!;
    const inst = top.components.find((c) => c.id === result.instanceId)!;
    expect(inst).toMatchObject({ type: "SUBCKT", name: "X1", subcircuitId: result.newLevelId, pins: ["p1", "tap", "inner"] });
    expect(top.components.some((c) => MIXED_SELECTION.includes(c.id))).toBe(false);
    // The edge markers carry the side their outside stubs came from.
    const markers = child.components.filter((c) => c.type === "NODE" && c.metadata?.side);
    expect(markers.map((m) => [m.name, m.metadata?.side])).toEqual([["p1", "left"], ["tap", "right"]]);
    expect(validateSubcktInstance(result.levels, "root", inst)).toBeNull();
    expect(firstHierarchyIssue(result.levels, "root")).toBeNull();
    // Every device pin in the child is wired (the backend refuses open pins).
    for (const c of child.components) {
      if (c.type === "NODE" || c.type === "LABEL") continue;
      for (const pin of getPinsForType(c.type, c)) {
        const wired = child.wires.some((w) =>
          [w.start, w.end].some((ep) => ep.kind === "pin" && ep.componentId === c.id && ep.pin === pin)
        );
        expect(wired, `${c.id}.${pin}`).toBe(true);
      }
    }
  });

  it("keeps the CE amplifier netlist identical and the net labels on the parent", () => {
    const levels = ceLevels();
    const before = netPartition(payloadOf(levels));
    const result = makeSubcircuitFromSelection(levels, "root", CE_SELECTION, { newId: counterIds() });
    expect(netPartition(payloadOf(result.levels))).toEqual(before);
    expect(result.ports).toEqual(["vcc", "base", "p1"]);
    const top = result.levels[0];
    const labels = top.components.filter((c) => c.type === "NODE").map((c) => c.name).sort();
    expect(labels).toEqual(["base", "out", "vcc", "vcc", "vin"]);
    const child = result.levels[1];
    const layout = entityPinLayout(child, result.ports);
    expect(layout.left).toEqual(["vcc", "base"]);
    expect(layout.right).toEqual(["p1"]);
  });

  it("supports nesting and moves the nested entity under the new one", () => {
    const levels = [mixedCircuit()];
    const ids = counterIds();
    const first = makeSubcircuitFromSelection(levels, "root", MIXED_SELECTION, { newId: ids });
    const before = netPartition(payloadOf(first.levels));
    const second = makeSubcircuitFromSelection(first.levels, "root", [first.instanceId, "c1", "g3"], { newId: ids });
    expect(netPartition(payloadOf(second.levels))).toEqual(before);
    expect(second.levels.find((l) => l.id === first.newLevelId)!.parentId).toBe(second.newLevelId);
    expect(breadcrumbPath(second.levels, first.newLevelId).map((l) => l.title)).toEqual(["Top level", "Subckt 2", "Subckt 1"]);
    expect(firstHierarchyIssue(second.levels, "root")).toBeNull();
  });

  it("keeps the ports of an entity when making a subcircuit inside it", () => {
    const ids = counterIds();
    const made = makeSubcircuitFromSelection(ceLevels(), "root", CE_SELECTION, { newId: ids });
    const before = netPartition(payloadOf(made.levels));
    const entityBefore = made.levels.find((l) => l.id === made.newLevelId)!;
    // The p1 port marker sits on the collector net, which lies entirely inside the new selection.
    const p1 = entityBefore.components.find((c) => c.type === "NODE" && c.name === "p1")!;
    const nested = makeSubcircuitFromSelection(made.levels, made.newLevelId, [p1.id, "q-1", "r-c", "r-e", "gnd-e"], { newId: ids });
    const entityAfter = nested.levels.find((l) => l.id === made.newLevelId)!;
    expect(new Set(entityPorts(entityAfter))).toEqual(new Set(entityPorts(entityBefore)));
    expect(nested.internalPorts).toEqual(["p1"]);
    expect(netPartition(payloadOf(nested.levels))).toEqual(before);
    expect(firstHierarchyIssue(nested.levels, "root")).toBeNull();
    expect(breadcrumbPath(nested.levels, nested.newLevelId).map((l) => l.title)).toEqual(["Top level", "Subckt 1", "Subckt 2"]);
  });

  it("refuses selections without parts or without any connection", () => {
    const levels = [mixedCircuit()];
    expect(() => makeSubcircuitFromSelection(levels, "root", ["tap-in"])).toThrow(HierarchyError);
    expect(() => makeSubcircuitFromSelection(levels, "root", [])).toThrow(/Select at least one part/);
    const all = levels[0].components.map((c) => c.id).filter((id) => id !== "inner" && id !== "tap-in" && id !== "tap-out");
    const isolated = [root(levels[0].components.filter((c) => all.includes(c.id)), levels[0].wires.filter((w) => !["w6", "w10", "w12"].includes(w.id)))];
    expect(() => makeSubcircuitFromSelection(isolated, "root", all)).toThrow(/no connection/);
  });

  it("round-trips through dissolve without changing the nets", () => {
    const levels = [mixedCircuit()];
    const ids = counterIds();
    const before = netPartition(payloadOf(levels));
    const made = makeSubcircuitFromSelection(levels, "root", MIXED_SELECTION, { newId: ids });
    const dissolved = dissolveSubcircuit(made.levels, "root", made.instanceId, { newId: ids });
    const top = dissolved.levels[0];
    expect(top.components.some((c) => c.type === "SUBCKT")).toBe(false);
    // Dissolved copies get new ids but keep their names: compare by name.
    expect(netsByName(netPartition(payloadOf(dissolved.levels)), top.components)).toEqual(
      netsByName(before, levels[0].components)
    );
  });
});

describe("duplicateEntity", () => {
  it("copies an entity subtree with new ids and relinks nested instances", () => {
    const levels = [mixedCircuit()];
    const ids = counterIds();
    const first = makeSubcircuitFromSelection(levels, "root", MIXED_SELECTION, { newId: ids });
    const second = makeSubcircuitFromSelection(first.levels, "root", [first.instanceId, "c1", "g3"], { newId: ids });
    const { levels: next, newLevelId } = duplicateEntity(second.levels, second.newLevelId, { newId: ids });
    const copy = next.find((l) => l.id === newLevelId)!;
    const original = next.find((l) => l.id === second.newLevelId)!;
    expect(copy.title).toBe("Subckt 2 copy");
    expect(copy.parentId).toBe("root");
    expect(copy.components.map((c) => c.id)).not.toEqual(original.components.map((c) => c.id));
    const nested = copy.components.find((c) => c.type === "SUBCKT")!;
    const nestedEntity = next.find((l) => l.id === nested.subcircuitId)!;
    expect(nestedEntity.id).not.toBe(first.newLevelId);
    expect(nestedEntity.parentId).toBe(newLevelId);
    expect(nestedEntity.title).toBe("Subckt 1 copy");
    expect(entityPorts(copy)).toEqual(entityPorts(original));
    for (const w of copy.wires) {
      for (const ep of [w.start, w.end]) {
        if (ep.kind === "pin") expect(copy.components.some((c) => c.id === ep.componentId)).toBe(true);
      }
    }
  });
});

describe("demo presets", () => {
  it("load without hierarchy problems and with pins on the intended sides", () => {
    for (const preset of DEMO_MENU_GROUPS.flatMap((g) => g.presets)) {
      const levels: SchematicLevel[] = [root(preset.components, preset.wires), ...(preset.extraLevels ?? [])];
      expect(firstHierarchyIssue(levels, "root"), preset.id).toBeNull();
    }
    const opamp = DEMO_MENU_GROUPS.flatMap((g) => g.presets).find((p) => p.id === "three-stage-opamp")!;
    const diff = opamp.extraLevels!.find((l) => l.id === "lvl-diff")!;
    expect(entityPinLayout(diff, ["in_p", "out", "vcc", "in_n"])).toEqual({
      left: ["in_p"],
      right: ["out"],
      top: ["vcc"],
      bottom: ["in_n"]
    });
  });
});

describe("workspace compatibility", () => {
  it("still reads version-1 files unchanged", () => {
    const v1 = {
      format: "mna-simulator-workspace",
      version: 1,
      savedAt: "2026-01-01T00:00:00Z",
      levels: [
        root([{ ...part("x1", "SUBCKT", 0, 0), subcircuitId: "e", pins: ["a", "b"] }], []),
        entity("e", [node("n1", "a", 0, 0), node("n2", "b", 100, 0)])
      ],
      activeLevelId: "root",
      analysis,
      netlistMode: "schematic",
      netlistText: ""
    };
    const parsed = parseWorkspace(JSON.stringify(v1));
    expect(parsed.levels).toEqual(v1.levels);
    expect(WORKSPACE_FILE_VERSION).toBe(1);
  });
});

/* ---- fixture for the Python pipeline test --------------------------------- */

describe("pipeline fixture", () => {
  it("matches tests/fixtures/make_subcircuit_payloads.json (run vitest -u to refresh)", async () => {
    const mixed = [mixedCircuit()];
    const mixedMade = makeSubcircuitFromSelection(mixed, "root", MIXED_SELECTION, { newId: counterIds() });
    const ce = ceLevels();
    const ceMade = makeSubcircuitFromSelection(ce, "root", CE_SELECTION, { newId: counterIds() });
    const tran: Partial<AnalysisState> = { tStop: "2m", tStep: "10u" };
    const fixture = {
      description:
        "Schematic payloads before and after makeSubcircuitFromSelection (app/web/src/lib/hierarchy.test.ts). " +
        "tests/test_schematic_pipeline.py checks that each pair simulates alike.",
      cases: [
        {
          name: "mixed",
          selection: MIXED_SELECTION,
          instance: "X1",
          before: payloadOf(mixed, "op"),
          after: payloadOf(mixedMade.levels, "op")
        },
        {
          name: "ce_amplifier",
          selection: CE_SELECTION,
          instance: "X1",
          before: payloadOf(ce, "tran", tran),
          after: payloadOf(ceMade.levels, "tran", tran)
        }
      ]
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      "../../../../tests/fixtures/make_subcircuit_payloads.json"
    );
  });
});
