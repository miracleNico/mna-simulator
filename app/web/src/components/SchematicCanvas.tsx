import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState
} from "react";
import {
  CanvasComponent,
  CanvasWire,
  DeviceType,
  GRID,
  LevelJunction,
  Rotation,
  WireEndpoint,
  componentBounds,
  computeConnectionCensus,
  computeNetAssignments,
  createDefaultComponent,
  endpointXY,
  getPinCoordinates,
  getPinsForType,
  pinEndpoint,
  pointEndpoint,
  schematicBounds,
  snap,
  splitWireAtPoint,
  wireKey
} from "../lib/schematic";
import { DEFAULT_THEME, drawComponent } from "../lib/symbols";

export type ProbeTarget = { componentId: string; pin: string; netLabel?: string; x: number; y: number };

export type SchematicCanvasHandle = {
  zoomIn: () => void;
  zoomOut: () => void;
  zoomToFit: () => void;
  resetZoom: () => void;
  focus: () => void;
};

type Props = {
  components: CanvasComponent[];
  wires: CanvasWire[];
  /** Named junctions for the active level (e.g. port_in, port_out for SUBCKT use). */
  junctions?: LevelJunction[];
  selectedIds: Set<string>;
  /** When non-null the canvas is in placement mode and drops this device type on click. */
  pendingDevice: DeviceType | null;
  /** True to capture a pin click as a dyn-probe target instead of wiring. */
  probeMode: boolean;
  /** LTspice-style wire mode: click anywhere to start a wire, click to add corners. */
  wireTool: boolean;
  showGrid: boolean;
  showNetNames: boolean;
  onSetComponents: (updater: (current: CanvasComponent[]) => CanvasComponent[]) => void;
  onSetWires: (updater: (current: CanvasWire[]) => CanvasWire[]) => void;
  onSelect: (ids: Set<string>) => void;
  /** Placement finished (Esc or right-click). */
  onPendingResolved: () => void;
  /** Wire tool finished (Esc or right-click while idle). */
  onWireToolResolved: () => void;
  onBeforeEdit: () => void;
  onProbePick: (target: ProbeTarget) => void;
  onCopy: () => void;
  onPaste: () => void;
  onDuplicate: () => void;
  onOpenProperties: (componentId: string) => void;
  onZoomChange?: (scale: number) => void;
  onCursorChange?: (pos: { x: number; y: number } | null) => void;
  onStatus?: (message: string) => void;
};

type PanState = { x: number; y: number; scale: number };

type InteractionState =
  | { kind: "idle" }
  | { kind: "panning"; startX: number; startY: number; origin: PanState; moved: boolean }
  | {
      kind: "moving";
      ids: string[];
      anchorWX: number;
      anchorWY: number;
      origin: Map<string, { x: number; y: number }>;
      wireOrigin: Map<string, { start?: { x: number; y: number }; end?: { x: number; y: number } }>;
    }
  | { kind: "wiring"; from: WireEndpoint; cursorX: number; cursorY: number }
  | { kind: "marquee"; startWX: number; startWY: number; cursorWX: number; cursorWY: number };

type Rect = { x: number; y: number; w: number; h: number };
type Segment = [number, number, number, number];

const PIN_HIT = 10;
const FREE_HIT = 9;
const WIRE_HIT = 5;
const MIN_SCALE = 0.1;
const MAX_SCALE = 6;
const AUTO_NET = /^n\d+$/;

export const SchematicCanvas = forwardRef<SchematicCanvasHandle, Props>(function SchematicCanvas(props, ref) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [pan, setPan] = useState<PanState>({ x: 40, y: 40, scale: 1 });
  const [interaction, setInteraction] = useState<InteractionState>({ kind: "idle" });
  const [hoverPin, setHoverPin] = useState<{ componentId: string; pin: string } | null>(null);
  const [hoverFreePoint, setHoverFreePoint] = useState<{ wireId: string; end: "start" | "end"; x: number; y: number } | null>(null);
  const [hoverWireId, setHoverWireId] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number; wx: number; wy: number } | null>(null);
  const [pendingRotation, setPendingRotation] = useState<Rotation>(0);
  const [pendingMirror, setPendingMirror] = useState(false);
  const moveHistorySavedRef = useRef(false);
  const panRef = useRef(pan);
  panRef.current = pan;

  const componentsById = useMemo(() => {
    const map = new Map<string, CanvasComponent>();
    props.components.forEach((c) => map.set(c.id, c));
    return map;
  }, [props.components]);

  // Net names mirror the backend union-find so the canvas shows the same
  // labels the simulator will use. Memoised: it only changes with topology.
  const nets = useMemo(
    () => computeNetAssignments(props.components, props.wires),
    [props.components, props.wires]
  );
  const census = useMemo(
    () => computeConnectionCensus(props.components, props.wires),
    [props.components, props.wires]
  );
  const obstacleBounds = useMemo(() => {
    const map = new Map<string, Rect>();
    for (const c of props.components) map.set(c.id, componentBounds(c));
    return map;
  }, [props.components]);

  useEffect(() => {
    setPendingRotation(0);
    setPendingMirror(false);
  }, [props.pendingDevice]);

  useEffect(() => {
    props.onZoomChange?.(pan.scale);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pan.scale]);

  const netLabelFor = (cid: string, pin: string): string | undefined => {
    const c = componentsById.get(cid);
    if (!c) return undefined;
    return `${c.name}.${pin}`;
  };

  const netNameForPin = (cid: string, pin: string): string | undefined => {
    const name = nets.byKey.get(`cp:${cid}:${pin}`);
    return name && name !== "?" ? name : undefined;
  };

  /* ---------------- Geometry helpers ---------------- */

  function clientToWorld(clientX: number, clientY: number) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const p = panRef.current;
    return { sx, sy, wx: (sx - p.x) / p.scale, wy: (sy - p.y) / p.scale };
  }

  function findPinAt(wx: number, wy: number): { componentId: string; pin: string } | null {
    let best: { ref: { componentId: string; pin: string }; d: number } | null = null;
    const tolerance = PIN_HIT / Math.min(1, Math.max(0.5, panRef.current.scale)) + 2;
    for (const c of props.components) {
      for (const pin of getPinsForType(c.type, c)) {
        const p = getPinCoordinates(c, pin);
        const d = Math.hypot(p.x - wx, p.y - wy);
        if (d < tolerance && (!best || d < best.d)) best = { ref: { componentId: c.id, pin }, d };
      }
    }
    return best?.ref ?? null;
  }

  function findFreeEndpointAt(wx: number, wy: number) {
    for (const w of props.wires) {
      for (const which of ["start", "end"] as const) {
        const ep = which === "start" ? w.start : w.end;
        if (ep.kind !== "point") continue;
        if (Math.hypot(ep.x - wx, ep.y - wy) < FREE_HIT) return { wireId: w.id, end: which, x: ep.x, y: ep.y };
      }
    }
    return null;
  }

  function findComponentAt(wx: number, wy: number): CanvasComponent | null {
    for (let i = props.components.length - 1; i >= 0; i--) {
      const c = props.components[i];
      const b = componentBounds(c);
      if (wx >= b.x && wx <= b.x + b.w && wy >= b.y && wy <= b.y + b.h) return c;
    }
    return null;
  }

  /** Route a wire exactly as it is drawn so hit-testing matches the picture. */
  function wireSegments(wire: CanvasWire): Segment[] | null {
    const a = endpointXY(wire.start, componentsById);
    const b = endpointXY(wire.end, componentsById);
    if (!a || !b) return null;
    const startOwner = wire.start.kind === "pin" ? wire.start.componentId : null;
    const endOwner = wire.end.kind === "pin" ? wire.end.componentId : null;
    const obstacles: Rect[] = [];
    for (const [cid, bounds] of obstacleBounds) {
      if (cid !== startOwner && cid !== endOwner) obstacles.push(bounds);
    }
    return routeOrthogonal(a.x, a.y, b.x, b.y, obstacles);
  }

  /** Locate a wire whose body lies under the cursor, plus the snapped point on it. */
  function findWireAt(wx: number, wy: number): { wire: CanvasWire; x: number; y: number } | null {
    const tol = WIRE_HIT / Math.max(0.5, panRef.current.scale);
    for (let i = props.wires.length - 1; i >= 0; i--) {
      const w = props.wires[i];
      const segs = wireSegments(w);
      if (!segs) continue;
      for (const [ax, ay, bx, by] of segs) {
        if (!pointNearSegment(wx, wy, ax, ay, bx, by, tol)) continue;
        // Project onto the orthogonal segment and snap to the grid.
        let px: number;
        let py: number;
        if (ay === by) {
          py = ay;
          px = Math.min(Math.max(snap(wx), Math.min(ax, bx)), Math.max(ax, bx));
        } else {
          px = ax;
          py = Math.min(Math.max(snap(wy), Math.min(ay, by)), Math.max(ay, by));
        }
        return { wire: w, x: px, y: py };
      }
    }
    return null;
  }

  /* ---------------- Selection ---------------- */

  function setSelection(ids: string[]): void {
    props.onSelect(new Set(ids));
  }

  function toggleSelection(id: string, additive: boolean): void {
    const next = new Set(props.selectedIds);
    if (additive) {
      if (next.has(id)) next.delete(id);
      else next.add(id);
    } else {
      next.clear();
      next.add(id);
    }
    props.onSelect(next);
  }

  function selectAndStartMove(hit: CanvasComponent, wx: number, wy: number, additive: boolean): void {
    const alreadySelected = props.selectedIds.has(hit.id);
    let nextSel: Set<string>;
    if (additive) {
      nextSel = new Set(props.selectedIds);
      if (alreadySelected) nextSel.delete(hit.id);
      else nextSel.add(hit.id);
    } else if (alreadySelected) {
      nextSel = new Set(props.selectedIds);
    } else {
      nextSel = new Set([hit.id]);
    }
    props.onSelect(nextSel);
    moveHistorySavedRef.current = false;
    const origin = new Map<string, { x: number; y: number }>();
    const wireOrigin = new Map<string, { start?: { x: number; y: number }; end?: { x: number; y: number } }>();
    for (const id of nextSel) {
      const cc = componentsById.get(id);
      if (cc) origin.set(id, { x: cc.x, y: cc.y });
    }
    for (const w of props.wires) {
      if (!nextSel.has(w.id)) continue;
      const entry: { start?: { x: number; y: number }; end?: { x: number; y: number } } = {};
      if (w.start.kind === "point") entry.start = { x: w.start.x, y: w.start.y };
      if (w.end.kind === "point") entry.end = { x: w.end.x, y: w.end.y };
      if (entry.start || entry.end) wireOrigin.set(w.id, entry);
    }
    setInteraction({ kind: "moving", ids: Array.from(nextSel), anchorWX: wx, anchorWY: wy, origin, wireOrigin });
  }

  /* ---------------- Wiring ---------------- */

  function finalizeWire(from: WireEndpoint, to: WireEndpoint): boolean {
    if (sameEndpoint(from, to)) return false;
    const fromXY = endpointXY(from, componentsById);
    const toXY = endpointXY(to, componentsById);
    if (fromXY && toXY && fromXY.x === toXY.x && fromXY.y === toXY.y) return false;
    const candidate: CanvasWire = { id: `w-${Math.random().toString(36).slice(2, 8)}`, start: from, end: to };
    const existing = new Set(props.wires.map(wireKey));
    if (existing.has(wireKey(candidate))) return false;
    props.onBeforeEdit();
    props.onSetWires((cur) => [...cur, candidate]);
    return true;
  }

  /** Split a wire at a point so a new wire can tap into it (a real T-junction). */
  function splitWire(wire: CanvasWire, point: { x: number; y: number }): WireEndpoint {
    const parts = splitWireAtPoint(wire, point, componentsById);
    const joint = pointEndpoint(point.x, point.y);
    if (!parts) return joint;
    props.onBeforeEdit();
    props.onSetWires((cur) => cur.flatMap((w) => (w.id === wire.id ? parts : [w])));
    return joint;
  }

  function startWiring(from: WireEndpoint, wx: number, wy: number): void {
    setInteraction({ kind: "wiring", from, cursorX: wx, cursorY: wy });
  }

  /* ---------------- Mouse ---------------- */

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const { wx, wy } = clientToWorld(e.clientX, e.clientY);
    canvasRef.current?.focus();

    // Right button: cancel the current gesture, else pan.
    if (e.button === 2) {
      e.preventDefault();
      if (interaction.kind === "wiring") {
        setInteraction({ kind: "idle" });
        return;
      }
      if (props.pendingDevice) {
        props.onPendingResolved();
        return;
      }
      if (props.wireTool) {
        props.onWireToolResolved();
        return;
      }
      setInteraction({ kind: "panning", startX: e.clientX, startY: e.clientY, origin: { ...pan }, moved: false });
      return;
    }
    if (e.button === 1) {
      e.preventDefault();
      setInteraction({ kind: "panning", startX: e.clientX, startY: e.clientY, origin: { ...pan }, moved: false });
      return;
    }
    if (e.button !== 0) return;

    // Placement mode: sticky like LTspice, every click drops another part.
    if (props.pendingDevice) {
      const type = props.pendingDevice;
      const nextIndex = props.components.filter((c) => c.type === type).length + 1;
      const nc = createDefaultComponent(type, nextIndex, wx, wy);
      nc.rotation = pendingRotation;
      nc.mirrored = pendingMirror || undefined;
      props.onBeforeEdit();
      props.onSetComponents((cur) => [...cur, nc]);
      return;
    }

    // Wiring in progress: the click completes or extends the wire.
    if (interaction.kind === "wiring") {
      const target = findPinAt(wx, wy);
      if (target) {
        finalizeWire(interaction.from, pinEndpoint(target.componentId, target.pin));
        setInteraction({ kind: "idle" });
        return;
      }
      const free = findFreeEndpointAt(wx, wy);
      if (free) {
        finalizeWire(interaction.from, pointEndpoint(free.x, free.y));
        setInteraction({ kind: "idle" });
        return;
      }
      const onWire = findWireAt(wx, wy);
      if (onWire) {
        const joint = splitWire(onWire.wire, { x: onWire.x, y: onWire.y });
        finalizeWire(interaction.from, joint);
        setInteraction({ kind: "idle" });
        return;
      }
      // Empty canvas: drop a corner and keep wiring from it.
      const joint = pointEndpoint(snap(wx), snap(wy));
      if (finalizeWire(interaction.from, joint)) {
        startWiring(joint, wx, wy);
      }
      return;
    }

    // Pin click → start wiring (or probe).
    const pin = findPinAt(wx, wy);
    if (pin) {
      const pinComponent = componentsById.get(pin.componentId);
      if (props.probeMode) {
        if (pinComponent) {
          const p = getPinCoordinates(pinComponent, pin.pin);
          props.onProbePick({
            componentId: pin.componentId,
            pin: pin.pin,
            netLabel: netLabelFor(pin.componentId, pin.pin),
            x: p.x,
            y: p.y
          });
        }
        return;
      }
      if (!props.wireTool && pinComponent?.type === "NODE") {
        selectAndStartMove(pinComponent, wx, wy, e.shiftKey);
        return;
      }
      startWiring(pinEndpoint(pin.componentId, pin.pin), wx, wy);
      return;
    }

    // Free endpoint → resume wiring from that point.
    const free = findFreeEndpointAt(wx, wy);
    if (free) {
      startWiring(pointEndpoint(free.x, free.y), free.x, free.y);
      return;
    }

    if (props.wireTool) {
      // Wire tool: tap an existing wire (splitting it) or start from empty space.
      const onWire = findWireAt(wx, wy);
      if (onWire) {
        const joint = splitWire(onWire.wire, { x: onWire.x, y: onWire.y });
        startWiring(joint, onWire.x, onWire.y);
        return;
      }
      startWiring(pointEndpoint(snap(wx), snap(wy)), wx, wy);
      return;
    }

    // Component click → select / start move.
    const hit = findComponentAt(wx, wy);
    if (hit) {
      selectAndStartMove(hit, wx, wy, e.shiftKey);
      return;
    }

    // Wire click → toggle that wire's selection.
    const wireHit = findWireAt(wx, wy);
    if (wireHit) {
      toggleSelection(wireHit.wire.id, e.shiftKey);
      return;
    }

    // Empty canvas → marquee select (shift keeps the current selection).
    if (!e.shiftKey) setSelection([]);
    setInteraction({ kind: "marquee", startWX: wx, startWY: wy, cursorWX: wx, cursorWY: wy });
  }

  function onDoubleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (props.pendingDevice || props.wireTool || props.probeMode) return;
    const { wx, wy } = clientToWorld(e.clientX, e.clientY);
    const hit = findComponentAt(wx, wy);
    if (hit) {
      props.onSelect(new Set([hit.id]));
      props.onOpenProperties(hit.id);
    }
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const { sx, sy, wx, wy } = clientToWorld(e.clientX, e.clientY);
    setMousePos({ x: sx, y: sy, wx, wy });
    props.onCursorChange?.({ x: snap(wx), y: snap(wy) });

    if (interaction.kind === "panning") {
      const dx = e.clientX - interaction.startX;
      const dy = e.clientY - interaction.startY;
      if (!interaction.moved && Math.hypot(dx, dy) > 2) {
        setInteraction({ ...interaction, moved: true });
      }
      setPan({ x: interaction.origin.x + dx, y: interaction.origin.y + dy, scale: interaction.origin.scale });
      return;
    }
    if (interaction.kind === "moving") {
      const dx = wx - interaction.anchorWX;
      const dy = wy - interaction.anchorWY;
      if (!moveHistorySavedRef.current && Math.hypot(dx, dy) > 1) {
        props.onBeforeEdit();
        moveHistorySavedRef.current = true;
      }
      const origin = interaction.origin;
      const wireOrigin = interaction.wireOrigin;
      props.onSetComponents((cur) =>
        cur.map((c) => {
          const o = origin.get(c.id);
          return o ? { ...c, x: snap(o.x + dx), y: snap(o.y + dy) } : c;
        })
      );
      if (wireOrigin.size > 0) {
        props.onSetWires((cur) =>
          cur.map((w) => {
            const o = wireOrigin.get(w.id);
            if (!o) return w;
            return {
              ...w,
              start: o.start ? pointEndpoint(o.start.x + dx, o.start.y + dy) : w.start,
              end: o.end ? pointEndpoint(o.end.x + dx, o.end.y + dy) : w.end
            };
          })
        );
      }
      return;
    }
    if (interaction.kind === "wiring") {
      setInteraction({ ...interaction, cursorX: wx, cursorY: wy });
    }
    if (interaction.kind === "marquee") {
      setInteraction({ ...interaction, cursorWX: wx, cursorWY: wy });
    }
    setHoverPin(findPinAt(wx, wy));
    setHoverFreePoint(findFreeEndpointAt(wx, wy));
    setHoverWireId(props.wireTool || interaction.kind === "wiring" ? findWireAt(wx, wy)?.wire.id ?? null : null);
  }

  function onMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
    if (interaction.kind === "marquee") {
      const { startWX, startWY, cursorWX, cursorWY } = interaction;
      if (Math.hypot(cursorWX - startWX, cursorWY - startWY) > 4) {
        const minx = Math.min(startWX, cursorWX);
        const maxx = Math.max(startWX, cursorWX);
        const miny = Math.min(startWY, cursorWY);
        const maxy = Math.max(startWY, cursorWY);
        const picked: string[] = e.shiftKey ? Array.from(props.selectedIds) : [];
        for (const c of props.components) {
          const b = componentBounds(c);
          const cx = b.x + b.w / 2;
          const cy = b.y + b.h / 2;
          if (cx >= minx && cx <= maxx && cy >= miny && cy <= maxy) picked.push(c.id);
        }
        for (const w of props.wires) {
          const a = endpointXY(w.start, componentsById);
          const b = endpointXY(w.end, componentsById);
          if (!a || !b) continue;
          if (
            a.x >= minx && a.x <= maxx && a.y >= miny && a.y <= maxy &&
            b.x >= minx && b.x <= maxx && b.y >= miny && b.y <= maxy
          ) picked.push(w.id);
        }
        setSelection(picked);
      }
    }
    if (interaction.kind !== "wiring") {
      setInteraction({ kind: "idle" });
    }
  }

  function onWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    zoomAround(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.001));
  }

  /* ---------------- View control ---------------- */

  const zoomAround = useCallback((sx: number, sy: number, factor: number) => {
    setPan((cur) => {
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cur.scale * factor));
      const wx = (sx - cur.x) / cur.scale;
      const wy = (sy - cur.y) / cur.scale;
      return { x: sx - wx * newScale, y: sy - wy * newScale, scale: newScale };
    });
  }, []);

  const canvasSize = () => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement as HTMLDivElement | null;
    return { w: parent?.clientWidth ?? 800, h: parent?.clientHeight ?? 600 };
  };

  const zoomToFit = useCallback(() => {
    const { w, h } = canvasSize();
    const bounds = schematicBounds(props.components, props.wires);
    if (!bounds) {
      setPan({ x: 40, y: 40, scale: 1 });
      return;
    }
    const margin = 60;
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.min((w - margin * 2) / bounds.w, (h - margin * 2) / bounds.h, 2)));
    const cx = bounds.x + bounds.w / 2;
    const cy = bounds.y + bounds.h / 2;
    setPan({ x: w / 2 - cx * scale, y: h / 2 - cy * scale, scale });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.components, props.wires]);

  useImperativeHandle(
    ref,
    () => ({
      zoomIn: () => {
        const { w, h } = canvasSize();
        zoomAround(w / 2, h / 2, 1.25);
      },
      zoomOut: () => {
        const { w, h } = canvasSize();
        zoomAround(w / 2, h / 2, 0.8);
      },
      zoomToFit,
      resetZoom: () => setPan({ x: 40, y: 40, scale: 1 }),
      focus: () => canvasRef.current?.focus()
    }),
    [zoomAround, zoomToFit]
  );

  /* ---------------- Editing commands ---------------- */

  function deleteSelection(): void {
    if (props.selectedIds.size === 0) return;
    const sel = props.selectedIds;
    props.onBeforeEdit();
    props.onSetComponents((cur) => cur.filter((c) => !sel.has(c.id)));
    props.onSetWires((cur) =>
      cur.filter((w) => {
        if (sel.has(w.id)) return false;
        if (w.start.kind === "pin" && sel.has(w.start.componentId)) return false;
        if (w.end.kind === "pin" && sel.has(w.end.componentId)) return false;
        return true;
      })
    );
    props.onSelect(new Set());
  }

  function rotateSelection(): void {
    if (props.selectedIds.size === 0) return;
    props.onBeforeEdit();
    props.onSetComponents((cur) =>
      cur.map((c) =>
        props.selectedIds.has(c.id) ? { ...c, rotation: ((c.rotation + 90) % 360) as Rotation } : c
      )
    );
  }

  function nudgeSelection(dx: number, dy: number): void {
    if (props.selectedIds.size === 0) return;
    const sel = props.selectedIds;
    props.onBeforeEdit();
    props.onSetComponents((cur) => cur.map((c) => (sel.has(c.id) ? { ...c, x: c.x + dx, y: c.y + dy } : c)));
    props.onSetWires((cur) =>
      cur.map((w) => {
        if (!sel.has(w.id)) return w;
        return {
          ...w,
          start: w.start.kind === "point" ? pointEndpoint(w.start.x + dx, w.start.y + dy) : w.start,
          end: w.end.kind === "point" ? pointEndpoint(w.end.x + dx, w.end.y + dy) : w.end
        };
      })
    );
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLCanvasElement>) {
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    // Handled keys stop here so the window-level shortcut handler in App
    // does not act on the same event a second time.
    const consume = () => {
      e.preventDefault();
      e.stopPropagation();
    };

    if (e.key === "Escape") {
      if (interaction.kind === "wiring" || interaction.kind === "marquee") {
        setInteraction({ kind: "idle" });
      } else if (props.pendingDevice) {
        props.onPendingResolved();
      } else if (props.wireTool) {
        props.onWireToolResolved();
      } else {
        props.onSelect(new Set());
      }
      consume();
      return;
    }

    // While placing, R / Ctrl+R rotate the ghost and Ctrl+E / E mirror it.
    if (props.pendingDevice) {
      if (key === "r") {
        setPendingRotation((r) => ((r + 90) % 360) as Rotation);
        consume();
        return;
      }
      if (key === "e") {
        setPendingMirror((m) => !m);
        consume();
        return;
      }
    }

    if (e.key === "Delete" || e.key === "Backspace") {
      if (props.selectedIds.size > 0) {
        deleteSelection();
        consume();
      }
      return;
    }
    if (!ctrl && key === "r") {
      rotateSelection();
      consume();
      return;
    }
    if (!ctrl && e.key === " ") {
      zoomToFit();
      consume();
      return;
    }
    if (ctrl && key === "c") {
      props.onCopy();
      consume();
      return;
    }
    if (ctrl && key === "v") {
      props.onPaste();
      consume();
      return;
    }
    if (ctrl && key === "d") {
      props.onDuplicate();
      consume();
      return;
    }
    if (ctrl && key === "a") {
      const all = new Set<string>();
      props.components.forEach((c) => all.add(c.id));
      props.wires.forEach((w) => all.add(w.id));
      props.onSelect(all);
      consume();
      return;
    }
    if (!ctrl && e.key.startsWith("Arrow") && props.selectedIds.size > 0) {
      const step = e.shiftKey ? GRID * 5 : GRID;
      if (e.key === "ArrowLeft") nudgeSelection(-step, 0);
      else if (e.key === "ArrowRight") nudgeSelection(step, 0);
      else if (e.key === "ArrowUp") nudgeSelection(0, -step);
      else if (e.key === "ArrowDown") nudgeSelection(0, step);
      consume();
    }
  }

  /* ---------------- Drawing ---------------- */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement as HTMLDivElement;
    const obs = new ResizeObserver(() => draw());
    obs.observe(parent);
    return () => obs.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    props.components,
    props.wires,
    props.junctions,
    pan,
    hoverPin,
    hoverFreePoint,
    hoverWireId,
    interaction,
    mousePos,
    props.selectedIds,
    props.pendingDevice,
    props.probeMode,
    props.wireTool,
    props.showGrid,
    props.showNetNames,
    pendingRotation,
    pendingMirror
  ]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement as HTMLDivElement;
    const ratio = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== w * ratio || canvas.height !== h * ratio) {
      canvas.width = w * ratio;
      canvas.height = h * ratio;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);

    drawGrid(ctx, w, h, pan, props.showGrid);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(pan.scale, pan.scale);

    // Wires
    const netLabelAnchors: { name: string; x: number; y: number; vertical: boolean }[] = [];
    const labeledNets = new Set<string>();
    for (const wire of props.wires) {
      const segs = wireSegments(wire);
      if (!segs) continue;
      const isSelected = props.selectedIds.has(wire.id);
      const isHover = hoverWireId === wire.id;
      ctx.save();
      ctx.strokeStyle = isSelected ? "#ffd26d" : isHover ? "#9fc4ff" : DEFAULT_THEME.accent;
      ctx.lineWidth = isSelected ? 2.6 : isHover ? 2.4 : 1.8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      strokeSegments(ctx, segs);
      ctx.restore();

      // One label per anonymous net (named nets already carry a NODE marker).
      const netName = nets.byWire.get(wire.id);
      if (props.showNetNames && netName && AUTO_NET.test(netName) && !labeledNets.has(netName)) {
        labeledNets.add(netName);
        // Prefer the longer segment so the label sits on a stretch of wire
        // rather than on a corner or a junction.
        const seg = segs.reduce((best, cur) =>
          Math.hypot(cur[2] - cur[0], cur[3] - cur[1]) > Math.hypot(best[2] - best[0], best[3] - best[1]) ? cur : best
        );
        const [ax, ay, bx, by] = seg;
        netLabelAnchors.push({ name: netName, x: (ax + bx) / 2, y: (ay + by) / 2, vertical: ax === bx });
      }
    }

    // Free (dangling) wire ends: a corner between two wires is invisible,
    // an open end that nothing else touches gets a small hollow marker.
    const endCount = new Map<string, number>();
    for (const wire of props.wires) {
      for (const ep of [wire.start, wire.end]) {
        if (ep.kind !== "point") continue;
        const key = `${ep.x}:${ep.y}`;
        endCount.set(key, (endCount.get(key) ?? 0) + 1);
      }
    }
    for (const [key, n] of endCount) {
      if (n !== 1) continue;
      const [x, y] = key.split(":").map(Number);
      ctx.save();
      ctx.fillStyle = "#0b0f17";
      ctx.strokeStyle = "#ffc46b";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Junction dots where three or more connections meet (LTspice style).
    ctx.save();
    ctx.fillStyle = DEFAULT_THEME.accent;
    for (const j of census.junctions) {
      ctx.beginPath();
      ctx.arc(j.x, j.y, 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Named junctions (e.g. SUBCKT port_in / port_out markers).
    for (const j of props.junctions ?? []) {
      ctx.save();
      ctx.fillStyle = "#ffd26d";
      ctx.strokeStyle = "#ffd26d";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(j.x, j.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(j.x, j.y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      drawNetLabel(ctx, j.x, j.y, j.id);
    }

    // Components
    for (const c of props.components) {
      drawComponent(ctx, c, DEFAULT_THEME);
      const b = componentBounds(c);
      // SUBCKT blocks print their name inside the body, so skip the caption.
      if (c.type !== "LABEL" && c.type !== "NODE" && c.type !== "SUBCKT") {
        ctx.fillStyle = DEFAULT_THEME.label;
        ctx.font = "500 11px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const labelY = labelBaseline(c, b);
        ctx.fillText(c.name, labelY.x, labelY.y);
        if (c.type !== "GND") {
          ctx.fillStyle = DEFAULT_THEME.muted;
          ctx.fillText(c.value, labelY.x, labelY.y + 13);
        }
      }
      if (props.selectedIds.has(c.id)) {
        ctx.save();
        ctx.strokeStyle = DEFAULT_THEME.accent;
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1;
        ctx.strokeRect(b.x - 2, b.y - 2, b.w + 4, b.h + 4);
        ctx.restore();
      }
      // Pins: wired pins are nearly invisible, open pins show a hollow
      // square so an unconnected terminal is obvious at a glance.
      if (c.type === "NODE") continue;
      for (const pin of getPinsForType(c.type, c)) {
        const p = getPinCoordinates(c, pin);
        const isHover = hoverPin?.componentId === c.id && hoverPin.pin === pin;
        const wired = census.wiredPins.has(`${c.id}:${pin}`);
        ctx.save();
        if (isHover) {
          ctx.fillStyle = DEFAULT_THEME.accent;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
          ctx.fill();
        } else if (wired) {
          ctx.fillStyle = "rgba(110,168,255,0.55)";
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.strokeStyle = "#ffc46b";
          ctx.fillStyle = "rgba(255,196,107,0.12)";
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.rect(p.x - 4, p.y - 4, 8, 8);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // Net labels sit on top of everything so pins never clip them.
    for (const anchor of netLabelAnchors) {
      drawNetLabel(ctx, anchor.x, anchor.y, anchor.name, anchor.vertical ? "right" : "above");
    }

    // In-flight wire
    if (interaction.kind === "wiring") {
      const fromXY = endpointXY(interaction.from, componentsById);
      if (fromXY) {
        const owner = interaction.from.kind === "pin" ? interaction.from.componentId : null;
        const obstacles = Array.from(obstacleBounds.entries())
          .filter(([cid]) => cid !== owner)
          .map(([, b]) => b);
        const tx = snap(interaction.cursorX);
        const ty = snap(interaction.cursorY);
        ctx.save();
        ctx.strokeStyle = DEFAULT_THEME.accent;
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.6;
        strokeSegments(ctx, routeOrthogonal(fromXY.x, fromXY.y, tx, ty, obstacles));
        ctx.restore();
      }
    }

    // Marquee box
    if (interaction.kind === "marquee") {
      const minx = Math.min(interaction.startWX, interaction.cursorWX);
      const maxx = Math.max(interaction.startWX, interaction.cursorWX);
      const miny = Math.min(interaction.startWY, interaction.cursorWY);
      const maxy = Math.max(interaction.startWY, interaction.cursorWY);
      ctx.save();
      ctx.strokeStyle = DEFAULT_THEME.accent;
      ctx.fillStyle = "rgba(110,168,255,0.08)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.fillRect(minx, miny, maxx - minx, maxy - miny);
      ctx.strokeRect(minx, miny, maxx - minx, maxy - miny);
      ctx.restore();
    }

    // Probe-mode crosshair
    if (props.probeMode && hoverPin) {
      const c = componentsById.get(hoverPin.componentId);
      if (c) {
        const p = getPinCoordinates(c, hoverPin.pin);
        ctx.save();
        ctx.strokeStyle = "#ffd26d";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Free-endpoint hover ring
    if (hoverFreePoint && interaction.kind !== "wiring") {
      ctx.save();
      ctx.strokeStyle = "#ffd26d";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(hoverFreePoint.x, hoverFreePoint.y, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Wire-tool cursor: snapped crosshair where the next wire would start.
    if ((props.wireTool || interaction.kind === "wiring") && mousePos && !hoverPin) {
      const x = snap(mousePos.wx);
      const y = snap(mousePos.wy);
      ctx.save();
      ctx.strokeStyle = "rgba(110,168,255,0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - 7, y);
      ctx.lineTo(x + 7, y);
      ctx.moveTo(x, y - 7);
      ctx.lineTo(x, y + 7);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();

    // Pending device ghost
    if (props.pendingDevice && mousePos) {
      ctx.save();
      ctx.translate(pan.x, pan.y);
      ctx.scale(pan.scale, pan.scale);
      ctx.globalAlpha = 0.6;
      drawComponent(
        ctx,
        {
          id: "ghost",
          type: props.pendingDevice,
          name: props.pendingDevice,
          value: "",
          x: snap(mousePos.wx),
          y: snap(mousePos.wy),
          rotation: pendingRotation,
          mirrored: pendingMirror,
          pins: props.pendingDevice === "SUBCKT" ? ["in_1", "out_1"] : undefined
        },
        DEFAULT_THEME
      );
      ctx.restore();
    }
  }

  const hint = props.probeMode
    ? "Probe mode: click a pin to open a live .dyn tile · Esc to exit"
    : interaction.kind === "wiring"
      ? "Wiring: click a pin or wire to finish, click empty space for a corner, Esc / right-click to stop"
      : props.wireTool
        ? "Wire tool: click a pin, a wire, or empty space to start a wire · Esc / right-click to exit"
        : props.pendingDevice
          ? `Placing ${props.pendingDevice}: click to place · R rotate · E mirror · Esc / right-click to finish`
          : null;

  const cursor = props.pendingDevice
    ? "copy"
    : props.probeMode || props.wireTool || interaction.kind === "wiring"
      ? "crosshair"
      : interaction.kind === "panning"
        ? "grabbing"
        : "default";

  return (
    <div className="canvasArea" style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        className="schematicCanvas"
        tabIndex={0}
        onMouseDown={onMouseDown}
        onDoubleClick={onDoubleClick}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          setHoverPin(null);
          setHoverFreePoint(null);
          setHoverWireId(null);
          setMousePos(null);
          props.onCursorChange?.(null);
          if (interaction.kind === "panning" || interaction.kind === "moving" || interaction.kind === "marquee") {
            setInteraction({ kind: "idle" });
          }
        }}
        onContextMenu={(e) => e.preventDefault()}
        onWheel={onWheel}
        onKeyDown={onKeyDown}
        style={{ cursor }}
      />
      {hoverPin && mousePos ? (
        <div className="hoverBadge" style={{ left: mousePos.x + 12, top: mousePos.y + 16 }}>
          <strong>{netLabelFor(hoverPin.componentId, hoverPin.pin)}</strong>
          {netNameForPin(hoverPin.componentId, hoverPin.pin) ? (
            <span className="hoverBadge__net">net {netNameForPin(hoverPin.componentId, hoverPin.pin)}</span>
          ) : (
            <span className="hoverBadge__net hoverBadge__net--open">unconnected</span>
          )}
        </div>
      ) : null}
      {hint ? <div className="canvasHint">{hint}</div> : null}
    </div>
  );
});

/* ---------- helpers ---------- */

function labelBaseline(c: CanvasComponent, b: Rect): { x: number; y: number } {
  // Two-terminal parts rotated to vertical keep their leads on the top and
  // bottom edge, so the label goes beside the body instead of onto the wire.
  const vertical = (c.rotation === 90 || c.rotation === 270) &&
    (c.type === "R" || c.type === "C" || c.type === "L" || c.type === "D" || c.type === "CCCS" || c.type === "CCVS");
  if (vertical) return { x: b.x + b.w + 18, y: c.y - 12 };
  const horizontalSource = (c.rotation === 90 || c.rotation === 270) && (c.type === "V" || c.type === "I");
  if (horizontalSource) return { x: c.x, y: b.y + b.h + 4 };
  if (c.type === "V" || c.type === "I" || c.type === "QNPN" || c.type === "QPNP" || c.type === "NMOS" || c.type === "PMOS") {
    return { x: b.x + b.w + 16, y: c.y - 12 };
  }
  return { x: c.x, y: b.y + b.h + 4 };
}

function sameEndpoint(a: WireEndpoint, b: WireEndpoint): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "pin" && b.kind === "pin") return a.componentId === b.componentId && a.pin === b.pin;
  if (a.kind === "point" && b.kind === "point") return a.x === b.x && a.y === b.y;
  return false;
}

function pointNearSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number, tol: number): boolean {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px - ax, py - ay) < tol;
  let t = ((px - ax) * dx + (py - ay) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + dx * t), py - (ay + dy * t)) < tol;
}

function drawNetLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  name: string,
  placement: "above" | "right" | "corner" = "corner"
) {
  ctx.save();
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const padding = 4;
  const boxW = ctx.measureText(name).width + padding * 2;
  const boxH = 14;
  // Keep the box clear of the wire it names: centred above a horizontal
  // run, beside a vertical run, or offset diagonally from a point marker.
  const bx = placement === "above" ? x - boxW / 2 : x + 7;
  const by = placement === "right" ? y - boxH / 2 : placement === "above" ? y - boxH - 5 : y - boxH / 2 - 4;
  ctx.fillStyle = "rgba(11,15,23,0.88)";
  ctx.strokeStyle = "rgba(110,168,255,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(bx, by, boxW, boxH);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#cfe1ff";
  ctx.fillText(name, bx + padding, by + boxH / 2);
  ctx.restore();
}

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number, pan: PanState, show: boolean) {
  ctx.save();
  ctx.fillStyle = "#0b0f17";
  ctx.fillRect(0, 0, w, h);
  if (!show) {
    ctx.restore();
    return;
  }
  const step = GRID * pan.scale;
  // Skip the minor grid when it would be denser than 6px; keep the major grid.
  if (step >= 6) {
    ctx.strokeStyle = "rgba(110,168,255,0.06)";
    ctx.lineWidth = 1;
    const ox = pan.x % step;
    const oy = pan.y % step;
    ctx.beginPath();
    for (let x = ox; x < w; x += step) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = oy; y < h; y += step) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
  }
  const bigStep = step * 4;
  if (bigStep >= 6) {
    ctx.strokeStyle = "rgba(110,168,255,0.12)";
    ctx.beginPath();
    const bx = pan.x % bigStep;
    const by = pan.y % bigStep;
    for (let x = bx; x < w; x += bigStep) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = by; y < h; y += bigStep) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function segmentCrossesRect(ax: number, ay: number, bx: number, by: number, rect: Rect, pad = 4): boolean {
  const rx1 = rect.x + pad;
  const ry1 = rect.y + pad;
  const rx2 = rect.x + rect.w - pad;
  const ry2 = rect.y + rect.h - pad;
  if (rx2 <= rx1 || ry2 <= ry1) return false;
  let t0 = 0;
  let t1 = 1;
  const dx = bx - ax;
  const dy = by - ay;
  const clip = (p: number, q: number): boolean => {
    if (p === 0) return q >= 0;
    const r = q / p;
    if (p < 0) {
      if (r > t1) return false;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return false;
      if (r < t1) t1 = r;
    }
    return true;
  };
  if (!clip(-dx, ax - rx1)) return false;
  if (!clip(dx, rx2 - ax)) return false;
  if (!clip(-dy, ay - ry1)) return false;
  if (!clip(dy, ry2 - ay)) return false;
  return t1 > t0;
}

function countCrossings(obstacles: Rect[], segments: Segment[]): number {
  let n = 0;
  for (const [ax, ay, bx, by] of segments) {
    for (const r of obstacles) if (segmentCrossesRect(ax, ay, bx, by, r)) n++;
  }
  return n;
}

/** Two-segment orthogonal route choosing the elbow that crosses fewer bodies. */
function routeOrthogonal(x1: number, y1: number, x2: number, y2: number, obstacles: Rect[]): Segment[] {
  if (x1 === x2 || y1 === y2) return [[x1, y1, x2, y2]];
  const hFirst: Segment[] = [[x1, y1, x2, y1], [x2, y1, x2, y2]];
  const vFirst: Segment[] = [[x1, y1, x1, y2], [x1, y2, x2, y2]];
  return countCrossings(obstacles, vFirst) < countCrossings(obstacles, hFirst) ? vFirst : hFirst;
}

function strokeSegments(ctx: CanvasRenderingContext2D, segments: Segment[]) {
  if (segments.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(segments[0][0], segments[0][1]);
  for (const [, , ex, ey] of segments) ctx.lineTo(ex, ey);
  ctx.stroke();
}
