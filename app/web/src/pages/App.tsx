import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { LeftPane } from "../components/LeftPane";
import { PlotTile } from "../components/PlotTile";
import { OpResultData, ResultTile } from "../components/ResultTile";
import { ProbeTarget, SchematicCanvas, SchematicCanvasHandle } from "../components/SchematicCanvas";
import { Toolbar } from "../components/Toolbar";
import { DEMO_MENU_GROUPS } from "../lib/demoPresets";
import {
  HierarchyError,
  PortSyncReport,
  breadcrumbPath,
  decorateSubcktInstances,
  describePortSync,
  dissolveSubcircuit,
  duplicateEntity,
  entityContainsLevel,
  entityPorts,
  firstHierarchyIssue,
  instanceTemplate,
  makeSubcircuitFromSelection,
  nextSubcktTitle,
  parentLevelId,
  remapInstancePinEndpoints,
  subcktDecorations,
  syncEntityPorts,
  withEntityPinLayout
} from "../lib/hierarchy";
import { PlotData } from "../lib/plot";
import {
  AnalysisState,
  CanvasComponent,
  CanvasWire,
  ClipboardData,
  DeviceType,
  SchematicLevel,
  SchematicPreset,
  ThreadSetting,
  buildSchematicPayload,
  extractClipboard,
  getPinCoordinates,
  parseWorkspace,
  serializeWorkspace
} from "../lib/schematic";
import { SubcktSymbolInfo } from "../lib/symbols";
import { USER_GUIDE_SECTIONS } from "../lib/userGuide";

const defaultAnalysis: AnalysisState = {
  mode: "op",
  tStop: "10m",
  tStep: "0.1m",
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

type TileRecord = {
  id: string;
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
  mode: "static" | "dyn";
  view: "plot" | "op";
  data?: PlotData;
  opData?: OpResultData;
  /** For dyn tiles: push function installed by the child. */
  push?: (t: number, values: number[], labels: string[]) => void;
  finalize?: () => void;
  /** Probe target for dyn tiles. */
  probe?: ProbeTarget;
  socket?: WebSocket;
  /** Rolling display window in sim seconds (dyn tiles only). */
  windowSeconds?: number;
};

type EditorSnapshot = {
  levels: SchematicLevel[];
  activeLevelId: string;
  netlistMode: "schematic" | "netlist";
  netlistText: string;
  generatedNetlist: string;
};

type MenuId = "file" | "edit" | "view" | "simulate" | "demo";

type MenuEntry =
  | { kind: "item"; label: string; shortcut?: string; onSelect: () => void; disabled?: boolean; checked?: boolean }
  | { kind: "separator" };

/** Tiny external store so cursor coordinates re-render only the status readout. */
function createCursorStore() {
  let value: { x: number; y: number } | null = null;
  const listeners = new Set<() => void>();
  return {
    set(next: { x: number; y: number } | null) {
      if (next === value) return;
      if (next && value && next.x === value.x && next.y === value.y) return;
      value = next;
      listeners.forEach((listener) => listener());
    },
    get: () => value,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    }
  };
}

type CursorStore = ReturnType<typeof createCursorStore>;

function CursorReadout({ store }: { store: CursorStore }) {
  const pos = useSyncExternalStore(store.subscribe, store.get);
  return (
    <span className="statusbar__slot statusbar__slot--mono" title="Cursor position (grid units)">
      {pos ? `x ${pos.x}  y ${pos.y}` : "x —  y —"}
    </span>
  );
}

function downloadText(filename: string, text: string, mime = "application/json"): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fileSlug(title: string): string {
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || "schematic";
}

const EMPTY_ROOT: SchematicLevel = {
  id: "root",
  title: "Top level",
  components: [],
  wires: [],
  pins: [],
  parentId: null
};

const LIBRARY_ITEMS: DeviceType[] = [
  "R", "C", "L", "V", "I", "D", "GND",
  "QNPN", "QPNP", "NMOS", "PMOS",
  "VCVS", "VCCS", "CCCS", "CCVS", "SUBCKT", "LABEL", "NODE"
];
const MAX_STATIC_PLOT_POINTS = 6000;
const MAX_RESPONSE_PREVIEW_VALUES = 20000;

function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function cloneWireEndpoint(endpoint: CanvasWire["start"]): CanvasWire["start"] {
  return endpoint.kind === "pin"
    ? { kind: "pin", componentId: endpoint.componentId, pin: endpoint.pin }
    : { kind: "point", x: endpoint.x, y: endpoint.y };
}

function cloneLevel(level: SchematicLevel): SchematicLevel {
  return {
    ...level,
    components: level.components.map((component) => ({
      ...component,
      pins: component.pins ? [...component.pins] : undefined,
      metadata: component.metadata ? { ...component.metadata } : undefined
    })),
    wires: level.wires.map((wire) => ({
      ...wire,
      start: cloneWireEndpoint(wire.start),
      end: cloneWireEndpoint(wire.end)
    })),
    pins: [...level.pins],
    junctions: level.junctions?.map((junction) => ({ ...junction }))
  };
}

function cloneLevels(levels: SchematicLevel[]): SchematicLevel[] {
  return levels.map(cloneLevel);
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

/** Accept "auto" or a positive integer from saved workspaces; anything else keeps the fallback. */
function threadSettingOr(value: unknown, fallback: ThreadSetting): ThreadSetting {
  if (value === "auto") return "auto";
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 ? n : fallback;
}

function simulationRequestOptions(analysis: AnalysisState): Record<string, unknown> {
  const options: Record<string, unknown> = {
    use_krylov: analysis.krylov,
    probe_nodes: analysis.probeNodes,
    num_threads: threadSettingOr(analysis.numThreads, "auto")
  };
  if (analysis.krylov) {
    options.krylov_rank =
      analysis.krylovRankMode === "auto"
        ? "auto"
        : Math.max(1, Math.floor(Number(analysis.krylovRank) || 1));
    options.krylov_method = analysis.krylovMethod;
  }
  if (analysis.mor) {
    options.use_mor = true;
    options.mor_method = analysis.morMethod;
    options.mor_order =
      analysis.morOrderMode === "auto"
        ? "auto"
        : Math.max(1, Math.floor(Number(analysis.morOrder) || 1));
    options.mor_output_nodes = analysis.morOutputNodes;
    options.mor_validate = true;
  }
  return options;
}

export function App() {
  /* -------------- Hierarchy levels -------------- */
  const [levels, setLevels] = useState<SchematicLevel[]>(() => [
    {
      id: "root",
      title: "Top level",
      components: [],
      wires: [],
      pins: [],
      parentId: null
    }
  ]);
  const [activeLevelId, setActiveLevelId] = useState<string>("root");

  const activeLevel = levels.find((l) => l.id === activeLevelId)!;

  const updateLevel = useCallback(
    (levelId: string, patch: Partial<SchematicLevel>) => {
      setLevels((cur) =>
        cur.map((l) => (l.id === levelId ? { ...l, ...patch } : l))
      );
    },
    []
  );

  /**
   * Port changes found while applying an edit (set inside the state updater,
   * shown in the status line after the render). Updaters may run twice in
   * strict mode; they compute the same report both times.
   */
  const portReportRef = useRef<PortSyncReport | null>(null);

  /**
   * Apply an edit to one level. When that level is an entity whose NODE
   * markers (= ports) changed, every instance of it follows in the same
   * state update, so the edit stays a single undo step.
   */
  const editLevel = useCallback(
    (levelId: string, edit: (level: SchematicLevel, all: SchematicLevel[]) => SchematicLevel) => {
      setLevels((cur) => {
        const next = cur.map((l) => (l.id === levelId ? edit(l, cur) : l));
        const synced = syncEntityPorts(cur, next, levelId);
        if (synced.report) portReportRef.current = synced.report;
        return synced.levels;
      });
    },
    []
  );

  const setActiveComponents = useCallback(
    (updater: (current: CanvasComponent[]) => CanvasComponent[]) => {
      editLevel(activeLevelId, (l) => ({ ...l, components: updater(l.components) }));
    },
    [activeLevelId, editLevel]
  );
  const setActiveWires = useCallback(
    (updater: (current: CanvasWire[]) => CanvasWire[]) => {
      editLevel(activeLevelId, (l) => ({ ...l, wires: updater(l.wires) }));
    },
    [activeLevelId, editLevel]
  );

  /* -------------- Analysis & UI state -------------- */
  const [analysis, setAnalysis] = useState<AnalysisState>({ ...defaultAnalysis });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [pendingDevice, setPendingDevice] = useState<DeviceType | null>(null);
  /** Fields of the part being placed, e.g. a SUBCKT linked to an entity ("Instantiate"). */
  const [pendingTemplate, setPendingTemplate] = useState<Partial<CanvasComponent> | null>(null);
  /** Open canvas context menu (screen position and the part under the cursor). */
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; componentId: string | null } | null>(null);
  const [probeMode, setProbeMode] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<"idle" | "ok" | "error" | "running">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [generatedNetlist, setGeneratedNetlist] = useState<string>("");
  const [netlistMode, setNetlistMode] = useState<"schematic" | "netlist">("schematic");
  const [netlistText, setNetlistText] = useState<string>("");
  const [responsePreview, setResponsePreview] = useState<string>("");
  const [tiles, setTiles] = useState<TileRecord[]>([]);
  const [clipboard, setClipboard] = useState<ClipboardData | null>(null);
  const [openMenu, setOpenMenu] = useState<MenuId | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [wireTool, setWireTool] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showNetNames, setShowNetNames] = useState(true);
  const [zoomScale, setZoomScale] = useState(1);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [propertiesFocusToken, setPropertiesFocusToken] = useState(0);
  /** Server thread budget from GET /health (null until known). */
  const [threadBudgetTotal, setThreadBudgetTotal] = useState<number | null>(null);
  const canvasHandleRef = useRef<SchematicCanvasHandle | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const redoRef = useRef<EditorSnapshot[]>([]);
  const cursorStore = useRef<CursorStore | null>(null);
  if (!cursorStore.current) cursorStore.current = createCursorStore();
  const [statusElapsedMs, setStatusElapsedMs] = useState(0);
  const [statusIterations, setStatusIterations] = useState(0);
  const tileCounterRef = useRef(1);
  const statusStartRef = useRef<number | null>(null);
  const statusTimerRef = useRef<number | null>(null);
  const statusIterationsRef = useRef(0);
  const statusIterationPaintRef = useRef(0);
  const activeDynamicTilesRef = useRef<Set<string>>(new Set());
  const levelsRef = useRef<SchematicLevel[]>(levels);
  const activeLevelIdRef = useRef(activeLevelId);
  const netlistModeRef = useRef(netlistMode);
  const netlistTextRef = useRef(netlistText);
  const generatedNetlistRef = useRef(generatedNetlist);
  const historyRef = useRef<EditorSnapshot[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch("/health")
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        const total = Number(body?.thread_budget?.total);
        if (!cancelled && Number.isInteger(total) && total >= 1) setThreadBudgetTotal(total);
      })
      .catch(() => {
        // The Threads control falls back to "auto" and 1 until the server answers.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    levelsRef.current = levels;
    activeLevelIdRef.current = activeLevelId;
    netlistModeRef.current = netlistMode;
    netlistTextRef.current = netlistText;
    generatedNetlistRef.current = generatedNetlist;
  }, [levels, activeLevelId, netlistMode, netlistText, generatedNetlist]);

  // Report port changes that an edit propagated to instances (see editLevel).
  useEffect(() => {
    const report = portReportRef.current;
    if (!report) return;
    portReportRef.current = null;
    const entity = levels.find((l) => l.id === report.entityId);
    setStatusMsg(describePortSync(report, entity?.title ?? report.entityId));
  }, [levels]);

  const captureSnapshot = useCallback(
    (): EditorSnapshot => ({
      levels: cloneLevels(levelsRef.current),
      activeLevelId: activeLevelIdRef.current,
      netlistMode: netlistModeRef.current,
      netlistText: netlistTextRef.current,
      generatedNetlist: generatedNetlistRef.current
    }),
    []
  );

  const pushEditorHistory = useCallback(() => {
    historyRef.current.push(captureSnapshot());
    if (historyRef.current.length > 100) {
      historyRef.current.shift();
    }
    // A fresh edit invalidates the redo branch, like every desktop editor.
    redoRef.current = [];
    setHistoryVersion((v) => v + 1);
  }, [captureSnapshot]);

  const restoreSnapshot = useCallback((snapshot: EditorSnapshot) => {
    setLevels(cloneLevels(snapshot.levels));
    setActiveLevelId(snapshot.activeLevelId);
    setNetlistMode(snapshot.netlistMode);
    setNetlistText(snapshot.netlistText);
    setGeneratedNetlist(snapshot.generatedNetlist);
    setSelectedIds(new Set());
    setPendingDevice(null);
    setProbeMode(false);
    setWireTool(false);
    setStatus("idle");
  }, []);

  const undoEditor = useCallback(() => {
    const snapshot = historyRef.current.pop();
    if (!snapshot) {
      setStatusMsg("Nothing to undo.");
      return;
    }
    redoRef.current.push(captureSnapshot());
    restoreSnapshot(snapshot);
    setHistoryVersion((v) => v + 1);
    setStatusMsg("Undid last edit.");
  }, [captureSnapshot, restoreSnapshot]);

  const redoEditor = useCallback(() => {
    const snapshot = redoRef.current.pop();
    if (!snapshot) {
      setStatusMsg("Nothing to redo.");
      return;
    }
    historyRef.current.push(captureSnapshot());
    restoreSnapshot(snapshot);
    setHistoryVersion((v) => v + 1);
    setStatusMsg("Redid edit.");
  }, [captureSnapshot, restoreSnapshot]);

  const canUndo = historyVersion >= 0 && historyRef.current.length > 0;
  const canRedo = historyVersion >= 0 && redoRef.current.length > 0;

  const stopStatusTimer = useCallback(() => {
    if (statusStartRef.current !== null) {
      setStatusElapsedMs(performance.now() - statusStartRef.current);
    }
    if (statusTimerRef.current !== null) {
      window.clearInterval(statusTimerRef.current);
      statusTimerRef.current = null;
    }
    statusStartRef.current = null;
  }, []);

  const startStatusTimer = useCallback(() => {
    stopStatusTimer();
    statusStartRef.current = performance.now();
    statusIterationsRef.current = 0;
    statusIterationPaintRef.current = 0;
    setStatusElapsedMs(0);
    setStatusIterations(0);
    statusTimerRef.current = window.setInterval(() => {
      if (statusStartRef.current !== null) {
        setStatusElapsedMs(performance.now() - statusStartRef.current);
      }
    }, 250);
  }, [stopStatusTimer]);

  const setStatusIterationCount = useCallback((count: number) => {
    statusIterationsRef.current = Math.max(0, Math.floor(count));
    setStatusIterations(statusIterationsRef.current);
  }, []);

  const bumpStatusIteration = useCallback(() => {
    statusIterationsRef.current += 1;
    const now = performance.now();
    if (now - statusIterationPaintRef.current > 250) {
      statusIterationPaintRef.current = now;
      setStatusIterations(statusIterationsRef.current);
    }
  }, []);

  const beginDynamicMetrics = useCallback((tileId: string) => {
    activeDynamicTilesRef.current.add(tileId);
    startStatusTimer();
  }, [startStatusTimer]);

  const endDynamicMetrics = useCallback((tileId: string) => {
    activeDynamicTilesRef.current.delete(tileId);
    setStatusIterationCount(statusIterationsRef.current);
    if (activeDynamicTilesRef.current.size === 0) {
      stopStatusTimer();
    }
  }, [setStatusIterationCount, stopStatusTimer]);

  useEffect(() => () => stopStatusTimer(), [stopStatusTimer]);

  /* -------------- Derived helpers -------------- */

  const netlistPreview = useMemo(() => {
    const modeLine =
      analysis.mode === "op"
        ? ".op"
        : analysis.mode === "tran"
          ? `.tran ${analysis.tStop} ${analysis.tStep}`
          : analysis.mode === "ac"
            ? `.ac ${analysis.fStart} ${analysis.fStop} ${analysis.points}`
            : analysis.hbTimeWindow.trim()
              ? `.hb ${analysis.harmonics} ${analysis.hbTimeWindow.trim()}`
              : `.hb ${analysis.harmonics}`;
    const lines = activeLevel.components
      .filter((c) => c.type !== "GND")
      .map((c) => `${c.name} <${c.type}> ${c.value}${c.subtype ? ` [${c.subtype}]` : ""}`);
    return [...lines, modeLine, ".end"].join("\n");
  }, [activeLevel.components, analysis]);

  const handleNetlistTextChange = useCallback(
    (nextText: string): boolean => {
      if (netlistMode !== "netlist") {
        const ok = window.confirm(
          "Editing the netlist will clear the current schematic hierarchy and switch this workspace to raw netlist mode. Continue?"
        );
        if (!ok) return false;
        pushEditorHistory();
        setLevels([
          {
            id: "root",
            title: "Netlist",
            components: [],
            wires: [],
            pins: [],
            parentId: null
          }
        ]);
        setActiveLevelId("root");
        setSelectedIds(new Set());
        setPendingDevice(null);
        setProbeMode(false);
        setGeneratedNetlist("");
        setResponsePreview("");
        setNetlistMode("netlist");
        setStatus("idle");
        setStatusMsg("Schematic cleared; netlist editor is active.");
      }
      setNetlistText(nextText);
      return true;
    },
    [netlistMode, pushEditorHistory]
  );

  /**
   * The probe-node picker lists labels exactly as they appear in the plot
   * (``V(<net>)`` and ``I(<branch>)``). Storing the full label in
   * ``analysis.probeNodes`` lets the waveform filter do a direct string match
   * — otherwise "n1" picks would never line up with "V(n1)" series.
   */
  const availableNodes = useMemo(() => {
    const nodeSet = new Set<string>();
    const branchSet = new Set<string>();
    const lines = (netlistMode === "netlist" ? netlistText : generatedNetlist).split(/\r?\n/);
    for (const raw of lines) {
      const tok = raw.trim().split(/\s+/);
      if (tok.length < 3) continue;
      if (raw.startsWith(".") || raw.startsWith("*")) continue;
      // device name → branch label for V/L/E/H lines (they generate a branch row).
      const dev = tok[0].toUpperCase();
      if (dev.startsWith("V") || dev.startsWith("L") || dev.startsWith("E") || dev.startsWith("H")) {
        branchSet.add(tok[0]);
      }
      // First two tokens after the name are nodes (passive form), 4th/5th for VCVS/VCCS, etc.
      const n1 = tok[1];
      const n2 = tok[2];
      if (n1 && n1 !== "0") nodeSet.add(n1);
      if (n2 && n2 !== "0") nodeSet.add(n2);
    }
    const nodes = Array.from(nodeSet).sort().map((n) => `V(${n})`);
    const branches = Array.from(branchSet).sort().map((b) => `I(${b})`);
    return [...nodes, ...branches];
  }, [generatedNetlist, netlistMode, netlistText]);

  /* -------------- Toolbar / library -------------- */

  const startDragType = useCallback((type: DeviceType) => {
    setPendingDevice(type);
    setPendingTemplate(null);
    setProbeMode(false);
    setWireTool(false);
    canvasHandleRef.current?.focus();
  }, []);

  const pickDevice = useCallback((type: DeviceType | null) => {
    setPendingDevice(type);
    setPendingTemplate(null);
    if (type) {
      setProbeMode(false);
      setWireTool(false);
      canvasHandleRef.current?.focus();
    }
  }, []);

  const toggleWireTool = useCallback(() => {
    setWireTool((v) => !v);
    setPendingDevice(null);
    setProbeMode(false);
    canvasHandleRef.current?.focus();
  }, []);

  const toggleProbeMode = useCallback(() => {
    setProbeMode((v) => !v);
    setPendingDevice(null);
    setWireTool(false);
  }, []);

  const rotateSelection = useCallback(() => {
    if (selectedIds.size === 0) return;
    pushEditorHistory();
    setActiveComponents((cur) =>
      cur.map((component) =>
        selectedIds.has(component.id)
          ? { ...component, rotation: ((component.rotation + 90) % 360) as CanvasComponent["rotation"] }
          : component
      )
    );
    setStatusMsg("Rotated selection.");
  }, [pushEditorHistory, selectedIds, setActiveComponents]);

  const mirrorSelection = useCallback(() => {
    if (selectedIds.size === 0) return;
    pushEditorHistory();
    setActiveComponents((cur) =>
      cur.map((component) =>
        selectedIds.has(component.id) ? { ...component, mirrored: !component.mirrored } : component
      )
    );
    setStatusMsg("Mirrored selection.");
  }, [pushEditorHistory, selectedIds, setActiveComponents]);

  const openHelp = useCallback(() => {
    setOpenMenu(null);
    setHelpOpen(true);
  }, []);

  const deleteSelection = useCallback(() => {
    if (selectedIds.size === 0) return;
    const sel = selectedIds;
    pushEditorHistory();
    setActiveComponents((cur) => cur.filter((c) => !sel.has(c.id)));
    setActiveWires((cur) =>
      cur.filter((w) => {
        if (sel.has(w.id)) return false;
        if (w.start.kind === "pin" && sel.has(w.start.componentId)) return false;
        if (w.end.kind === "pin" && sel.has(w.end.componentId)) return false;
        return true;
      })
    );
    setSelectedIds(new Set());
    setStatusMsg(`Deleted ${sel.size} item${sel.size === 1 ? "" : "s"}.`);
  }, [pushEditorHistory, selectedIds, setActiveComponents, setActiveWires]);

  const selectAll = useCallback(() => {
    const all = new Set<string>();
    activeLevel.components.forEach((c) => all.add(c.id));
    activeLevel.wires.forEach((w) => all.add(w.id));
    setSelectedIds(all);
  }, [activeLevel]);

  const copySelection = useCallback((): ClipboardData | null => {
    const data = extractClipboard(activeLevel.components, activeLevel.wires, selectedIds);
    if (!data) return null;
    setClipboard(data);
    setStatusMsg(`Copied ${data.components.length} component${data.components.length === 1 ? "" : "s"}.`);
    return data;
  }, [activeLevel, selectedIds]);

  /* Keyboard shortcuts are installed after the file/zoom helpers below. */

  /* -------------- Property update -------------- */

  const updateComponent = useCallback(
    (id: string, patch: Partial<CanvasComponent>) => {
      pushEditorHistory();
      editLevel(activeLevelId, (level, all) => {
        const previous = level.components.find((component) => component.id === id);
        if (!previous) return level;
        const components = level.components.map((component) =>
          component.id === id ? { ...component, ...patch } : component
        );
        // A SUBCKT pin list edited by hand (unlinked) or replaced by relinking:
        // move the wire ends to the matching new pins, detach the others.
        const geometry = withEntityPinLayout(previous, all);
        const wires =
          previous.type === "SUBCKT" && patch.pins
            ? remapInstancePinEndpoints(level.wires, id, previous.pins ?? [], patch.pins, (pin) =>
                getPinCoordinates(geometry, pin)
              )
            : level.wires;
        return { ...level, components, wires };
      });
    },
    [activeLevelId, editLevel, pushEditorHistory]
  );

  /* -------------- Hierarchy navigation and transforms -------------- */

  /** Open a level: clears the selection and any tool in progress. */
  const navigateToLevel = useCallback((levelId: string, fit = true) => {
    setActiveLevelId(levelId);
    setSelectedIds(new Set());
    setPendingDevice(null);
    setPendingTemplate(null);
    setWireTool(false);
    setContextMenu(null);
    if (fit) window.setTimeout(() => canvasHandleRef.current?.zoomToFit(), 0);
  }, []);

  /** Double-click on an instance: open its entity (or its properties when unlinked). */
  const descendIntoInstance = useCallback(
    (componentId: string) => {
      const instance = activeLevel.components.find((c) => c.id === componentId);
      const entity = instance?.subcircuitId ? levels.find((l) => l.id === instance.subcircuitId) : undefined;
      if (!instance || !entity || entity.parentId === null) {
        setSelectedIds(new Set([componentId]));
        setPropertiesFocusToken((t) => t + 1);
        setStatusMsg("This SUBCKT instance is not linked to an entity: pick one in its properties.");
        return;
      }
      navigateToLevel(entity.id);
      setStatusMsg(`Opened '${entity.title}' (instance ${instance.name}). Alt+Up goes back up.`);
    },
    [activeLevel, levels, navigateToLevel]
  );

  const ascendLevel = useCallback(() => {
    const parent = parentLevelId(levels, activeLevelId);
    if (!parent) {
      setStatusMsg("Already at the top of the hierarchy.");
      return;
    }
    navigateToLevel(parent);
  }, [activeLevelId, levels, navigateToLevel]);

  const makeSubcircuit = useCallback(() => {
    setContextMenu(null);
    if (netlistMode === "netlist") return;
    try {
      const result = makeSubcircuitFromSelection(levels, activeLevelId, selectedIds);
      pushEditorHistory();
      setLevels(result.levels);
      setSelectedIds(new Set([result.instanceId]));
      const entity = result.levels.find((l) => l.id === result.newLevelId);
      const instance = result.levels
        .find((l) => l.id === activeLevelId)
        ?.components.find((c) => c.id === result.instanceId);
      const internal = result.internalPorts.length
        ? ` Labels ${result.internalPorts.join(", ")} became ports too and stay labelled here.`
        : "";
      setStatus("idle");
      setStatusMsg(
        `Made '${entity?.title}' with ports ${result.ports.join(", ")}; ${instance?.name} replaces the selection. Double-click it to open.${internal}`
      );
    } catch (err) {
      setStatus("error");
      setStatusMsg(err instanceof HierarchyError ? err.message : String(err));
    }
  }, [activeLevelId, levels, netlistMode, pushEditorHistory, selectedIds]);

  const dissolveInstance = useCallback(
    (componentId?: string) => {
      setContextMenu(null);
      const id = componentId ?? (selectedIds.size === 1 ? Array.from(selectedIds)[0] : undefined);
      if (!id) return;
      try {
        const result = dissolveSubcircuit(levels, activeLevelId, id);
        pushEditorHistory();
        setLevels(result.levels);
        setSelectedIds(new Set(result.addedIds));
        setStatus("idle");
        setStatusMsg("Dissolved the subcircuit: its contents now sit on this level (the entity is kept).");
      } catch (err) {
        setStatus("error");
        setStatusMsg(err instanceof HierarchyError ? err.message : String(err));
      }
    },
    [activeLevelId, levels, pushEditorHistory, selectedIds]
  );

  const instantiateEntity = useCallback(
    (entityId: string) => {
      const entity = levels.find((l) => l.id === entityId);
      if (!entity) return;
      if (entityContainsLevel(levels, entityId, activeLevelId)) {
        setStatus("error");
        setStatusMsg(`'${entity.title}' contains the open level '${activeLevel.title}': instancing it here would be recursive.`);
        return;
      }
      const template = instanceTemplate(levels, entityId);
      if (!template) return;
      setPendingDevice("SUBCKT");
      setPendingTemplate(template);
      setProbeMode(false);
      setWireTool(false);
      setStatus("idle");
      setStatusMsg(`Placing an instance of '${entity.title}': click on the canvas, Esc to stop.`);
      canvasHandleRef.current?.focus();
    },
    [activeLevel, activeLevelId, levels]
  );

  const duplicateLevel = useCallback(
    (entityId: string) => {
      try {
        const result = duplicateEntity(levels, entityId);
        pushEditorHistory();
        setLevels(result.levels);
        const copy = result.levels.find((l) => l.id === result.newLevelId);
        setStatus("idle");
        setStatusMsg(`Duplicated as '${copy?.title}'.`);
      } catch (err) {
        setStatus("error");
        setStatusMsg(err instanceof HierarchyError ? err.message : String(err));
      }
    },
    [levels, pushEditorHistory]
  );

  /** Refuse to run a hierarchy the backend would reject; returns true when it is fine. */
  const checkHierarchyBeforeRun = useCallback((): boolean => {
    const issue = firstHierarchyIssue(levels, activeLevelId);
    if (!issue) return true;
    setStatus("error");
    setStatusMsg(issue);
    return false;
  }, [activeLevelId, levels]);

  /* Derived view of the active level: instances carry their entity's pin layout. */
  const canvasComponents = useMemo(
    () => decorateSubcktInstances(activeLevel.components, levels),
    [activeLevel.components, levels]
  );
  const subcktInfo = useMemo(() => {
    const map = new Map<string, SubcktSymbolInfo>();
    for (const [id, info] of subcktDecorations(levels, activeLevelId)) map.set(id, info);
    return map;
  }, [activeLevelId, levels]);
  const canvasPendingTemplate = useMemo(() => {
    if (!pendingTemplate || pendingDevice !== "SUBCKT") return pendingTemplate;
    const ghost = withEntityPinLayout({ ...(pendingTemplate as CanvasComponent), type: "SUBCKT" }, levels);
    return { ...pendingTemplate, pinLayout: ghost.pinLayout };
  }, [levels, pendingDevice, pendingTemplate]);
  const breadcrumb = useMemo(() => breadcrumbPath(levels, activeLevelId), [activeLevelId, levels]);
  const focusOnMount = useCallback((el: HTMLElement | null) => el?.focus(), []);

  /* -------------- Run simulation -------------- */

  const runSimulation = useCallback(async (override?: Partial<AnalysisState>) => {
    if (running) return;
    const effective: AnalysisState = override ? { ...analysis, ...override } : analysis;
    if (override) setAnalysis(effective);
    setOpenMenu(null);
    if (netlistMode !== "netlist" && !checkHierarchyBeforeRun()) return;
    activeDynamicTilesRef.current.clear();
    startStatusTimer();
    setRunning(true);
    setStatus("running");
    setStatusMsg("");
    const schematicPayload =
      netlistMode === "netlist"
        ? null
        : buildSchematicPayload(
            activeLevel.components,
            activeLevel.wires,
            effective,
            levels,
            activeLevel.junctions ?? []
          );
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          netlist_text: netlistMode === "netlist" ? netlistText : "",
          mode: effective.mode,
          options: { ...simulationRequestOptions(effective), output_max_points: MAX_STATIC_PLOT_POINTS },
          schematic: schematicPayload ?? undefined
        })
      });
      const body = await res.json();
      if (!res.ok) {
        setStatus("error");
        setStatusMsg(body?.detail?.message ?? "Simulation failed.");
        setResponsePreview(formatResponsePreview(body));
      } else {
        setStatus("ok");
        const metrics = runMetrics(body?.metadata);
        const done = `${effective.mode.toUpperCase()} complete (${body.status})`;
        setStatusMsg(metrics ? `${done} · ${metrics}` : `${done}.`);
        setStatusIterationCount(responseIterationCount(body, effective.mode));
        if (body?.metadata?.generated_netlist) {
          setGeneratedNetlist(String(body.metadata.generated_netlist));
        } else if (netlistMode === "netlist") {
          setGeneratedNetlist(netlistText);
        }
        setResponsePreview(formatResponsePreview(body));

        // Materialize a result tile.
        if (effective.mode === "tran" && body.waveform) {
          const data = toPlotData(body.waveform, "time (s)", "V", effective.probeNodes);
          addTile({
            title: `.tran — ${effective.tStop}`,
            mode: "static",
            view: "plot",
            data
          });
        } else if (effective.mode === "ac" && body.spectrum) {
          const data = toSpectrumPlot(body.spectrum, { kind: "line", db: true, filterLabels: effective.probeNodes });
          addTile({ title: `.ac sweep`, mode: "static", view: "plot", data });
        } else if (effective.mode === "hb" && body.spectrum) {
          const f0 = Number(body?.metadata?.base_frequency_hz ?? 0);
          const data = toSpectrumPlot(body.spectrum, { kind: "stem", filterLabels: effective.probeNodes });
          addTile({
            title: f0 > 0 ? `.hb spectrum · f0 ${formatCompact(f0)}Hz` : `.hb spectrum`,
            mode: "static",
            view: "plot",
            data,
            w: 520,
            h: 300
          });
        } else if (effective.mode === "hb" && body.waveform) {
          const data = toPlotData(body.waveform, "time (s)", "V", effective.probeNodes);
          addTile({
            title: `.hb reconstruct — ${effective.hbTimeWindow || "window"}`,
            mode: "static",
            view: "plot",
            data,
            w: 520,
            h: 300
          });
        } else if (effective.mode === "op" && body.dc_solution) {
          addTile({
            title: `.op — matrices and operating point`,
            mode: "static",
            view: "op",
            opData: {
              labels: body.labels ?? [],
              values: body.dc_solution ?? [],
              matrices: body.matrices ?? {},
              deviceOperatingPoints: body.metadata?.device_operating_points ?? [],
              filterLabels: effective.probeNodes
            },
            w: 760,
            h: 520
          });
        }
      }
    } catch (err) {
      setStatus("error");
      setStatusMsg(String(err));
    } finally {
      setRunning(false);
      stopStatusTimer();
    }
  }, [analysis, activeLevel, checkHierarchyBeforeRun, levels, netlistMode, netlistText, running, setStatusIterationCount, startStatusTimer, stopStatusTimer]);

  /* -------------- Probe click → open .dyn tile -------------- */

  const onProbePick = useCallback(
    (target: ProbeTarget) => {
      if (netlistMode === "netlist") {
        setStatus("error");
        setStatusMsg("Dynamic probe needs a schematic. Run the edited netlist with Run.");
        return;
      }
      if (!checkHierarchyBeforeRun()) return;
      const payload = buildSchematicPayload(
        activeLevel.components,
        activeLevel.wires,
        analysis,
        levels,
        activeLevel.junctions ?? []
      );
      const speed = parseValue(analysis.dynSpeed);
      const windowSim = parseValue(analysis.dynWindow);
      const tileId = genId("tile");
      const nodeLabel = target.netLabel ?? `${target.componentId}.${target.pin}`;

      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/dyn`);
      const titleSuffix = "∞";
      const tileRecord: TileRecord = {
        id: tileId,
        title: `.dyn ${nodeLabel} @ ${analysis.dynSpeed}s/s · ${titleSuffix}`,
        x: Math.max(40, target.x + 60),
        y: Math.max(40, target.y - 80),
        w: 380,
        h: 240,
        mode: "dyn",
        view: "plot",
        socket: ws,
        probe: target,
        windowSeconds: windowSim
      };
      setTiles((cur) => [...cur, tileRecord]);
      beginDynamicMetrics(tileId);
      setStatus("running");
      setStatusMsg(`Dynamic probe: ${nodeLabel}.`);

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            schematic: payload,
            speed,
            t_stop: analysis.tStop,
            t_step: analysis.tStep,
            window: analysis.dynWindow,
            pin_refs: [{ component_id: target.componentId, pin: target.pin }],
            continuous: true,
            ...simulationRequestOptions(analysis)
          })
        );
      };

      let streamLabels: string[] = [];
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "meta") {
          streamLabels = msg.labels ?? [];
          const metrics = runMetrics(msg);
          setStatusMsg(`Dynamic probe: ${streamLabels.length} series${metrics ? ` · ${metrics}` : "."}`);
        } else if (msg.type === "frame") {
          const rec = tilesRef.current.get(tileId);
          if (rec?.push && streamLabels.length > 0) {
            const vals = (msg.values as number[]).map((v) => Number(v) || 0);
            rec.push(Number(msg.t), vals, streamLabels);
            bumpStatusIteration();
          }
        } else if (msg.type === "loop") {
          // Loop boundaries are accepted for compatibility with older clients.
        } else if (msg.type === "done") {
          const rec = tilesRef.current.get(tileId);
          rec?.finalize?.();
          endDynamicMetrics(tileId);
        } else if (msg.type === "error") {
          setStatus("error");
          setStatusMsg(String(msg.message));
          endDynamicMetrics(tileId);
        }
      };
      ws.onerror = () => {
        setStatus("error");
        setStatusMsg("Realtime stream failed.");
        endDynamicMetrics(tileId);
      };
    },
    [activeLevel, analysis, beginDynamicMetrics, bumpStatusIteration, checkHierarchyBeforeRun, endDynamicMetrics, levels, netlistMode]
  );

  const startCaredNodeDisplay = useCallback(() => {
    if (netlistMode === "netlist") {
      setStatus("error");
      setStatusMsg("Dynamic display needs a schematic. Run the edited netlist with Run.");
      return;
    }
    if (!checkHierarchyBeforeRun()) return;
    const payload = buildSchematicPayload(
      activeLevel.components,
      activeLevel.wires,
      analysis,
      levels,
      activeLevel.junctions ?? []
    );
    const speed = parseValue(analysis.dynSpeed);
    const windowSim = parseValue(analysis.dynWindow);
    const tileId = genId("tile");
    const canvas = document.querySelector(".canvasArea") as HTMLElement | null;
    const tileW = 520;
    const tileH = 280;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/dyn`);
    const titleSuffix = "∞";
    const labelsTitle =
      analysis.probeNodes.length === 0
        ? "all nodes"
        : analysis.probeNodes.length === 1
          ? analysis.probeNodes[0]
          : `${analysis.probeNodes.length} nodes`;
    const tileRecord: TileRecord = {
      id: tileId,
      title: `.dyn ${labelsTitle} @ ${analysis.dynSpeed}s/s · ${titleSuffix}`,
      x: Math.max(40, (canvas?.clientWidth ?? 620) - tileW - 24),
      y: 56,
      w: tileW,
      h: tileH,
      mode: "dyn",
      view: "plot",
      socket: ws,
      windowSeconds: windowSim
    };
    setTiles((cur) => [...cur, tileRecord]);
    beginDynamicMetrics(tileId);
    setStatus("running");
    setStatusMsg(`Dynamic display: ${labelsTitle}.`);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          schematic: payload,
          speed,
          t_stop: analysis.tStop,
          t_step: analysis.tStep,
          window: analysis.dynWindow,
          nodes: analysis.probeNodes,
          continuous: true,
          ...simulationRequestOptions(analysis)
        })
      );
    };

    let streamLabels: string[] = [];
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "meta") {
        streamLabels = msg.labels ?? [];
        const metrics = runMetrics(msg);
        setStatusMsg(`Dynamic display: ${streamLabels.length} series${metrics ? ` · ${metrics}` : "."}`);
      } else if (msg.type === "frame") {
        const rec = tilesRef.current.get(tileId);
        if (rec?.push && streamLabels.length > 0) {
          const vals = (msg.values as number[]).map((v) => Number(v) || 0);
          rec.push(Number(msg.t), vals, streamLabels);
          bumpStatusIteration();
        }
      } else if (msg.type === "done") {
        const rec = tilesRef.current.get(tileId);
        rec?.finalize?.();
        endDynamicMetrics(tileId);
      } else if (msg.type === "loop") {
        // Continuous display loop boundary.
      } else if (msg.type === "error") {
        setStatus("error");
        setStatusMsg(String(msg.message));
        endDynamicMetrics(tileId);
      }
    };
    ws.onerror = () => {
      setStatus("error");
      setStatusMsg("Realtime display failed.");
      endDynamicMetrics(tileId);
    };
  }, [activeLevel, analysis, beginDynamicMetrics, bumpStatusIteration, checkHierarchyBeforeRun, endDynamicMetrics, levels, netlistMode]);

  /* Track tile-id → push/finalize handles without re-renders. */
  const tilesRef = useRef<Map<string, TileRecord>>(new Map());
  useEffect(() => {
    tilesRef.current = new Map(tiles.map((t) => [t.id, t]));
  }, [tiles]);

  /* -------------- Tile layout -------------- */

  function addTile(partial: Omit<TileRecord, "id" | "x" | "y" | "w" | "h"> & Partial<Pick<TileRecord, "x" | "y" | "w" | "h">>) {
    const idx = tileCounterRef.current++;
    const offset = (idx % 6) * 26;
    const tile: TileRecord = {
      id: genId("tile"),
      x: 60 + offset,
      y: 60 + offset,
      w: 420,
      h: 260,
      ...partial
    };
    setTiles((cur) => [...cur, tile]);
  }

  function moveTile(id: string, x: number, y: number) {
    setTiles((cur) => cur.map((t) => (t.id === id ? { ...t, x, y } : t)));
  }
  function resizeTile(id: string, w: number, h: number) {
    setTiles((cur) => cur.map((t) => (t.id === id ? { ...t, w, h } : t)));
  }
  function closeTile(id: string) {
    setTiles((cur) => {
      cur.find((t) => t.id === id)?.socket?.close();
      return cur.filter((t) => t.id !== id);
    });
    endDynamicMetrics(id);
  }

  /* -------------- Probe node list for .tran picker -------------- */

  const addProbeNode = useCallback(
    (label: string) => {
      setAnalysis((cur) =>
        cur.probeNodes.includes(label) ? cur : { ...cur, probeNodes: [...cur.probeNodes, label] }
      );
    },
    []
  );
  const removeProbeNode = useCallback(
    (label: string) => setAnalysis((cur) => ({ ...cur, probeNodes: cur.probeNodes.filter((n) => n !== label) })),
    []
  );
  const clearProbeNodes = useCallback(() => {
    setAnalysis((cur) => ({ ...cur, probeNodes: [] }));
  }, []);
  const addMorOutputNode = useCallback(
    (label: string) => {
      setAnalysis((cur) =>
        cur.morOutputNodes.includes(label) ? cur : { ...cur, morOutputNodes: [...cur.morOutputNodes, label] }
      );
    },
    []
  );
  const removeMorOutputNode = useCallback(
    (label: string) => setAnalysis((cur) => ({ ...cur, morOutputNodes: cur.morOutputNodes.filter((n) => n !== label) })),
    []
  );
  const clearMorOutputNodes = useCallback(() => {
    setAnalysis((cur) => ({ ...cur, morOutputNodes: [] }));
  }, []);
  const resetAnalysisControls = useCallback(() => {
    setAnalysis({ ...defaultAnalysis });
    setStatus("idle");
    setStatusMsg("Simulation controls reset.");
  }, []);

  /* -------------- Presets / demos -------------- */

  function applyPreset(preset: SchematicPreset) {
    pushEditorHistory();
    setWireTool(false);
    setProbeMode(false);
    setPendingDevice(null);
    const rootLevel: SchematicLevel = {
      id: "root",
      title: preset.title,
      components: preset.components.map((c) => ({ ...c })),
      wires: preset.wires.map((w) => ({
        ...w,
        start: { ...w.start } as CanvasWire["start"],
        end: { ...w.end } as CanvasWire["end"]
      })),
      pins: [],
      parentId: null
    };
    const extras: SchematicLevel[] = (preset.extraLevels ?? []).map((lv) => ({
      ...lv,
      components: lv.components.map((c) => ({ ...c })),
      wires: lv.wires.map((w) => ({
        ...w,
        start: { ...w.start } as CanvasWire["start"],
        end: { ...w.end } as CanvasWire["end"]
      })),
      pins: [...lv.pins]
    }));
    setLevels([rootLevel, ...extras]);
    setActiveLevelId("root");
    setAnalysis((cur) => ({ ...defaultAnalysis, ...preset.analysis, numThreads: cur.numThreads }));
    setSelectedIds(new Set());
    setGeneratedNetlist("");
    setNetlistMode("schematic");
    setNetlistText("");
    setResponsePreview("");
    setStatus("idle");
    setStatusMsg(`Preset "${preset.title}" loaded.`);
  }

  /* -------------- File menu -------------- */

  const workspaceHasContent = levels.some((l) => l.components.length > 0 || l.wires.length > 0) || netlistText.trim().length > 0;

  function newWorkspace() {
    setOpenMenu(null);
    if (workspaceHasContent && !window.confirm("Start a new empty schematic? The current workspace stays in the undo history.")) {
      return;
    }
    pushEditorHistory();
    setLevels([{ ...EMPTY_ROOT }]);
    setActiveLevelId("root");
    setAnalysis((cur) => ({ ...defaultAnalysis, numThreads: cur.numThreads }));
    setSelectedIds(new Set());
    setGeneratedNetlist("");
    setNetlistMode("schematic");
    setNetlistText("");
    setResponsePreview("");
    setPendingDevice(null);
    setWireTool(false);
    setProbeMode(false);
    setStatus("idle");
    setStatusMsg("New empty schematic.");
  }

  function saveWorkspace() {
    setOpenMenu(null);
    const root = levels.find((l) => l.parentId === null);
    const text = serializeWorkspace({ levels, activeLevelId, analysis, netlistMode, netlistText });
    downloadText(`${fileSlug(root?.title ?? "schematic")}.mna.json`, text);
    setStatusMsg("Workspace saved as JSON.");
  }

  function openWorkspacePicker() {
    setOpenMenu(null);
    fileInputRef.current?.click();
  }

  async function openWorkspaceFile(file: File) {
    try {
      const parsed = parseWorkspace(await file.text());
      pushEditorHistory();
      setLevels(cloneLevels(parsed.levels));
      setActiveLevelId(parsed.activeLevelId);
      setAnalysis((cur) => ({
        ...defaultAnalysis,
        ...parsed.analysis,
        numThreads: threadSettingOr(parsed.analysis.numThreads, cur.numThreads)
      }));
      setNetlistMode(parsed.netlistMode);
      setNetlistText(parsed.netlistText);
      setGeneratedNetlist("");
      setResponsePreview("");
      setSelectedIds(new Set());
      setPendingDevice(null);
      setWireTool(false);
      setProbeMode(false);
      setStatus("idle");
      setStatusMsg(`Opened "${file.name}".`);
      window.setTimeout(() => canvasHandleRef.current?.zoomToFit(), 0);
    } catch (err) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message : String(err));
    }
  }

  function exportNetlist() {
    setOpenMenu(null);
    const text = netlistMode === "netlist" ? netlistText : generatedNetlist;
    if (!text.trim()) {
      setStatus("error");
      setStatusMsg("Run a simulation once to generate the netlist, then export it.");
      return;
    }
    const root = levels.find((l) => l.parentId === null);
    downloadText(`${fileSlug(root?.title ?? "schematic")}.cir`, text, "text/plain");
    setStatusMsg("Netlist exported.");
  }

  /* -------------- Hierarchy handlers -------------- */

  function addLevel(parentId: string | null) {
    pushEditorHistory();
    const id = genId("level");
    const title = parentId === null ? `Level ${levels.length}` : nextSubcktTitle(levels);
    // A new entity starts with two ports (NODE markers "in" and "out") so an
    // instance of it is usable right away; edit, add or delete them freely.
    const components: CanvasComponent[] =
      parentId === null
        ? []
        : [
            { id: genId("node"), type: "NODE", name: "in", value: "node", x: 120, y: 200, rotation: 0, mirrored: true },
            { id: genId("node"), type: "NODE", name: "out", value: "node", x: 420, y: 200, rotation: 0 }
          ];
    setLevels((cur) => [
      ...cur,
      { id, title, components, wires: [], pins: parentId === null ? [] : ["in", "out"], parentId }
    ]);
    navigateToLevel(id);
  }
  function renameLevel(id: string, title: string) {
    pushEditorHistory();
    updateLevel(id, { title });
  }
  function deleteLevel(id: string) {
    pushEditorHistory();
    // Collect ``id`` plus every descendant via BFS so an entire subtree is
    // dropped in one operation. Without this, deleting a parent would orphan
    // its children, leaving stale entries in the hierarchy tree.
    const doomed = new Set<string>([id]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const l of levels) {
        if (!doomed.has(l.id) && l.parentId && doomed.has(l.parentId)) {
          doomed.add(l.id);
          grew = true;
        }
      }
    }
    setLevels((cur) => {
      const survivors = cur.filter((l) => !doomed.has(l.id));
      if (survivors.length > 0) {
        return survivors;
      }
      return [
        {
          id: "root",
          title: "Top level",
          components: [],
          wires: [],
          pins: [],
          parentId: null
        }
      ];
    });
    if (doomed.has(activeLevelId)) {
      const survivors = levels.filter((l) => !doomed.has(l.id));
      setActiveLevelId(survivors.find((l) => l.parentId === null)?.id ?? survivors[0]?.id ?? "root");
    }
  }

  /* -------------- Copy / paste -------------- */

  const pasteClipboard = useCallback((source: ClipboardData | null) => {
    if (!source || source.components.length === 0) return;
    const clipboard = source;
    pushEditorHistory();
    const idRemap = new Map<string, string>();
    const dx = 40;
    const dy = 40;
    const newComponents: CanvasComponent[] = clipboard.components.map((c) => {
      const newId = genId(c.type.toLowerCase());
      idRemap.set(c.id, newId);
      // Bump trailing-number in name if present so we don't duplicate names.
      const m = c.name.match(/^([A-Za-z_]+)(\d+)$/);
      const newName = m ? `${m[1]}${Number(m[2]) + 1000}` : `${c.name}_copy`;
      return { ...c, id: newId, name: newName, x: c.x + dx, y: c.y + dy };
    });
    const newWires: CanvasWire[] = clipboard.wires.map((w) => {
      const start =
        w.start.kind === "pin"
          ? {
              kind: "pin" as const,
              componentId: idRemap.get(w.start.componentId) ?? w.start.componentId,
              pin: w.start.pin
            }
          : { kind: "point" as const, x: w.start.x + dx, y: w.start.y + dy };
      const end =
        w.end.kind === "pin"
          ? {
              kind: "pin" as const,
              componentId: idRemap.get(w.end.componentId) ?? w.end.componentId,
              pin: w.end.pin
            }
          : { kind: "point" as const, x: w.end.x + dx, y: w.end.y + dy };
      return {
        id: `w-${Math.random().toString(36).slice(2, 8)}`,
        start,
        end
      };
    });
    setActiveComponents((cur) => [...cur, ...newComponents]);
    setActiveWires((cur) => [...cur, ...newWires]);
    // Select the freshly pasted items so a follow-up drag affects only them.
    const sel = new Set<string>();
    newComponents.forEach((c) => sel.add(c.id));
    newWires.forEach((w) => sel.add(w.id));
    setSelectedIds(sel);
  }, [pushEditorHistory, setActiveComponents, setActiveWires]);

  const onPaste = useCallback(() => pasteClipboard(clipboard), [clipboard, pasteClipboard]);

  /** LTspice F6: copy the selection and drop the copy offset right away. */
  const duplicateSelection = useCallback(() => {
    const data = copySelection();
    if (data) pasteClipboard(data);
  }, [copySelection, pasteClipboard]);

  /* -------------- Keyboard shortcuts -------------- */

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;
      const key = event.key.toLowerCase();
      const ctrl = event.ctrlKey || event.metaKey;

      if (event.key === "Escape") {
        if (contextMenu) setContextMenu(null);
        else if (openMenu) setOpenMenu(null);
        else if (helpOpen) setHelpOpen(false);
        else if (pendingDevice) {
          setPendingDevice(null);
          setPendingTemplate(null);
          setStatusMsg("Component placement finished.");
        } else if (wireTool) setWireTool(false);
        else if (probeMode) setProbeMode(false);
        else return;
        event.preventDefault();
        return;
      }

      // Alt+Up: go to the parent level in the hierarchy tree.
      if (event.altKey && !ctrl && event.key === "ArrowUp") {
        ascendLevel();
        event.preventDefault();
        return;
      }

      // Function keys follow LTspice: F1 help, F3 wire, F4 net label, F6 duplicate, F9 undo.
      if (event.key === "F1") {
        openHelp();
        event.preventDefault();
        return;
      }
      if (event.key === "F3") {
        toggleWireTool();
        event.preventDefault();
        return;
      }
      if (event.key === "F4") {
        pickDevice("NODE");
        event.preventDefault();
        return;
      }
      if (event.key === "F6") {
        duplicateSelection();
        event.preventDefault();
        return;
      }
      if (event.key === "F9") {
        if (event.shiftKey) redoEditor();
        else undoEditor();
        event.preventDefault();
        return;
      }

      if (!ctrl) {
        if (key === "w" && !pendingDevice) {
          toggleWireTool();
          event.preventDefault();
        } else if (key === "r" && !pendingDevice) {
          rotateSelection();
          event.preventDefault();
        } else if ((event.key === "Delete" || event.key === "Backspace") && selectedIds.size > 0) {
          deleteSelection();
          event.preventDefault();
        } else if (event.key === " " && !pendingDevice) {
          canvasHandleRef.current?.zoomToFit();
          event.preventDefault();
        }
        return;
      }

      switch (key) {
        case "z":
          if (event.shiftKey) redoEditor();
          else undoEditor();
          break;
        case "y":
          redoEditor();
          break;
        case "r":
          if (!pendingDevice) rotateSelection();
          break;
        case "e":
          if (!pendingDevice) mirrorSelection();
          break;
        case "d":
          duplicateSelection();
          break;
        case "c":
          copySelection();
          break;
        case "v":
          onPaste();
          break;
        case "a":
          selectAll();
          break;
        case "s":
          saveWorkspace();
          break;
        case "o":
          openWorkspacePicker();
          break;
        case "g":
          if (!event.shiftKey) return;
          makeSubcircuit();
          break;
        case "=":
        case "+":
          canvasHandleRef.current?.zoomIn();
          break;
        case "-":
          canvasHandleRef.current?.zoomOut();
          break;
        case "0":
          canvasHandleRef.current?.zoomToFit();
          break;
        case "enter":
          void runSimulation();
          break;
        default:
          return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  /* -------------- Render -------------- */

  const selectedComponent =
    selectedIds.size === 1
      ? activeLevel.components.find((c) => selectedIds.has(c.id)) ?? null
      : null;
  const hasSelection = selectedIds.size > 0;
  const selectedInstance = selectedComponent?.type === "SUBCKT" ? selectedComponent : null;
  const contextTarget = contextMenu?.componentId
    ? activeLevel.components.find((c) => c.id === contextMenu.componentId) ?? null
    : null;
  const contextEntries: MenuEntry[] = [
    { kind: "item", label: "Make subcircuit", shortcut: "Ctrl+Shift+G", onSelect: makeSubcircuit, disabled: !hasSelection || netlistMode === "netlist" },
    ...(contextTarget?.type === "SUBCKT"
      ? ([
          { kind: "item", label: "Open subcircuit", shortcut: "Double-click", onSelect: () => descendIntoInstance(contextTarget.id) },
          { kind: "item", label: "Dissolve subcircuit", onSelect: () => dissolveInstance(contextTarget.id) }
        ] as MenuEntry[])
      : []),
    { kind: "item", label: "Up one level", shortcut: "Alt+Up", onSelect: ascendLevel, disabled: !parentLevelId(levels, activeLevelId) },
    { kind: "separator" },
    {
      kind: "item",
      label: "Properties",
      shortcut: contextTarget?.type === "SUBCKT" ? undefined : "Double-click",
      onSelect: () => setPropertiesFocusToken((t) => t + 1),
      disabled: !contextTarget
    },
    { kind: "item", label: "Copy", shortcut: "Ctrl+C", onSelect: () => void copySelection(), disabled: !hasSelection },
    { kind: "item", label: "Duplicate", shortcut: "Ctrl+D", onSelect: duplicateSelection, disabled: !hasSelection },
    { kind: "item", label: "Rotate", shortcut: "R", onSelect: rotateSelection, disabled: !hasSelection },
    { kind: "item", label: "Mirror", shortcut: "Ctrl+E", onSelect: mirrorSelection, disabled: !hasSelection },
    { kind: "item", label: "Delete", shortcut: "Del", onSelect: deleteSelection, disabled: !hasSelection }
  ];

  const fileMenu: MenuEntry[] = [
    { kind: "item", label: "New schematic", shortcut: "", onSelect: newWorkspace },
    { kind: "item", label: "Open workspace…", shortcut: "Ctrl+O", onSelect: openWorkspacePicker },
    { kind: "item", label: "Save workspace as JSON", shortcut: "Ctrl+S", onSelect: saveWorkspace },
    { kind: "separator" },
    { kind: "item", label: "Export SPICE netlist (.cir)", onSelect: exportNetlist }
  ];
  const editMenu: MenuEntry[] = [
    { kind: "item", label: "Undo", shortcut: "Ctrl+Z", onSelect: undoEditor, disabled: !canUndo },
    { kind: "item", label: "Redo", shortcut: "Ctrl+Y", onSelect: redoEditor, disabled: !canRedo },
    { kind: "separator" },
    { kind: "item", label: "Copy", shortcut: "Ctrl+C", onSelect: () => void copySelection(), disabled: !hasSelection },
    { kind: "item", label: "Paste", shortcut: "Ctrl+V", onSelect: onPaste, disabled: !clipboard },
    { kind: "item", label: "Duplicate", shortcut: "Ctrl+D / F6", onSelect: duplicateSelection, disabled: !hasSelection },
    { kind: "item", label: "Delete", shortcut: "Del", onSelect: deleteSelection, disabled: !hasSelection },
    { kind: "item", label: "Select all", shortcut: "Ctrl+A", onSelect: selectAll },
    { kind: "separator" },
    { kind: "item", label: "Rotate", shortcut: "Ctrl+R / R", onSelect: rotateSelection, disabled: !hasSelection },
    { kind: "item", label: "Mirror", shortcut: "Ctrl+E", onSelect: mirrorSelection, disabled: !hasSelection },
    { kind: "separator" },
    { kind: "item", label: "Wire tool", shortcut: "W / F3", onSelect: toggleWireTool, checked: wireTool },
    { kind: "item", label: "Place net label (NODE)", shortcut: "F4", onSelect: () => pickDevice("NODE"), checked: pendingDevice === "NODE" },
    { kind: "separator" },
    { kind: "item", label: "Make subcircuit", shortcut: "Ctrl+Shift+G", onSelect: makeSubcircuit, disabled: !hasSelection || netlistMode === "netlist" },
    { kind: "item", label: "Open subcircuit", shortcut: "Double-click", onSelect: () => selectedInstance && descendIntoInstance(selectedInstance.id), disabled: !selectedInstance },
    { kind: "item", label: "Dissolve subcircuit", onSelect: () => selectedInstance && dissolveInstance(selectedInstance.id), disabled: !selectedInstance },
    { kind: "item", label: "Up one level", shortcut: "Alt+Up", onSelect: ascendLevel, disabled: !parentLevelId(levels, activeLevelId) }
  ];
  const viewMenu: MenuEntry[] = [
    { kind: "item", label: "Zoom in", shortcut: "Ctrl + =", onSelect: () => canvasHandleRef.current?.zoomIn() },
    { kind: "item", label: "Zoom out", shortcut: "Ctrl + −", onSelect: () => canvasHandleRef.current?.zoomOut() },
    { kind: "item", label: "Zoom to fit", shortcut: "Space / Ctrl+0", onSelect: () => canvasHandleRef.current?.zoomToFit() },
    { kind: "item", label: "Actual size (100%)", onSelect: () => canvasHandleRef.current?.resetZoom() },
    { kind: "separator" },
    { kind: "item", label: "Show grid", onSelect: () => setShowGrid((v) => !v), checked: showGrid },
    { kind: "item", label: "Show auto net names", onSelect: () => setShowNetNames((v) => !v), checked: showNetNames },
    { kind: "separator" },
    { kind: "item", label: "Close all result tiles", onSelect: () => tiles.forEach((t) => closeTile(t.id)), disabled: tiles.length === 0 }
  ];
  const simulateMenu: MenuEntry[] = [
    { kind: "item", label: "Run .op (operating point)", onSelect: () => void runSimulation({ mode: "op" }), disabled: running },
    { kind: "item", label: "Run .tran (transient)", onSelect: () => void runSimulation({ mode: "tran" }), disabled: running },
    { kind: "item", label: "Run .ac (frequency sweep)", onSelect: () => void runSimulation({ mode: "ac" }), disabled: running },
    { kind: "item", label: "Run .hb (harmonic balance)", onSelect: () => void runSimulation({ mode: "hb" }), disabled: running },
    { kind: "separator" },
    { kind: "item", label: "Run current mode", shortcut: "Ctrl+Enter", onSelect: () => void runSimulation(), disabled: running },
    { kind: "item", label: "Start dynamic display", onSelect: () => { setOpenMenu(null); startCaredNodeDisplay(); }, disabled: running },
    { kind: "item", label: "Probe mode", onSelect: toggleProbeMode, checked: probeMode },
    { kind: "separator" },
    { kind: "item", label: "Reset simulation controls", onSelect: resetAnalysisControls }
  ];

  const renderMenu = (id: MenuId, title: string, entries: MenuEntry[]) => (
    <div className="menubar__dropdown" key={id}>
      <button
        className={`menubar__item ${openMenu === id ? "open" : ""}`}
        onClick={() => setOpenMenu((cur) => (cur === id ? null : id))}
        onMouseEnter={() => {
          if (openMenu && openMenu !== id) setOpenMenu(id);
        }}
        aria-haspopup="menu"
        aria-expanded={openMenu === id}
      >
        {title}
      </button>
      {openMenu === id ? (
        <div className="menubar__dropdownPanel" role="menu">
          {entries.map((entry, index) =>
            entry.kind === "separator" ? (
              <div key={`sep-${index}`} className="menuRow__sep" />
            ) : (
              <button
                key={entry.label}
                className="menubar__dropdownItem menuRow"
                role="menuitem"
                disabled={entry.disabled}
                onClick={() => {
                  setOpenMenu(null);
                  entry.onSelect();
                }}
              >
                <span className="menuRow__check">{entry.checked ? "✓" : ""}</span>
                <span className="menuRow__label">{entry.label}</span>
                {entry.shortcut ? <span className="menuRow__shortcut">{entry.shortcut}</span> : null}
              </button>
            )
          )}
        </div>
      ) : null}
    </div>
  );

  return (
    <div className="app">
      {openMenu ? <div className="menuBackdrop" onMouseDown={() => setOpenMenu(null)} /> : null}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        className="srOnly"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void openWorkspaceFile(file);
          e.target.value = "";
        }}
      />
      <div className="menubar">
        <div className="menubar__brand">
          <span className="dot" /> MNA Simulator
        </div>
        <div className="menubar__menu">
          {renderMenu("file", "File", fileMenu)}
          {renderMenu("edit", "Edit", editMenu)}
          {renderMenu("view", "View", viewMenu)}
          {renderMenu("simulate", "Simulate", simulateMenu)}
          <div className="menubar__dropdown">
            <button
              className={`menubar__item ${openMenu === "demo" ? "open" : ""}`}
              onClick={() => setOpenMenu((cur) => (cur === "demo" ? null : "demo"))}
              onMouseEnter={() => {
                if (openMenu && openMenu !== "demo") setOpenMenu("demo");
              }}
              aria-haspopup="menu"
              aria-expanded={openMenu === "demo"}
            >
              Demo
            </button>
            {openMenu === "demo" ? (
              <div className="menubar__dropdownPanel" role="menu">
                {DEMO_MENU_GROUPS.map((group) => (
                  <div key={group.title} className="menubar__dropdownGroup">
                    <div className="menubar__dropdownHeading">{group.title}</div>
                    {group.presets.map((d) => (
                      <button
                        key={d.id}
                        className="menubar__dropdownItem"
                        role="menuitem"
                        onClick={() => {
                          applyPreset(d);
                          setOpenMenu(null);
                          window.setTimeout(() => canvasHandleRef.current?.zoomToFit(), 0);
                        }}
                      >
                        <strong>{d.title}</strong>
                        <span className="menubar__dropdownDesc">{d.description}</span>
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
          <button className="menubar__item" onClick={openHelp}>Help</button>
        </div>
        <div className="menubar__spacer" />
        <div className="menubar__status">
          <span className={`statusDot ${status}`} />
          <span>{statusMsg || (status === "idle" ? "Ready" : status)}</span>
          <span className="menubar__metric">t {formatElapsed(statusElapsedMs)}</span>
          <span className="menubar__metric">iter {formatCount(statusIterations)}</span>
        </div>
      </div>

      <Toolbar
        pendingDevice={pendingDevice}
        onPickDevice={pickDevice}
        wireTool={wireTool}
        onToggleWireTool={toggleWireTool}
        onToggleProbe={toggleProbeMode}
        probeMode={probeMode}
        onStartDisplay={startCaredNodeDisplay}
        onRun={() => void runSimulation()}
        analysisMode={analysis.mode}
        onUndo={undoEditor}
        onRedo={redoEditor}
        canUndo={canUndo}
        canRedo={canRedo}
        onRotate={rotateSelection}
        onMirror={mirrorSelection}
        onDelete={deleteSelection}
        hasSelection={hasSelection}
        onZoomToFit={() => canvasHandleRef.current?.zoomToFit()}
        onOpenHelp={openHelp}
        running={running}
      />

      {helpOpen ? (
        <div className="modalBackdrop" onMouseDown={() => setHelpOpen(false)}>
          <section
            className="modal helpDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-dialog-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="helpDialog__header">
              <div>
                <h2 id="help-dialog-title">MNA Simulation User Guide</h2>
                <p>
                  A practical map for drawing circuits, running analyses, reading results, and using the larger demos.
                </p>
              </div>
              <button className="helpDialog__close" onClick={() => setHelpOpen(false)} aria-label="Close help">
                ×
              </button>
            </div>
            <div className="helpDialog__body">
              {USER_GUIDE_SECTIONS.map((section) => (
                <section key={section.title} className="helpSection">
                  <h3>{section.title}</h3>
                  <ul>
                    {section.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
            <div className="modalActions">
              <button className="ghostBtn" onClick={() => setHelpOpen(false)}>Close</button>
            </div>
          </section>
        </div>
      ) : null}

      <LeftPane
        analysis={analysis}
        onAnalysisChange={(patch) => setAnalysis((cur) => ({ ...cur, ...patch }))}
        threadBudgetTotal={threadBudgetTotal}
        onRun={() => void runSimulation()}
        running={running}
        availableNodes={availableNodes}
        onAddProbeNode={addProbeNode}
        onRemoveProbeNode={removeProbeNode}
        onClearProbeNodes={clearProbeNodes}
        onAddMorOutputNode={addMorOutputNode}
        onRemoveMorOutputNode={removeMorOutputNode}
        onClearMorOutputNodes={clearMorOutputNodes}
        onResetAnalysis={resetAnalysisControls}
        levels={levels}
        activeLevelId={activeLevelId}
        onSelectLevel={(id) => navigateToLevel(id)}
        onAddLevel={addLevel}
        onRenameLevel={renameLevel}
        onDeleteLevel={deleteLevel}
        onInstantiateLevel={instantiateEntity}
        onDuplicateLevel={duplicateLevel}
        onOpenEntity={(id) => navigateToLevel(id)}
        libraryComponents={LIBRARY_ITEMS}
        onStartDrag={startDragType}
        selectedComponent={selectedComponent}
        onUpdateComponent={updateComponent}
        propertiesFocusToken={propertiesFocusToken}
        netlistPreview={netlistPreview}
        generatedNetlist={generatedNetlist}
        netlistMode={netlistMode}
        netlistText={netlistText}
        onNetlistTextChange={handleNetlistTextChange}
      />

      <div className="canvasArea" style={{ position: "relative" }}>
        <nav className="breadcrumbBar" aria-label="Hierarchy path">
          {breadcrumb.map((level, index) => {
            const last = index === breadcrumb.length - 1;
            return (
              <span key={level.id} className="breadcrumbBar__item">
                {index > 0 ? <span className="breadcrumbBar__sep">›</span> : null}
                {last ? (
                  <span className="breadcrumbBar__current" aria-current="page">{level.title}</span>
                ) : (
                  <button className="breadcrumbBar__link" onClick={() => navigateToLevel(level.id)}>
                    {level.title}
                  </button>
                )}
              </span>
            );
          })}
          {breadcrumb.length > 1 ? (
            <button className="breadcrumbBar__up" onClick={ascendLevel} title="Up one level (Alt+Up)">
              ↑ Up
            </button>
          ) : null}
          {activeLevel.parentId !== null ? (
            <span className="breadcrumbBar__ports" title="Ports of this entity (its NODE markers)">
              ports: {entityPorts(activeLevel).join(", ") || "none"}
            </span>
          ) : null}
        </nav>
        <div className="canvasViewport">
          <SchematicCanvas
            ref={canvasHandleRef}
            components={canvasComponents}
            wires={activeLevel.wires}
            junctions={activeLevel.junctions}
            selectedIds={selectedIds}
            pendingDevice={pendingDevice}
            pendingTemplate={canvasPendingTemplate}
            probeMode={probeMode}
            wireTool={wireTool}
            showGrid={showGrid}
            showNetNames={showNetNames}
            subcktInfo={subcktInfo}
            onSetComponents={setActiveComponents}
            onSetWires={setActiveWires}
            onSelect={setSelectedIds}
            onPendingResolved={() => {
              setPendingDevice(null);
              setPendingTemplate(null);
            }}
            onWireToolResolved={() => setWireTool(false)}
            onBeforeEdit={pushEditorHistory}
            onProbePick={onProbePick}
            onCopy={() => void copySelection()}
            onPaste={onPaste}
            onDuplicate={duplicateSelection}
            onOpenProperties={() => setPropertiesFocusToken((t) => t + 1)}
            onDescend={descendIntoInstance}
            onContextMenu={({ clientX, clientY, componentId }) => setContextMenu({ x: clientX, y: clientY, componentId })}
            onZoomChange={setZoomScale}
            onCursorChange={(pos) => cursorStore.current?.set(pos)}
          />
        </div>
        {contextMenu ? (
          <>
            <div
              className="menuBackdrop"
              onMouseDown={() => setContextMenu(null)}
              onContextMenu={(e) => {
                e.preventDefault();
                setContextMenu(null);
              }}
            />
            <div
              className="contextMenu menubar__dropdownPanel"
              role="menu"
              tabIndex={-1}
              ref={focusOnMount}
              style={{
                left: Math.max(4, Math.min(contextMenu.x, window.innerWidth - 270)),
                top: Math.max(4, Math.min(contextMenu.y, window.innerHeight - 380))
              }}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  e.stopPropagation();
                  setContextMenu(null);
                }
              }}
            >
              {contextEntries.map((entry, index) =>
                entry.kind === "separator" ? (
                  <div key={`sep-${index}`} className="menuRow__sep" />
                ) : (
                  <button
                    key={entry.label}
                    className="menubar__dropdownItem menuRow"
                    role="menuitem"
                    disabled={entry.disabled}
                    onClick={() => {
                      setContextMenu(null);
                      entry.onSelect();
                    }}
                  >
                    <span className="menuRow__check" />
                    <span className="menuRow__label">{entry.label}</span>
                    {entry.shortcut ? <span className="menuRow__shortcut">{entry.shortcut}</span> : null}
                  </button>
                )
              )}
            </div>
          </>
        ) : null}
        <div className="tileLayer">
          {tiles.map((tile) => (
            tile.view === "op" && tile.opData ? (
              <ResultTile
                key={tile.id}
                id={tile.id}
                title={tile.title}
                x={tile.x}
                y={tile.y}
                w={tile.w}
                h={tile.h}
                opData={tile.opData}
                onMove={moveTile}
                onResize={resizeTile}
                onClose={closeTile}
              />
            ) : (
              <PlotTile
                key={tile.id}
                id={tile.id}
                title={tile.title}
                x={tile.x}
                y={tile.y}
                w={tile.w}
                h={tile.h}
                mode={tile.mode}
                data={tile.data}
                windowSeconds={tile.windowSeconds}
                onMount={
                  tile.mode === "dyn"
                    ? (push, finalize) => {
                        setTiles((cur) =>
                          cur.map((t) => (t.id === tile.id ? { ...t, push, finalize } : t))
                        );
                      }
                    : undefined
                }
                onMove={moveTile}
                onResize={resizeTile}
                onClose={closeTile}
              />
            )
          ))}
        </div>
      </div>

      <div className="statusbar">
        <span className="statusbar__slot">Level: {activeLevel.title}</span>
        <span className="statusbar__slot">
          {activeLevel.components.length} comp · {activeLevel.wires.length} wires
          {selectedIds.size > 0 ? ` · ${selectedIds.size} selected` : ""}
        </span>
        {cursorStore.current ? <CursorReadout store={cursorStore.current} /> : null}
        <span className="statusbar__slot statusbar__slot--mono" style={{ minWidth: 70 }} title="Zoom (Space = fit)">
          {Math.round(zoomScale * 100)}%
        </span>
        <span className="statusbar__slot">
          {wireTool ? "Wire tool" : probeMode ? "Probe" : pendingDevice ? `Placing ${pendingDevice}` : netlistMode === "netlist" ? "Netlist mode" : "Select"}
        </span>
        <span style={{ flex: 1 }} />
        <span className="statusbar__slot statusbar__hint">
          W wire · R rotate · Ctrl+E mirror · Ctrl+D duplicate · Ctrl+Z/Y undo/redo · Space fit · right-drag pan · right-click menu · double-click edit / open subcircuit · Ctrl+Shift+G make subcircuit · Alt+↑ up
        </span>
      </div>
    </div>
  );
}

/* ---------- helpers ---------- */

function parseValue(raw: string): number {
  if (!raw) return 1;
  const m = raw.match(/^\s*([-+]?[0-9]*\.?[0-9]+)([a-zA-Zµ]*)\s*$/);
  if (!m) return Number(raw) || 1;
  const base = Number(m[1]);
  const suf = (m[2] || "").toLowerCase();
  const table: Record<string, number> = {
    "": 1,
    f: 1e-15,
    p: 1e-12,
    n: 1e-9,
    u: 1e-6,
    µ: 1e-6,
    m: 1e-3,
    k: 1e3,
    meg: 1e6,
    g: 1e9
  };
  if (suf in table) return base * table[suf];
  return base;
}

function formatElapsed(ms: number): string {
  const seconds = Math.max(0, ms / 1000);
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

/** "2 threads · 412 ms" from run metadata (threads_used, elapsed_ms); empty when absent. */
function runMetrics(metadata: Record<string, unknown> | null | undefined): string {
  const parts: string[] = [];
  const threads = Number(metadata?.threads_used);
  if (Number.isInteger(threads) && threads > 0) parts.push(`${threads} thread${threads === 1 ? "" : "s"}`);
  const elapsedMs = metadata?.elapsed_ms;
  if (typeof elapsedMs === "number" && Number.isFinite(elapsedMs) && elapsedMs >= 0) {
    parts.push(elapsedMs < 1000 ? `${elapsedMs < 10 ? elapsedMs.toFixed(1) : elapsedMs.toFixed(0)} ms` : `${(elapsedMs / 1000).toFixed(2)} s`);
  }
  return parts.join(" · ");
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(value);
}

function sampledIndexes(length: number, maxPoints: number): number[] | null {
  if (length <= maxPoints) return null;
  const stride = Math.max(1, Math.ceil(length / maxPoints));
  const indexes: number[] = [];
  for (let i = 0; i < length; i += stride) indexes.push(i);
  if (indexes[indexes.length - 1] !== length - 1) indexes.push(length - 1);
  return indexes;
}

function responseIterationCount(body: Record<string, any>, mode: string): number {
  const krylovIterations = Number(body?.metadata?.krylov_iterations);
  if (Number.isFinite(krylovIterations) && krylovIterations > 0) {
    return krylovIterations;
  }
  const decimation = body?.metadata?.output_decimation;
  if (mode === "tran" && body?.waveform?.time) {
    return Number(decimation?.original_points ?? body.waveform.time.length ?? 0);
  }
  if (body?.waveform?.time) return Number(body.waveform.time.length ?? 0);
  if (body?.spectrum?.frequencies) return Number(body.spectrum.frequencies.length ?? 0);
  if (body?.dc_solution) return Number(body.dc_solution.length ?? 0);
  return 0;
}

function nestedValueCount(values: unknown): number {
  if (!Array.isArray(values)) return 0;
  if (values.length === 0) return 0;
  if (Array.isArray(values[0])) {
    return (values as unknown[][]).reduce((sum, row) => sum + row.length, 0);
  }
  return values.length;
}

function compactSeriesPayload(payload: unknown, axisKey: "time" | "frequencies"): unknown {
  if (!payload || typeof payload !== "object") return payload;
  const record = payload as Record<string, unknown>;
  const axis = Array.isArray(record[axisKey]) ? (record[axisKey] as unknown[]) : [];
  const valueKey = record.values !== undefined ? "values" : "magnitudes";
  const values = record[valueKey];
  const totalValues = nestedValueCount(values);
  if (axis.length + totalValues <= MAX_RESPONSE_PREVIEW_VALUES) return payload;
  const valueRows = Array.isArray(values) ? values.length : 0;
  const valueCols = valueRows > 0 && Array.isArray((values as unknown[])[0]) ? ((values as unknown[][])[0]?.length ?? 0) : valueRows;
  return {
    [axisKey]: {
      count: axis.length,
      first: axis[0],
      last: axis[axis.length - 1]
    },
    [valueKey]: {
      shape: [valueRows, valueCols],
      omitted: totalValues
    },
    labels: record.labels
  };
}

function formatResponsePreview(body: unknown): string {
  if (!body || typeof body !== "object") {
    return JSON.stringify(body, null, 2);
  }
  const record = body as Record<string, unknown>;
  const preview = {
    ...record,
    waveform: compactSeriesPayload(record.waveform, "time"),
    spectrum: compactSeriesPayload(record.spectrum, "frequencies")
  };
  return JSON.stringify(preview, null, 2);
}

function toPlotData(
  waveform: { time: number[]; values: number[][]; labels: string[] },
  xLabel: string,
  yLabel: string,
  filterLabels: string[] = []
): PlotData {
  const t = waveform.time ?? [];
  const vs = waveform.values ?? [];
  const labels = waveform.labels ?? [];
  const wanted = filterLabels.length > 0 ? new Set(filterLabels) : null;
  const labelIndexes = labels
    .map((label, index) => ({ label, index }))
    .filter((entry) => !wanted || wanted.has(entry.label));
  const sample = sampledIndexes(t.length, MAX_STATIC_PLOT_POINTS);
  const x = sample ? sample.map((i) => Number(t[i] ?? 0)) : t;
  const rows = vs.length;
  const cols = rows > 0 ? (Array.isArray(vs[0]) ? vs[0].length : 0) : 0;
  const isRowMajor = rows === t.length && cols === labels.length;
  let series: { label: string; values: number[] }[];
  if (isRowMajor) {
    series = labelIndexes.map(({ label, index }) => ({
      label,
      values: sample
        ? sample.map((i) => Number((vs[i] as number[] | undefined)?.[index] ?? 0))
        : vs.map((row) => Number((row as number[])[index] ?? 0))
    }));
  } else {
    series = labelIndexes.map(({ label, index }) => ({
      label,
      values: sample
        ? sample.map((i) => Number((vs[index] as number[] | undefined)?.[i] ?? 0))
        : (vs[index] as number[] | undefined)?.map((v) => Number(v)) ?? []
    }));
  }
  return { x, series, xLabel, yLabel };
}

function toSpectrumPlot(spectrum: {
  frequencies: number[];
  magnitudes: number[][] | number[];
  labels: string[];
}, options: { kind?: PlotData["kind"]; db?: boolean; filterLabels?: string[] } = {}): PlotData {
  const kind = options.kind ?? "line";
  const freq = spectrum.frequencies ?? [];
  const mag = spectrum.magnitudes ?? [];
  const labels = spectrum.labels ?? [];
  const isMatrix = Array.isArray(mag[0]);
  let series: { label: string; values: number[] }[];
  if (isMatrix) {
    const matrix = mag as number[][];
    const rows = matrix.length;
    const cols = rows > 0 && Array.isArray(matrix[0]) ? matrix[0].length : 0;
    const labelRows = rows === labels.length && cols === freq.length;
    const pointRows = rows === freq.length && cols === labels.length;
    if (labelRows) {
      series = labels.map((lbl, j) => ({
        label: lbl,
        values: (matrix[j] ?? []).map((v) => Number(v))
      }));
    } else if (pointRows) {
      series = labels.map((lbl, j) => ({
        label: lbl,
        values: matrix.map((row) => Number(row[j] ?? 0))
      }));
    } else {
      series = labels.map((lbl, j) => ({
        label: lbl,
        values: (matrix[j] ?? []).map((v) => Number(v))
      }));
    }
  } else {
    series = [{ label: labels[0] ?? "|H|", values: (mag as number[]).map((v) => Number(v)) }];
  }
  if (options.filterLabels && options.filterLabels.length > 0) {
    const wanted = new Set(options.filterLabels);
    series = series.filter((s) => wanted.has(s.label));
  }
  if (options.db) {
    series = series.map((s) => ({
      ...s,
      values: s.values.map((v) => 20 * Math.log10(Math.max(Math.abs(v), 1e-15)))
    }));
  }
  return {
    x: freq,
    series,
    xLabel: "frequency (Hz)",
    yLabel: options.db ? "magnitude (dB)" : "magnitude",
    kind,
    logX: kind === "line",
    title: kind === "stem" ? "Harmonic magnitudes" : options.db ? "AC magnitude" : undefined
  };
}

function opResultPlotData(labels: string[], values: number[], filterLabels: string[] = []): PlotData {
  const wanted = new Set(filterLabels);
  const pairs = labels
    .map((label, i) => ({ label, value: Number(values[i] ?? 0) }))
    .filter((entry) => filterLabels.length === 0 || wanted.has(entry.label));
  const x = pairs.map((_, i) => i);
  return {
    x,
    series: [{ label: "DC", values: pairs.map((entry) => entry.value) }],
    xLabel: "",
    yLabel: "value",
    kind: "bar",
    xTickLabels: pairs.map((entry) => entry.label),
    title: "DC operating point"
  };
}

function formatCompact(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(2).replace(/\.?0+$/, "")}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2).replace(/\.?0+$/, "")}k`;
  return value.toFixed(2).replace(/\.?0+$/, "");
}
