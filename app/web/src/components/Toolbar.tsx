import { ReactNode, useEffect, useRef } from "react";
import { BasicAnalysisMode, DeviceType, DEVICE_LABELS } from "../lib/schematic";
import { drawPalettePreview } from "../lib/symbols";

type Props = {
  pendingDevice: DeviceType | null;
  onPickDevice: (type: DeviceType | null) => void;
  wireTool: boolean;
  onToggleWireTool: () => void;
  onToggleProbe: () => void;
  probeMode: boolean;
  onStartDisplay: () => void;
  onRun: () => void;
  analysisMode: BasicAnalysisMode;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onRotate: () => void;
  onMirror: () => void;
  onDelete: () => void;
  hasSelection: boolean;
  onZoomToFit: () => void;
  onOpenHelp: () => void;
  running: boolean;
};

const REGULAR: DeviceType[] = ["R", "C", "L", "V", "I", "D", "GND"];
const TRANSISTORS: DeviceType[] = ["QNPN", "QPNP", "NMOS", "PMOS"];
const CONTROLLED: DeviceType[] = ["VCVS", "VCCS", "CCCS", "CCVS"];
const STRUCTURE: DeviceType[] = ["SUBCKT", "NODE", "LABEL"];

export function Toolbar(props: Props) {
  const pick = (t: DeviceType) => props.onPickDevice(props.pendingDevice === t ? null : t);
  const deviceGroup = (title: string, types: DeviceType[]) => (
    <div className="toolGroup" role="group" aria-label={title}>
      {types.map((t) => (
        <DeviceButton key={t} type={t} active={props.pendingDevice === t} onClick={() => pick(t)} />
      ))}
    </div>
  );
  return (
    <div className="toolbar">
      {deviceGroup("Components", REGULAR)}
      {deviceGroup("Transistors", TRANSISTORS)}
      {deviceGroup("Controlled sources", CONTROLLED)}
      {deviceGroup("Hierarchy and nets", STRUCTURE)}

      <div className="toolGroup" role="group" aria-label="Edit">
        <ToolButton label="Wire (W / F3)" active={props.wireTool} onClick={props.onToggleWireTool}>
          <WireIcon />
        </ToolButton>
        <ToolButton label="Rotate (Ctrl+R)" onClick={props.onRotate} disabled={!props.hasSelection}>
          <RotateIcon />
        </ToolButton>
        <ToolButton label="Mirror (Ctrl+E)" onClick={props.onMirror} disabled={!props.hasSelection}>
          <MirrorIcon />
        </ToolButton>
        <ToolButton label="Delete (Del)" onClick={props.onDelete} disabled={!props.hasSelection}>
          <TrashIcon />
        </ToolButton>
        <ToolButton label="Undo (Ctrl+Z)" onClick={props.onUndo} disabled={!props.canUndo}>
          <UndoIcon />
        </ToolButton>
        <ToolButton label="Redo (Ctrl+Y)" onClick={props.onRedo} disabled={!props.canRedo}>
          <UndoIcon flipped />
        </ToolButton>
        <ToolButton label="Zoom to fit (Space)" onClick={props.onZoomToFit}>
          <FitIcon />
        </ToolButton>
      </div>

      <div style={{ flex: 1 }} />

      <div className="toolGroup" role="group" aria-label="Simulate">
        <ToolButton label="Probe: click a pin to open a live .dyn tile" active={props.probeMode} onClick={props.onToggleProbe}>
          <ProbeIcon />
        </ToolButton>
        <ToolButton label="Dynamic display of selected display nodes" onClick={props.onStartDisplay} disabled={props.running}>
          <PlayIcon />
        </ToolButton>
        <button
          className="toolRunBtn"
          onClick={props.onRun}
          disabled={props.running}
          title={`Run .${props.analysisMode} (Ctrl+Enter)`}
        >
          <RunIcon running={props.running} />
          <span>{props.running ? "Running…" : `Run .${props.analysisMode}`}</span>
        </button>
        <ToolButton label="User guide (F1)" onClick={props.onOpenHelp}>
          <HelpIcon />
        </ToolButton>
      </div>
    </div>
  );
}

function ToolButton({
  label,
  active,
  disabled,
  onClick,
  children
}: {
  label: string;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      className={`toolBtn ${active ? "active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      aria-pressed={active === undefined ? undefined : active}
    >
      {children}
      <span className="toolBtn__tip">{label}</span>
    </button>
  );
}

function DeviceButton({ type, active, onClick }: { type: DeviceType; active: boolean; onClick: () => void }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ratio = window.devicePixelRatio || 1;
    const size = 30;
    c.width = size * ratio;
    c.height = size * ratio;
    c.style.width = `${size}px`;
    c.style.height = `${size}px`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.translate(size / 2, size / 2);
    ctx.scale(0.4, 0.4);
    drawPalettePreview(ctx, type);
  }, [type]);

  return (
    <button
      className={`toolBtn ${active ? "active" : ""}`}
      onClick={onClick}
      title={DEVICE_LABELS[type]}
      aria-label={`Place ${DEVICE_LABELS[type]}`}
      aria-pressed={active}
    >
      <canvas ref={ref} />
      <span className="toolBtn__tip">{DEVICE_LABELS[type]}</span>
    </button>
  );
}

const iconProps = {
  width: 18,
  height: 18,
  viewBox: "0 0 20 20",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const
};

function WireIcon() {
  return (
    <svg {...iconProps}>
      <path d="M3 15h6V5h8" />
      <circle cx="3" cy="15" r="1.6" fill="currentColor" />
      <circle cx="17" cy="5" r="1.6" fill="currentColor" />
    </svg>
  );
}

function RotateIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 10a6 6 0 1 1 1.8 4.3" />
      <path d="M4 15v-4h4" />
    </svg>
  );
}

function MirrorIcon() {
  return (
    <svg {...iconProps}>
      <path d="M10 3v14" strokeDasharray="2 2" />
      <path d="M7 6L3 10l4 4z" fill="currentColor" />
      <path d="M13 6l4 4-4 4z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 6h12M8 6V4h4v2M6 6l1 10h6l1-10" />
    </svg>
  );
}

function UndoIcon({ flipped }: { flipped?: boolean }) {
  return (
    <svg {...iconProps} style={flipped ? { transform: "scaleX(-1)" } : undefined}>
      <path d="M7 6L3 10l4 4" />
      <path d="M3 10h8a4 4 0 0 1 0 8H9" />
    </svg>
  );
}

function FitIcon() {
  return (
    <svg {...iconProps}>
      <path d="M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4" />
      <rect x="7" y="7" width="6" height="6" rx="1" />
    </svg>
  );
}

function ProbeIcon() {
  return (
    <svg {...iconProps}>
      <path d="M3 17l5-5" />
      <path d="M8 12l4-4 3 3-4 4z" />
      <path d="M12 8l3-3" />
      <circle cx="16" cy="4" r="1.5" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor">
      <path d="M6 4l10 6-10 6V4z" />
    </svg>
  );
}

function RunIcon({ running }: { running: boolean }) {
  if (running) {
    return (
      <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="10" cy="10" r="7" strokeDasharray="22 22" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor">
      <path d="M6 4l10 6-10 6V4z" />
    </svg>
  );
}

function HelpIcon() {
  return (
    <svg {...iconProps} strokeWidth={1.7}>
      <circle cx="10" cy="10" r="8" />
      <path d="M7.7 7.4a2.6 2.6 0 0 1 4.9 1.2c0 1.8-2.1 2.1-2.1 3.4" />
      <path d="M10 15h.01" />
    </svg>
  );
}
