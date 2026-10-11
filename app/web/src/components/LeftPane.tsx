import { ReactNode, useEffect, useRef, useState } from "react";
import {
  AnalysisState,
  BasicAnalysisMode,
  CanvasComponent,
  DeviceType,
  SchematicLevel,
  ThreadSetting,
  DEVICE_LABELS
} from "../lib/schematic";
import { drawPalettePreview } from "../lib/symbols";

type Props = {
  analysis: AnalysisState;
  onAnalysisChange: (patch: Partial<AnalysisState>) => void;
  /** Server thread budget (GET /health thread_budget.total); null until known. */
  threadBudgetTotal: number | null;
  onRun: () => void;
  running: boolean;
  availableNodes: string[];
  onAddProbeNode: (label: string) => void;
  onRemoveProbeNode: (label: string) => void;
  onClearProbeNodes: () => void;
  onAddMorOutputNode: (label: string) => void;
  onRemoveMorOutputNode: (label: string) => void;
  onClearMorOutputNodes: () => void;
  onResetAnalysis: () => void;

  levels: SchematicLevel[];
  activeLevelId: string;
  onSelectLevel: (id: string) => void;
  onAddLevel: (parentId: string | null) => void;
  onRenameLevel: (id: string, newTitle: string) => void;
  onDeleteLevel: (id: string) => void;

  libraryComponents: DeviceType[];
  onStartDrag: (type: DeviceType) => void;

  selectedComponent: CanvasComponent | null;
  onUpdateComponent: (id: string, patch: Partial<CanvasComponent>) => void;
  /** Bumped when the user double-clicks a part: expands and focuses Properties. */
  propertiesFocusToken: number;

  netlistPreview: string;
  generatedNetlist: string;
  netlistMode: "schematic" | "netlist";
  netlistText: string;
  onNetlistTextChange: (text: string) => boolean;
};

export function LeftPane(props: Props) {
  return (
    <aside className="leftPane">
      <CollapsibleSection title="Simulation" defaultOpen>
        <SimulationControls
          analysis={props.analysis}
          onAnalysisChange={props.onAnalysisChange}
          threadBudgetTotal={props.threadBudgetTotal}
          availableNodes={props.availableNodes}
          onAddProbeNode={props.onAddProbeNode}
          onRemoveProbeNode={props.onRemoveProbeNode}
          onClearProbeNodes={props.onClearProbeNodes}
          onAddMorOutputNode={props.onAddMorOutputNode}
          onRemoveMorOutputNode={props.onRemoveMorOutputNode}
          onClearMorOutputNodes={props.onClearMorOutputNodes}
          onResetAnalysis={props.onResetAnalysis}
          onRun={props.onRun}
          running={props.running}
        />
      </CollapsibleSection>

      <CollapsibleSection title="Hierarchy" defaultOpen>
        <HierarchyTree
          levels={props.levels}
          activeLevelId={props.activeLevelId}
          onSelect={props.onSelectLevel}
          onAdd={props.onAddLevel}
          onRename={props.onRenameLevel}
          onDelete={props.onDeleteLevel}
        />
      </CollapsibleSection>

      <CollapsibleSection title="Netlist">
        <div className="hintText" style={{ marginTop: 0, marginBottom: 8 }}>
          Generated SPICE netlist after a run. Edit it to switch to raw netlist mode.
        </div>
        <textarea
          className="textarea miniPreview netlistEditor"
          spellCheck={false}
          value={props.netlistMode === "netlist" ? props.netlistText : props.generatedNetlist || props.netlistPreview}
          onChange={(e) => {
            props.onNetlistTextChange(e.target.value);
          }}
        />
        <div className="hintText" style={{ marginTop: 6 }}>
          Editing this netlist clears the schematic after confirmation.
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="Library">
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {props.libraryComponents.map((t) => (
            <LibItem key={t} type={t} onClick={() => props.onStartDrag(t)} />
          ))}
          <div className="hintText">
            Click a library item then click in the canvas to place it.
          </div>
        </div>
      </CollapsibleSection>

      {props.selectedComponent ? (
        <CollapsibleSection
          title={`Properties — ${props.selectedComponent.name}`}
          defaultOpen
          openToken={props.propertiesFocusToken}
          id="properties-section"
        >
          <PropertyEditor
            component={props.selectedComponent}
            levels={props.levels}
            onUpdate={(patch) => props.onUpdateComponent(props.selectedComponent!.id, patch)}
            focusToken={props.propertiesFocusToken}
          />
        </CollapsibleSection>
      ) : null}
    </aside>
  );
}

function parseMetric(raw?: string): number {
  if (!raw) return NaN;
  const match = raw.trim().match(/^([-+]?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?)([a-zA-Zµ]*)$/i);
  if (!match) return Number(raw);
  const base = Number(match[1]);
  const suffix = (match[2] || "").toLowerCase();
  const scale: Record<string, number> = {
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
  return suffix in scale ? base * scale[suffix] : Number(raw);
}

function formatMetric(value: number): string {
  const absValue = Math.abs(value);
  const choices: [number, string][] = [
    [1e9, "G"],
    [1e6, "Meg"],
    [1e3, "k"],
    [1, ""],
    [1e-3, "m"],
    [1e-6, "u"],
    [1e-9, "n"],
    [1e-12, "p"]
  ];
  const [scale, suffix] = choices.find(([candidate]) => absValue >= candidate) ?? [1e-15, "f"];
  const scaled = value / scale;
  return `${Number(scaled.toPrecision(4))}${suffix}`;
}

function sourceSubtypePatch(component: CanvasComponent, subtype: string): Partial<CanvasComponent> {
  const previousSubtype = component.subtype ?? "DC";
  const patch: Partial<CanvasComponent> = { subtype };
  if ((subtype === "SIN" || subtype === "COS") && previousSubtype === "STEP") {
    const delay = parseMetric(component.value2);
    patch.value2 = Number.isFinite(delay) && delay > 0 ? formatMetric(1 / delay) : "1Meg";
  } else if ((subtype === "SIN" || subtype === "COS") && (!component.value2 || previousSubtype === "FUNC")) {
    patch.value2 = "1Meg";
  } else if (subtype === "STEP" && (previousSubtype === "SIN" || previousSubtype === "COS")) {
    const frequency = parseMetric(component.value2);
    patch.value2 = Number.isFinite(frequency) && frequency > 0 ? formatMetric(1 / frequency) : "1u";
  } else if (subtype === "FUNC" && previousSubtype !== "FUNC") {
    patch.value2 = "0";
  }
  return patch;
}

function CollapsibleSection({
  title,
  defaultOpen,
  openToken,
  id,
  children
}: {
  title: string;
  defaultOpen?: boolean;
  /** When this value changes the section expands (used for double-click → Properties). */
  openToken?: number;
  id?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(!!defaultOpen);
  useEffect(() => {
    if (openToken) setOpen(true);
  }, [openToken]);
  return (
    <div className={`section ${open ? "" : "collapsed"}`} id={id}>
      <button
        type="button"
        className="section__header"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className="chev">▾</span>
      </button>
      <div className="section__body">{children}</div>
    </div>
  );
}

/** "auto" or 1..N worker threads, N = the server's thread budget from /health. */
function ThreadsControl({
  value,
  budgetTotal,
  onChange
}: {
  value: ThreadSetting;
  budgetTotal: number | null;
  onChange: (value: ThreadSetting) => void;
}) {
  const current = value === "auto" ? "auto" : Math.max(1, Math.floor(Number(value) || 1));
  const total = Math.max(1, budgetTotal ?? 1);
  const counts = Array.from({ length: total }, (_, i) => i + 1);
  // A saved workspace may ask for more threads than this server has; keep it selectable.
  if (current !== "auto" && current > total) counts.push(current);
  const budgetText = budgetTotal ? `The server budget is ${budgetTotal} threads` : "The server budget is not known yet";
  return (
    <div className="formRow">
      <label
        title={`Worker threads for each run. auto = as many as the server has free right now; a number is an upper limit. ${budgetText} (MNA_NUM_THREADS); concurrent runs share it.`}
      >
        Threads
      </label>
      <select
        className="select"
        value={String(current)}
        onChange={(e) => onChange(e.target.value === "auto" ? "auto" : Number(e.target.value))}
      >
        <option value="auto">auto</option>
        {counts.map((n) => (
          <option key={n} value={String(n)}>
            {budgetTotal !== null && n > budgetTotal ? `${n} (capped at ${budgetTotal})` : n}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Secondary disclosure used for advanced solver options inside a section. */
function SubPanel({ title, children, defaultOpen }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className={`subPanel ${open ? "open" : ""}`}>
      <button type="button" className="subPanel__header" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="chev">▾</span>
        <span>{title}</span>
      </button>
      {open ? <div className="subPanel__body">{children}</div> : null}
    </div>
  );
}

function SimulationControls({
  analysis,
  onAnalysisChange,
  threadBudgetTotal,
  availableNodes,
  onAddProbeNode,
  onRemoveProbeNode,
  onClearProbeNodes,
  onAddMorOutputNode,
  onRemoveMorOutputNode,
  onClearMorOutputNodes,
  onResetAnalysis,
  onRun,
  running
}: {
  analysis: AnalysisState;
  onAnalysisChange: (patch: Partial<AnalysisState>) => void;
  threadBudgetTotal: number | null;
  availableNodes: string[];
  onAddProbeNode: (label: string) => void;
  onRemoveProbeNode: (label: string) => void;
  onClearProbeNodes: () => void;
  onAddMorOutputNode: (label: string) => void;
  onRemoveMorOutputNode: (label: string) => void;
  onClearMorOutputNodes: () => void;
  onResetAnalysis: () => void;
  onRun: () => void;
  running: boolean;
}) {
  const modeOptions: { value: BasicAnalysisMode; label: string }[] = [
    { value: "op", label: ".op" },
    { value: "tran", label: ".tran" },
    { value: "ac", label: ".ac" },
    { value: "hb", label: ".hb" }
  ];
  const krylovValueLabel =
    analysis.krylovMethod === "arnoldi_gmres"
      ? "restart"
      : analysis.krylovMethod === "auto"
        ? "rank/budget"
        : "iter budget";
  const krylovHint =
    analysis.krylovMethod === "arnoldi_gmres"
      ? "Arnoldi/GMRES uses the manual value as the restarted subspace size."
      : analysis.krylovMethod === "conjugate_gradient"
        ? "CG uses the manual value as a maximum iteration budget; there is no restarted subspace."
        : analysis.krylovMethod === "conjugate_residual"
          ? "MINRES/CR uses the manual value as a maximum iteration budget; there is no restarted subspace."
          : "Auto chooses the method after matrix classification; the value is restart for Arnoldi and iteration budget for CG/MINRES.";
  return (
    <>
      <div className="formRow">
        <label>Mode</label>
        <select
          className="select"
          value={analysis.mode}
          onChange={(e) => onAnalysisChange({ mode: e.target.value as BasicAnalysisMode })}
        >
          {modeOptions.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {analysis.mode === "tran" && (
        <>
          <div className="formRow">
            <label>t_stop</label>
            <input
              className="input"
              value={analysis.tStop}
              onChange={(e) => onAnalysisChange({ tStop: e.target.value })}
            />
          </div>
          <div className="formRow">
            <label>t_step</label>
            <input
              className="input"
              value={analysis.tStep}
              onChange={(e) => onAnalysisChange({ tStep: e.target.value })}
            />
          </div>
        </>
      )}

      {analysis.mode === "ac" && (
        <>
          <div className="formRow">
            <label>f_start</label>
            <input
              className="input"
              value={analysis.fStart}
              onChange={(e) => onAnalysisChange({ fStart: e.target.value })}
            />
          </div>
          <div className="formRow">
            <label>f_stop</label>
            <input
              className="input"
              value={analysis.fStop}
              onChange={(e) => onAnalysisChange({ fStop: e.target.value })}
            />
          </div>
          <div className="formRow">
            <label>points</label>
            <input
              className="input"
              type="number"
              value={analysis.points}
              onChange={(e) => onAnalysisChange({ points: Number(e.target.value) })}
            />
          </div>
        </>
      )}

      {analysis.mode === "hb" && (
        <>
          <div className="formRow">
            <label>harmonics</label>
            <input
              className="input"
              type="number"
              min={1}
              value={analysis.harmonics}
              onChange={(e) => onAnalysisChange({ harmonics: Number(e.target.value) })}
            />
          </div>
          <div className="formRow">
            <label>window</label>
            <input
              className="input"
              value={analysis.hbTimeWindow}
              onChange={(e) => onAnalysisChange({ hbTimeWindow: e.target.value })}
              placeholder="spectrum"
            />
          </div>
        </>
      )}

      <ThreadsControl
        value={analysis.numThreads}
        budgetTotal={threadBudgetTotal}
        onChange={(numThreads) => onAnalysisChange({ numThreads })}
      />

      <div className="formRow" style={{ flexDirection: "column", alignItems: "stretch" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span className="fieldLabel" style={{ flex: 1 }}>Display nodes</span>
          <button
            className="ghostBtn"
            type="button"
            onClick={onClearProbeNodes}
            disabled={analysis.probeNodes.length === 0}
          >
            Clear
          </button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap" }}>
          {analysis.probeNodes.length === 0 ? (
            <span className="hintText" style={{ marginLeft: 0 }}>
              Empty = show all.
            </span>
          ) : (
            analysis.probeNodes.map((n) => (
              <span key={n} className="tag">
                {n}
                <button onClick={() => onRemoveProbeNode(n)} aria-label="Remove">
                  ×
                </button>
              </span>
            ))
          )}
        </div>
        <select
          className="select"
          defaultValue=""
          onChange={(e) => {
            if (e.target.value) onAddProbeNode(e.target.value);
            e.currentTarget.value = "";
          }}
          style={{ marginTop: 6 }}
        >
          <option value="" disabled>
            Add node…
          </option>
          {availableNodes
            .filter((n) => !analysis.probeNodes.includes(n))
            .map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
        </select>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
        <button className="runBtn" disabled={running} onClick={onRun} style={{ flex: 1, width: "auto" }}>
          {running ? "Running…" : `Run .${analysis.mode}`}
        </button>
        <button className="ghostBtn" type="button" onClick={onResetAnalysis} title="Reset simulation controls to defaults">
          Reset
        </button>
      </div>

      <div className="controlDivider" />
      <div className="fieldLabel" style={{ marginBottom: 4 }}>Advanced</div>
      <SubPanel title="Dynamic display">
        <div className="hintText" style={{ marginTop: 0, marginBottom: 6 }}>
          Live streaming .tran: add display nodes and press the toolbar play icon, or use probe to click a pin.
        </div>
        <div className="formRow">
          <label title="Simulation seconds per real-time second. e.g. 1m = 1ms of sim per 1 s of wall time.">
            speed (s/s)
          </label>
          <input
            className="input"
            value={analysis.dynSpeed}
            onChange={(e) => onAnalysisChange({ dynSpeed: e.target.value })}
          />
        </div>
        <div className="formRow">
          <label title="Sliding window of sim-time kept on screen. e.g. 5m = newest 5 ms of samples, older points are discarded so the plot doesn't accumulate.">
            window
          </label>
          <input
            className="input"
            value={analysis.dynWindow}
            onChange={(e) => onAnalysisChange({ dynWindow: e.target.value })}
            placeholder="5m"
          />
        </div>
      </SubPanel>

      <SubPanel title="Krylov subspace solver">
        <div className="formRow">
          <label title="Enable a Krylov subspace iterative solver for the full MNA system. Auto chooses by matrix class; manual algorithm selection overrides the class.">
            Enable
          </label>
          <input
            type="checkbox"
            checked={analysis.krylov}
            onChange={(e) => onAnalysisChange({ krylov: e.target.checked })}
          />
        </div>
        {analysis.krylov ? (
          <>
            <div className="formRow">
              <label>algorithm</label>
              <select
                className="select"
                value={analysis.krylovMethod}
                onChange={(e) => onAnalysisChange({ krylovMethod: e.target.value as AnalysisState["krylovMethod"] })}
              >
                <option value="auto">Auto</option>
                <option value="arnoldi_gmres">Arnoldi / GMRES</option>
                <option value="conjugate_residual">MINRES / CR</option>
                <option value="conjugate_gradient">Conjugate Gradient</option>
              </select>
            </div>
            <div className="formRow">
              <label title="Auto uses 50% of the actual generated matrix dimension. Manual sends the integer rank you enter.">
                rank
              </label>
              <select
                className="select"
                value={analysis.krylovRankMode}
                onChange={(e) => onAnalysisChange({ krylovRankMode: e.target.value as AnalysisState["krylovRankMode"] })}
              >
                <option value="auto">Auto</option>
                <option value="manual">Manual</option>
              </select>
            </div>
            {analysis.krylovRankMode === "manual" ? (
              <div className="formRow">
                <label>{krylovValueLabel}</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  step={1}
                  value={analysis.krylovRank}
                  onChange={(e) => {
                    const next = Math.max(1, Math.floor(Number(e.target.value) || 1));
                    onAnalysisChange({ krylovRank: next });
                  }}
                />
              </div>
            ) : null}
          </>
        ) : null}
        <div className="hintText" style={{ marginTop: 4 }}>
          {analysis.krylov
            ? `${krylovHint} Auto rank resolves to ceil(50% of matrix dimension).`
            : "Krylov subspace solver disabled: simulations use the regular direct solver path."}
        </div>
      </SubPanel>

      <SubPanel title="Model-order reduction (MOR)">
        <div className="formRow">
          <label title="Enable model-order reduction for a few selected outputs.">
            Enable
          </label>
          <input
            type="checkbox"
            checked={analysis.mor}
            onChange={(e) => onAnalysisChange({ mor: e.target.checked })}
          />
        </div>
        {analysis.mor ? (
          <>
            <div className="formRow">
              <label>method</label>
              <select
                className="select"
                value={analysis.morMethod}
                onChange={(e) => onAnalysisChange({ morMethod: e.target.value as AnalysisState["morMethod"] })}
              >
                <option value="auto">Auto</option>
                <option value="linear_krylov">Linear Krylov MOR</option>
                <option value="tpwl">TPWL / POD</option>
              </select>
            </div>
            <div className="formRow">
              <label title="Auto order uses min(n, max(10, min(120, 4*(inputs+outputs)))).">
                order
              </label>
              <select
                className="select"
                value={analysis.morOrderMode}
                onChange={(e) => onAnalysisChange({ morOrderMode: e.target.value as AnalysisState["morOrderMode"] })}
              >
                <option value="auto">Auto</option>
                <option value="manual">Manual</option>
              </select>
            </div>
            {analysis.morOrderMode === "manual" ? (
              <div className="formRow">
                <label>basis size</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  step={1}
                  value={analysis.morOrder}
                  onChange={(e) => {
                    const next = Math.max(1, Math.floor(Number(e.target.value) || 1));
                    onAnalysisChange({ morOrder: next });
                  }}
                />
              </div>
            ) : null}
            <div className="formRow" style={{ flexDirection: "column", alignItems: "stretch" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span className="fieldLabel" style={{ flex: 1 }}>MOR outputs</span>
                <button
                  className="ghostBtn"
                  type="button"
                  onClick={onClearMorOutputNodes}
                  disabled={analysis.morOutputNodes.length === 0}
                >
                  Clear
                </button>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap" }}>
                {analysis.morOutputNodes.length === 0 ? (
                  <span className="hintText" style={{ marginLeft: 0 }}>
                    Pick cared outputs.
                  </span>
                ) : (
                  analysis.morOutputNodes.map((n) => (
                    <span key={n} className="tag">
                      {n}
                      <button onClick={() => onRemoveMorOutputNode(n)} aria-label="Remove">
                        ×
                      </button>
                    </span>
                  ))
                )}
              </div>
              <select
                className="select"
                defaultValue=""
                onChange={(e) => {
                  if (e.target.value) onAddMorOutputNode(e.target.value);
                  e.currentTarget.value = "";
                }}
                style={{ marginTop: 6 }}
              >
                <option value="" disabled>
                  Add output…
                </option>
                {availableNodes
                  .filter((n) => !analysis.morOutputNodes.includes(n))
                  .map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
              </select>
            </div>
          </>
        ) : null}
        <div className="hintText" style={{ marginTop: 4 }}>
          {analysis.mor
            ? "Reduced results contain only MOR outputs. Display nodes must be empty or a subset of those outputs."
            : "MOR disabled: full MNA state is solved and available."}
        </div>
      </SubPanel>

    </>
  );
}

function HierarchyTree({
  levels,
  activeLevelId,
  onSelect,
  onAdd,
  onRename,
  onDelete
}: {
  levels: SchematicLevel[];
  activeLevelId: string;
  onSelect: (id: string) => void;
  onAdd: (parentId: string | null) => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
}) {
  const roots = levels.filter((l) => l.parentId === null);
  return (
    <div className="tree">
      {roots.map((root) => (
        <LevelBranch
          key={root.id}
          level={root}
          depth={0}
          levels={levels}
          activeId={activeLevelId}
          onSelect={onSelect}
          onAdd={onAdd}
          onRename={onRename}
          onDelete={onDelete}
        />
      ))}
      <button className="ghostBtn" style={{ marginTop: 8 }} onClick={() => onAdd(null)}>
        + New top-level
      </button>
    </div>
  );
}

function LevelBranch({
  level,
  depth,
  levels,
  activeId,
  onSelect,
  onAdd,
  onRename,
  onDelete
}: {
  level: SchematicLevel;
  depth: number;
  levels: SchematicLevel[];
  activeId: string;
  onSelect: (id: string) => void;
  onAdd: (parentId: string | null) => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
}) {
  const children = levels.filter((l) => l.parentId === level.id);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(level.title);
  return (
    <>
      <div
        className={`treeItem ${activeId === level.id ? "active" : ""}`}
        onClick={() => onSelect(level.id)}
        onDoubleClick={() => setRenaming(true)}
      >
        <span className="treeItem__depth" style={{ width: 8 + depth * 12 }} />
        <span className="treeItem__chev">{children.length ? "▾" : "•"}</span>
        {renaming ? (
          <input
            className="input"
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              onRename(level.id, draft || level.title);
              setRenaming(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onRename(level.id, draft || level.title);
                setRenaming(false);
              }
            }}
          />
        ) : (
          <span>{level.title}</span>
        )}
        <span
          className="treeItem__add"
          onClick={(e) => {
            e.stopPropagation();
            onAdd(level.id);
          }}
        >
          +sub
        </span>
        <span
          className="treeItem__add"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(level.id);
          }}
          title="Delete this level (children are also removed)"
        >
          ×
        </span>
      </div>
      {children.map((c) => (
        <LevelBranch
          key={c.id}
          level={c}
          depth={depth + 1}
          levels={levels}
          activeId={activeId}
          onSelect={onSelect}
          onAdd={onAdd}
          onRename={onRename}
          onDelete={onDelete}
        />
      ))}
    </>
  );
}

function LibItem({ type, onClick }: { type: DeviceType; onClick: () => void }) {
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
    ctx.scale(0.3, 0.3);
    drawPalettePreview(ctx, type);
  }, [type]);
  return (
    <div className="libItem" onClick={onClick}>
      <canvas ref={ref} />
      <span>{DEVICE_LABELS[type]}</span>
    </div>
  );
}

function PropertyEditor({
  component,
  levels,
  onUpdate,
  focusToken
}: {
  component: CanvasComponent;
  levels: SchematicLevel[];
  onUpdate: (patch: Partial<CanvasComponent>) => void;
  focusToken?: number;
}) {
  const firstInputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (!focusToken) return;
    const el = firstInputRef.current;
    if (!el) return;
    el.scrollIntoView({ block: "center" });
    el.focus();
    el.select();
  }, [focusToken]);
  const isSource = component.type === "V" || component.type === "I";
  const isControlled = component.type === "VCVS" || component.type === "VCCS";
  const isCurrentControlled = component.type === "CCCS" || component.type === "CCVS";
  const isBjt = component.type === "QNPN" || component.type === "QPNP";
  const isMos = component.type === "NMOS" || component.type === "PMOS";
  const isSubckt = component.type === "SUBCKT";
  const isLabel = component.type === "LABEL";
  const isNode = component.type === "NODE";
  const bjtModel = component.metadata?.model === "small_signal" ? "small_signal" : "level1";
  const mosModel = component.metadata?.model === "level1" ? "level1" : "small_signal";
  const updateMetadata = (key: string, value: string) =>
    onUpdate({ metadata: { ...(component.metadata ?? {}), [key]: value } });
  const updateBjtModel = (model: "small_signal" | "level1") => {
    const metadata: Record<string, string> = { ...(component.metadata ?? {}), model };
    if (model === "level1") {
      onUpdate({
        metadata: {
          ...metadata,
          vaf: metadata.vaf ?? "100",
          var: metadata.var ?? "25",
          cje: metadata.cje ?? "4p",
          cjc: metadata.cjc ?? "2p",
          rb: metadata.rb ?? "50",
          re: metadata.re ?? "0.5",
          rc: metadata.rc ?? "5"
        },
        value: component.value && component.value !== "40m" ? component.value : "1e-15",
        value2: component.value2 && component.value2 !== "2.5k" ? component.value2 : "150",
        value3: component.value3 && component.value3 !== "100k" ? component.value3 : "3"
      });
    } else {
      onUpdate({
        metadata: {
          ...metadata,
          cpi: metadata.cpi ?? "8p",
          cmu: metadata.cmu ?? "3p",
          ccs: metadata.ccs ?? "0",
          rb: metadata.rb ?? "0",
          re: metadata.re ?? "0"
        },
        value: component.value && component.value !== "1e-15" ? component.value : "40m",
        value2: component.value2 && component.value2 !== "150" ? component.value2 : "2.5k",
        value3: component.value3 && component.value3 !== "3" ? component.value3 : "100k"
      });
    }
  };
  const updateMosModel = (model: "small_signal" | "level1") => {
    const metadata: Record<string, string> = { ...(component.metadata ?? {}), model };
    if (model === "level1") {
      onUpdate({
        metadata: { ...metadata, cgs: metadata.cgs ?? "2p", cgd: metadata.cgd ?? "1p" },
        value: component.value || "1m",
        value2: component.value2 && component.value2 !== "50k" ? component.value2 : "0.4",
        value3: component.value3 && component.value3 !== "5p" ? component.value3 : "0.02"
      });
    } else {
      onUpdate({
        metadata: { ...metadata, cgd: metadata.cgd ?? "1p", gmb: metadata.gmb ?? "0", cbs: metadata.cbs ?? "0", cbd: metadata.cbd ?? "0" },
        value: component.value || "5m",
        value2: component.value2 && component.value2 !== "0.4" ? component.value2 : "50k",
        value3: component.value3 && component.value3 !== "0.02" ? component.value3 : "5p"
      });
    }
  };
  const updatePinCount = (count: number) => {
    const safeCount = Math.max(1, Math.min(24, count || 1));
    const current = component.pins ?? [];
    const pins = Array.from({ length: safeCount }, (_, i) => current[i] ?? `node_${i + 1}`);
    onUpdate({ pins });
  };
  const updatePinName = (index: number, value: string) => {
    const pins = [...(component.pins ?? [])];
    pins[index] = value.trim() || `node_${index + 1}`;
    onUpdate({ pins });
  };
  return (
    <>
      <div className="formRow">
        <label>{isSubckt ? "displayed_name" : isLabel ? "text" : isNode ? "node" : "Name"}</label>
        <input
          ref={firstInputRef}
          className="input"
          value={component.name}
          onChange={(e) => onUpdate({ name: e.target.value })}
        />
      </div>
      {isSubckt ? (
        <>
          <div className="formRow">
            <label>entity_name</label>
            <select
              className="select"
              value={component.subcircuitId ?? ""}
              onChange={(e) => {
                const entity = levels.find((level) => level.id === e.target.value);
                onUpdate({
                  subcircuitId: e.target.value,
                  pins: entity?.pins?.length ? [...entity.pins] : component.pins
                });
              }}
            >
              <option value="">Unlinked</option>
              {levels
                .filter((level) => level.parentId !== null)
                .map((level) => (
                  <option key={level.id} value={level.id}>
                    {level.title}
                  </option>
                ))}
            </select>
          </div>
          <div className="formRow">
            <label>nodes</label>
            <input
              className="input"
              type="number"
              min={1}
              max={24}
              value={(component.pins ?? []).length}
              onChange={(e) => updatePinCount(Number(e.target.value))}
            />
          </div>
          {(component.pins ?? []).map((pin, index) => (
            <div className="formRow" key={`${component.id}-pin-${index}`}>
              <label>{`node ${index + 1}`}</label>
              <input
                className="input"
                value={pin}
                onChange={(e) => updatePinName(index, e.target.value)}
              />
            </div>
          ))}
        </>
      ) : null}
      {isNode ? (
        <div className="hintText">
          Same-named Node markers are electrically connected. In a sub-entity, a node named in_1 exposes port_in_1 to matching SUBCKT pins.
        </div>
      ) : null}
      {component.type !== "GND" && !isBjt && !isMos && !isSubckt && !isNode && !isLabel ? (
        <div className="formRow">
          <label>Value</label>
          <input
            className="input"
            value={component.value}
            onChange={(e) => onUpdate({ value: e.target.value })}
          />
        </div>
      ) : null}
      {isBjt ? (
        <>
          <div className="formRow">
            <label>model</label>
            <select
              className="select"
              value={bjtModel}
              onChange={(e) => updateBjtModel(e.target.value as "small_signal" | "level1")}
            >
              <option value="level1">Level-1</option>
              <option value="small_signal">small-signal</option>
            </select>
          </div>
          {bjtModel === "level1" ? (
            <>
              <div className="formRow">
                <label>IS</label>
                <input className="input" value={component.value} onChange={(e) => onUpdate({ value: e.target.value })} />
              </div>
              <div className="formRow">
                <label>BF</label>
                <input className="input" value={component.value2 ?? ""} onChange={(e) => onUpdate({ value2: e.target.value })} />
              </div>
              <div className="formRow">
                <label>BR</label>
                <input className="input" value={component.value3 ?? ""} onChange={(e) => onUpdate({ value3: e.target.value })} />
              </div>
              <div className="formRow">
                <label>VAF</label>
                <input className="input" value={component.metadata?.vaf ?? ""} onChange={(e) => updateMetadata("vaf", e.target.value)} />
              </div>
              <div className="formRow">
                <label>VAR</label>
                <input className="input" value={component.metadata?.var ?? ""} onChange={(e) => updateMetadata("var", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Cje</label>
                <input className="input" value={component.metadata?.cje ?? ""} onChange={(e) => updateMetadata("cje", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Cjc</label>
                <input className="input" value={component.metadata?.cjc ?? ""} onChange={(e) => updateMetadata("cjc", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Rb</label>
                <input className="input" value={component.metadata?.rb ?? ""} onChange={(e) => updateMetadata("rb", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Re</label>
                <input className="input" value={component.metadata?.re ?? ""} onChange={(e) => updateMetadata("re", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Rc</label>
                <input className="input" value={component.metadata?.rc ?? ""} onChange={(e) => updateMetadata("rc", e.target.value)} />
              </div>
            </>
          ) : (
            <>
              <div className="formRow">
                <label>gm</label>
                <input className="input" value={component.value} onChange={(e) => onUpdate({ value: e.target.value })} />
              </div>
              <div className="formRow">
                <label>rπ</label>
                <input className="input" value={component.value2 ?? ""} onChange={(e) => onUpdate({ value2: e.target.value })} />
              </div>
              <div className="formRow">
                <label>ro</label>
                <input className="input" value={component.value3 ?? ""} onChange={(e) => onUpdate({ value3: e.target.value })} />
              </div>
              <div className="formRow">
                <label>Cπ</label>
                <input className="input" value={component.metadata?.cpi ?? ""} onChange={(e) => updateMetadata("cpi", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Cμ</label>
                <input className="input" value={component.metadata?.cmu ?? ""} onChange={(e) => updateMetadata("cmu", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Ccs</label>
                <input className="input" value={component.metadata?.ccs ?? "0"} onChange={(e) => updateMetadata("ccs", e.target.value)} />
              </div>
            </>
          )}
        </>
      ) : null}
      {isMos ? (
        <>
          <div className="formRow">
            <label>model</label>
            <select
              className="select"
              value={mosModel}
              onChange={(e) => updateMosModel(e.target.value as "small_signal" | "level1")}
            >
              <option value="level1">Level-1</option>
              <option value="small_signal">small-signal</option>
            </select>
          </div>
          {mosModel === "level1" ? (
            <>
              <div className="formRow">
                <label>beta</label>
                <input className="input" value={component.value} onChange={(e) => onUpdate({ value: e.target.value })} />
              </div>
              <div className="formRow">
                <label>vth</label>
                <input className="input" value={component.value2 ?? ""} onChange={(e) => onUpdate({ value2: e.target.value })} />
              </div>
              <div className="formRow">
                <label>lambda</label>
                <input className="input" value={component.value3 ?? ""} onChange={(e) => onUpdate({ value3: e.target.value })} />
              </div>
              <div className="formRow">
                <label>Cgs</label>
                <input className="input" value={component.metadata?.cgs ?? ""} onChange={(e) => updateMetadata("cgs", e.target.value)} />
              </div>
              <div className="formRow">
                <label>Cgd</label>
                <input className="input" value={component.metadata?.cgd ?? ""} onChange={(e) => updateMetadata("cgd", e.target.value)} />
              </div>
            </>
          ) : (
            <>
          <div className="formRow">
            <label>gm</label>
            <input className="input" value={component.value} onChange={(e) => onUpdate({ value: e.target.value })} />
          </div>
          <div className="formRow">
            <label>ro</label>
            <input className="input" value={component.value2 ?? ""} onChange={(e) => onUpdate({ value2: e.target.value })} />
          </div>
          <div className="formRow">
            <label>Cgs</label>
            <input className="input" value={component.value3 ?? ""} onChange={(e) => onUpdate({ value3: e.target.value })} />
          </div>
          <div className="formRow">
            <label>Cgd</label>
            <input className="input" value={component.metadata?.cgd ?? ""} onChange={(e) => updateMetadata("cgd", e.target.value)} />
          </div>
          <div className="formRow">
            <label>gmb</label>
            <input className="input" value={component.metadata?.gmb ?? "0"} onChange={(e) => updateMetadata("gmb", e.target.value)} />
          </div>
          <div className="formRow">
            <label>Cbs</label>
            <input className="input" value={component.metadata?.cbs ?? "0"} onChange={(e) => updateMetadata("cbs", e.target.value)} />
          </div>
          <div className="formRow">
            <label>Cbd</label>
            <input className="input" value={component.metadata?.cbd ?? "0"} onChange={(e) => updateMetadata("cbd", e.target.value)} />
          </div>
            </>
          )}
        </>
      ) : null}
      {isSource ? (
        <>
          <div className="formRow">
            <label>Subtype</label>
            <select
              className="select"
              value={component.subtype ?? "DC"}
              onChange={(e) => onUpdate(sourceSubtypePatch(component, e.target.value))}
            >
              <option value="DC">DC</option>
              <option value="AC">AC</option>
              <option value="SIN">SIN</option>
              <option value="COS">COS</option>
              <option value="STEP">STEP</option>
              <option value="FUNC">FUNC</option>
            </select>
          </div>
          {(component.subtype === "SIN" ||
            component.subtype === "COS" ||
            component.subtype === "STEP" ||
            component.subtype === "FUNC") && (
            <div className="formRow">
              <label>
                {component.subtype === "FUNC" ? "expr" : component.subtype === "STEP" ? "t_on" : "freq"}
              </label>
              <input
                className="input"
                value={component.value2 ?? ""}
                onChange={(e) => onUpdate({ value2: e.target.value })}
              />
            </div>
          )}
        </>
      ) : null}
      {isControlled ? (
        <>
          <div className="formRow">
            <label>ctrl n+</label>
            <input
              className="input"
              value={component.ctrlNode1 ?? ""}
              onChange={(e) => onUpdate({ ctrlNode1: e.target.value })}
            />
          </div>
          <div className="formRow">
            <label>ctrl n-</label>
            <input
              className="input"
              value={component.ctrlNode2 ?? ""}
              onChange={(e) => onUpdate({ ctrlNode2: e.target.value })}
            />
          </div>
        </>
      ) : null}
      {isCurrentControlled ? (
        <div className="formRow">
          <label>ctrl src</label>
          <input
            className="input"
            value={component.ctrlSource ?? ""}
            placeholder="V1"
            onChange={(e) => onUpdate({ ctrlSource: e.target.value })}
          />
        </div>
      ) : null}
      <div className="formRow">
        <label>Rotation</label>
        <select
          className="select"
          value={component.rotation}
          onChange={(e) => onUpdate({ rotation: Number(e.target.value) as CanvasComponent["rotation"] })}
        >
          <option value={0}>0°</option>
          <option value={90}>90°</option>
          <option value={180}>180°</option>
          <option value={270}>270°</option>
        </select>
      </div>
      <div className="formRow">
        <label>Mirror</label>
        <input
          type="checkbox"
          checked={Boolean(component.mirrored)}
          onChange={(e) => onUpdate({ mirrored: e.target.checked })}
        />
      </div>
    </>
  );
}
