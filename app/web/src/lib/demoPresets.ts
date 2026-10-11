import {
  AnalysisState,
  CanvasComponent,
  CanvasWire,
  SchematicLevel,
  SchematicPreset,
  pinEndpoint,
  pointEndpoint,
  snap
} from "./schematic";

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

function cloneAnalysis(overrides: Partial<AnalysisState>): AnalysisState {
  return { ...defaultAnalysis, ...overrides };
}

function mk(
  partial: Omit<CanvasComponent, "rotation"> & Partial<Pick<CanvasComponent, "rotation">>
): CanvasComponent {
  return { rotation: 0, ...partial, x: snap(partial.x), y: snap(partial.y) };
}

function pinWire(
  id: string,
  startCid: string,
  startPin: string,
  endCid: string,
  endPin: string
): CanvasWire {
  return {
    id,
    start: pinEndpoint(startCid, startPin),
    end: pinEndpoint(endCid, endPin)
  };
}

const level1BjtMetadata: Record<string, string> = {
  model: "level1",
  vaf: "100",
  var: "25",
  cje: "4p",
  cjc: "2p",
  rb: "50",
  re: "0.5",
  rc: "5"
};

function level1NpnProps(): Pick<CanvasComponent, "value" | "value2" | "value3" | "metadata"> {
  return {
    value: "1e-15",
    value2: "150",
    value3: "3",
    metadata: { ...level1BjtMetadata }
  };
}

/** Visible-in-the-Preset-dropdown demo schematics. Cleared per user request. */
export const SCHEMATIC_PRESETS: SchematicPreset[] = [];

/* ----------------------------------------------------------------------- */
/* Hidden demos accessible only via the Demo top-menu dropdown.            */
/* ----------------------------------------------------------------------- */

/** CE amp: +15 V biased Level-1 BJT demo with AC-coupled input/output. */
const ceAmplifier: SchematicPreset = (() => {
  const components: CanvasComponent[] = [
    mk({ id: "v-vcc", type: "V", name: "VCC", value: "15", subtype: "DC", x: 160, y: 130 }),
    mk({ id: "node-vcc-src", type: "NODE", name: "vcc", value: "node", x: 160, y: 94 }),
    mk({ id: "node-vcc-rc", type: "NODE", name: "vcc", value: "node", x: 500, y: 134 }),
    mk({ id: "node-vcc-bias", type: "NODE", name: "vcc", value: "node", x: 424, y: 172 }),
    mk({ id: "v-in", type: "V", name: "Vin", value: "0.02", subtype: "SIN", value2: "1k", x: 160, y: 320 }),
    mk({ id: "node-vin", type: "NODE", name: "vin", value: "node", x: 240, y: 290 }),
    mk({ id: "c-in", type: "C", name: "Cin", value: "1u", x: 320, y: 280 }),
    mk({ id: "node-base", type: "NODE", name: "base", value: "node", x: 424, y: 280 }),
    mk({
      id: "q-1",
      type: "QNPN",
      name: "Q1",
      ...level1NpnProps(),
      x: 500,
      y: 280
    }),
    mk({ id: "r-c", type: "R", name: "Rc", value: "12k", x: 500, y: 170, rotation: 90 }),
    mk({ id: "r-e", type: "R", name: "Re", value: "330", x: 500, y: 390, rotation: 90 }),
    mk({ id: "r-btop", type: "R", name: "RbTop", value: "220k", x: 424, y: 208, rotation: 90 }),
    mk({ id: "r-bbot", type: "R", name: "RbBot", value: "15k", x: 424, y: 352, rotation: 90 }),
    mk({ id: "c-out", type: "C", name: "Cout", value: "4.7u", x: 640, y: 244 }),
    mk({ id: "node-out", type: "NODE", name: "out", value: "node", x: 720, y: 260 }),
    mk({ id: "r-load", type: "R", name: "Rload", value: "10k", x: 760, y: 300, rotation: 90 }),
    mk({ id: "gnd-vcc", type: "GND", name: "GNDvcc", value: "0", x: 160, y: 190 }),
    mk({ id: "gnd-in", type: "GND", name: "GNDin", value: "0", x: 160, y: 380 }),
    mk({ id: "gnd-bias", type: "GND", name: "GNDbias", value: "0", x: 424, y: 430 }),
    mk({ id: "gnd-e", type: "GND", name: "GNDe", value: "0", x: 500, y: 450 }),
    mk({ id: "gnd-load", type: "GND", name: "GNDload", value: "0", x: 760, y: 360 })
  ];
  const wires: CanvasWire[] = [
    pinWire("ce-w-vcc-p", "v-vcc", "p", "node-vcc-src", "n"),
    pinWire("ce-w-vcc-n", "v-vcc", "n", "gnd-vcc", "g"),
    pinWire("ce-w-in-g", "v-in", "n", "gnd-in", "g"),
    pinWire("ce-w-in-node", "v-in", "p", "node-vin", "n"),
    pinWire("ce-w-in-c", "node-vin", "n", "c-in", "p"),
    pinWire("ce-w-c-base", "c-in", "n", "node-base", "n"),
    pinWire("ce-w-q-base", "q-1", "b", "node-base", "n"),
    pinWire("ce-w-btop-vcc", "r-btop", "p", "node-vcc-bias", "n"),
    pinWire("ce-w-btop-base", "r-btop", "n", "node-base", "n"),
    pinWire("ce-w-bbot-base", "r-bbot", "p", "node-base", "n"),
    pinWire("ce-w-bbot-g", "r-bbot", "n", "gnd-bias", "g"),
    pinWire("ce-w-rc-vcc", "r-c", "p", "node-vcc-rc", "n"),
    pinWire("ce-w-rc-c", "r-c", "n", "q-1", "c"),
    pinWire("ce-w-e-re", "q-1", "e", "r-e", "p"),
    pinWire("ce-w-re-g", "r-e", "n", "gnd-e", "g"),
    pinWire("ce-w-c-out", "q-1", "c", "c-out", "p"),
    pinWire("ce-w-out-node", "c-out", "n", "node-out", "n"),
    pinWire("ce-w-out-load", "node-out", "n", "r-load", "p"),
    pinWire("ce-w-load-g", "r-load", "n", "gnd-load", "g")
  ];
  return {
    id: "ce-amplifier",
    title: "CE Amplifier",
    description:
      "Common-emitter NPN Level-1 amplifier biased from a +15 V rail near mid-collector voltage, with emitter degeneration, junction capacitances, Early effect, and AC-coupled input/output.",
    analysis: cloneAnalysis({ mode: "tran", tStop: "10m", tStep: "5u", probeNodes: ["V(vin)", "V(out)"] }),
    components,
    wires
  };
})();

/* ----------------------------------------------------------------------- */
/* 3-stage op-amp using the SUBCKT mechanism.                              */
/*                                                                          */
/* Top level shows three SUBCKT rectangles (DiffAmp, MidStage, OutStage)   */
/* wired in cascade. Each sub-level holds editable NODE markers whose      */
/* names become ``port_<pin>`` junctions for the backend flattener.        */
/* ----------------------------------------------------------------------- */

const threeStageOpamp: SchematicPreset = (() => {
  /*
   * BJT-only three-stage +15 V physical Level-1 demo:
   *
   * 1. Differential input pair: two QNPNs share a tail resistor, each collector
   *    has a load to the shared +15 V rail, and the right collector is the
   *    single-ended out.
   * 2. Common-emitter voltage gain block: QNPN collector output with Rc to
   *    the +15 V rail and emitter degeneration to ground.
   * 3. Follower output buffer: QNPN emitter follower, the BJT equivalent of the
   *    requested source-follower behavior.
   *
   * The backend keeps these as QNPN components through subcircuit flattening.
   * .op solves the nonlinear bias, .ac linearizes around that bias, and .tran
   * continues solving nonlinear device currents at each timestep.
   */
  function buildDifferentialStage(): SchematicLevel {
    const prefix = "diff";
    return {
      id: "lvl-diff",
      title: "Differential amp",
      parentId: "root",
      pins: ["in_p", "out", "vcc", "in_n"],
      components: [
        mk({ id: `${prefix}-port-inp`, type: "NODE", name: "in_p", value: "node", x: 80, y: 260 }),
        mk({ id: `${prefix}-port-inn`, type: "NODE", name: "in_n", value: "node", x: 80, y: 340 }),
        mk({ id: `${prefix}-port-vcc`, type: "NODE", name: "vcc", value: "node", x: 380, y: 70 }),
        mk({ id: `${prefix}-port-out`, type: "NODE", name: "out", value: "node", x: 660, y: 210 }),
        mk({
          id: `${prefix}-q1`,
          type: "QNPN",
          name: "Q1",
          ...level1NpnProps(),
          x: 300,
          y: 260
        }),
        mk({
          id: `${prefix}-q2`,
          type: "QNPN",
          name: "Q2",
          ...level1NpnProps(),
          x: 460,
          y: 260
        }),
        mk({ id: `${prefix}-rc1`, type: "R", name: "Rc1", value: "5.6k", x: 300, y: 130, rotation: 90 }),
        mk({ id: `${prefix}-rc2`, type: "R", name: "Rc2", value: "5.6k", x: 460, y: 130, rotation: 90 }),
        mk({ id: `${prefix}-tail`, type: "R", name: "Rtail", value: "2.2k", x: 380, y: 360, rotation: 90 }),
        mk({ id: `${prefix}-rbp-top`, type: "R", name: "RbpTop", value: "150k", x: 180, y: 150, rotation: 90 }),
        mk({ id: `${prefix}-rbp-bot`, type: "R", name: "RbpBot", value: "20k", x: 180, y: 430, rotation: 90 }),
        mk({ id: `${prefix}-rbn-top`, type: "R", name: "RbnTop", value: "150k", x: 540, y: 150, rotation: 90 }),
        mk({ id: `${prefix}-rbn-bot`, type: "R", name: "RbnBot", value: "20k", x: 540, y: 430, rotation: 90 }),
        mk({ id: `${prefix}-gnd-tail`, type: "GND", name: "GNDt", value: "0", x: 380, y: 420 }),
        mk({ id: `${prefix}-gnd-bias`, type: "GND", name: "GNDbias", value: "0", x: 360, y: 520 })
      ],
      wires: [
        pinWire(`${prefix}-w-inp`, `${prefix}-q1`, "b", `${prefix}-port-inp`, "n"),
        pinWire(`${prefix}-w-inn`, `${prefix}-q2`, "b", `${prefix}-port-inn`, "n"),
        pinWire(`${prefix}-w-c1`, `${prefix}-q1`, "c", `${prefix}-rc1`, "n"),
        pinWire(`${prefix}-w-c2`, `${prefix}-q2`, "c", `${prefix}-rc2`, "n"),
        pinWire(`${prefix}-w-rc1-vcc`, `${prefix}-rc1`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-rc2-vcc`, `${prefix}-rc2`, "p", `${prefix}-port-vcc`, "n"),
        { id: `${prefix}-w-e1`, start: pinEndpoint(`${prefix}-q1`, "e"), end: pointEndpoint(380, 296) },
        { id: `${prefix}-w-e2`, start: pinEndpoint(`${prefix}-q2`, "e"), end: pointEndpoint(380, 296) },
        { id: `${prefix}-w-tail`, start: pinEndpoint(`${prefix}-tail`, "p"), end: pointEndpoint(380, 296) },
        pinWire(`${prefix}-w-tail-g`, `${prefix}-tail`, "n", `${prefix}-gnd-tail`, "g"),
        pinWire(`${prefix}-w-rbp-top-vcc`, `${prefix}-rbp-top`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-rbp-top-in`, `${prefix}-rbp-top`, "n", `${prefix}-port-inp`, "n"),
        pinWire(`${prefix}-w-rbp-bot-in`, `${prefix}-rbp-bot`, "p", `${prefix}-port-inp`, "n"),
        pinWire(`${prefix}-w-rbp-bot-g`, `${prefix}-rbp-bot`, "n", `${prefix}-gnd-bias`, "g"),
        pinWire(`${prefix}-w-rbn-top-vcc`, `${prefix}-rbn-top`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-rbn-top-in`, `${prefix}-rbn-top`, "n", `${prefix}-port-inn`, "n"),
        pinWire(`${prefix}-w-rbn-bot-in`, `${prefix}-rbn-bot`, "p", `${prefix}-port-inn`, "n"),
        pinWire(`${prefix}-w-rbn-bot-g`, `${prefix}-rbn-bot`, "n", `${prefix}-gnd-bias`, "g"),
        pinWire(`${prefix}-w-out`, `${prefix}-q2`, "c", `${prefix}-port-out`, "n")
      ]
    };
  }

  function buildCommonEmitterStage(): SchematicLevel {
    const prefix = "mid";
    return {
      id: "lvl-mid",
      title: "CE gain stage",
      parentId: "root",
      pins: ["in", "out", "vcc"],
      components: [
        mk({ id: `${prefix}-port-in`, type: "NODE", name: "in", value: "node", x: 80, y: 240 }),
        mk({ id: `${prefix}-port-vcc`, type: "NODE", name: "vcc", value: "node", x: 300, y: 70 }),
        mk({ id: `${prefix}-port-out`, type: "NODE", name: "out", value: "node", x: 560, y: 180 }),
        mk({
          id: `${prefix}-q1`,
          type: "QNPN",
          name: "Q1",
          ...level1NpnProps(),
          x: 300,
          y: 240
        }),
        mk({ id: `${prefix}-rc`, type: "R", name: "Rc", value: "12k", x: 300, y: 110, rotation: 90 }),
        mk({ id: `${prefix}-re`, type: "R", name: "Re", value: "1.5k", x: 300, y: 350, rotation: 90 }),
        mk({ id: `${prefix}-rbase-top`, type: "R", name: "RbTop", value: "150k", x: 160, y: 150, rotation: 90 }),
        mk({ id: `${prefix}-rbase-bot`, type: "R", name: "RbBot", value: "20k", x: 160, y: 340, rotation: 90 }),
        mk({ id: `${prefix}-gnd-e`, type: "GND", name: "GNDe", value: "0", x: 300, y: 410 }),
        mk({ id: `${prefix}-gnd-bias`, type: "GND", name: "GNDbias", value: "0", x: 160, y: 400 })
      ],
      wires: [
        pinWire(`${prefix}-w-in`, `${prefix}-q1`, "b", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-c`, `${prefix}-q1`, "c", `${prefix}-rc`, "n"),
        pinWire(`${prefix}-w-rc-vcc`, `${prefix}-rc`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-e-re`, `${prefix}-q1`, "e", `${prefix}-re`, "p"),
        pinWire(`${prefix}-w-re-g`, `${prefix}-re`, "n", `${prefix}-gnd-e`, "g"),
        pinWire(`${prefix}-w-rbase-top-vcc`, `${prefix}-rbase-top`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-rbase-top-in`, `${prefix}-rbase-top`, "n", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-rbase-bot-in`, `${prefix}-rbase-bot`, "p", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-rbase-bot-g`, `${prefix}-rbase-bot`, "n", `${prefix}-gnd-bias`, "g"),
        pinWire(`${prefix}-w-out`, `${prefix}-q1`, "c", `${prefix}-port-out`, "n")
      ]
    };
  }

  function buildFollowerStage(): SchematicLevel {
    const prefix = "out";
    return {
      id: "lvl-out",
      title: "Emitter follower",
      parentId: "root",
      pins: ["in", "out", "vcc"],
      components: [
        mk({ id: `${prefix}-port-in`, type: "NODE", name: "in", value: "node", x: 80, y: 220 }),
        mk({ id: `${prefix}-port-vcc`, type: "NODE", name: "vcc", value: "node", x: 300, y: 80 }),
        mk({ id: `${prefix}-port-out`, type: "NODE", name: "out", value: "node", x: 560, y: 300 }),
        mk({
          id: `${prefix}-q1`,
          type: "QNPN",
          name: "Q1",
          ...level1NpnProps(),
          x: 300,
          y: 220
        }),
        mk({ id: `${prefix}-re`, type: "R", name: "Re", value: "1.2k", x: 300, y: 400, rotation: 90 }),
        mk({ id: `${prefix}-rbase-top`, type: "R", name: "RbTop", value: "68k", x: 160, y: 150, rotation: 90 }),
        mk({ id: `${prefix}-rbase-bot`, type: "R", name: "RbBot", value: "75k", x: 160, y: 360, rotation: 90 }),
        mk({ id: `${prefix}-gnd-e`, type: "GND", name: "GNDe", value: "0", x: 300, y: 500 }),
        mk({ id: `${prefix}-gnd-bias`, type: "GND", name: "GNDbias", value: "0", x: 160, y: 420 })
      ],
      wires: [
        pinWire(`${prefix}-w-in`, `${prefix}-q1`, "b", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-c-vcc`, `${prefix}-q1`, "c", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-e`, `${prefix}-q1`, "e", `${prefix}-re`, "p"),
        pinWire(`${prefix}-w-re-g`, `${prefix}-re`, "n", `${prefix}-gnd-e`, "g"),
        pinWire(`${prefix}-w-rbase-top-vcc`, `${prefix}-rbase-top`, "p", `${prefix}-port-vcc`, "n"),
        pinWire(`${prefix}-w-rbase-top-in`, `${prefix}-rbase-top`, "n", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-rbase-bot-in`, `${prefix}-rbase-bot`, "p", `${prefix}-port-in`, "n"),
        pinWire(`${prefix}-w-rbase-bot-g`, `${prefix}-rbase-bot`, "n", `${prefix}-gnd-bias`, "g"),
        pinWire(`${prefix}-w-out`, `${prefix}-q1`, "e", `${prefix}-port-out`, "n")
      ]
    };
  }

  const lvlDiff = buildDifferentialStage();
  const lvlMid = buildCommonEmitterStage();
  const lvlOut = buildFollowerStage();

  /* ------------------- Top level: SUBCKT cascade ----------------------- */
  // SUBCKT pin ordering follows getPinCoordinates. The differential block uses
  // four pins so in_p sits left, out sits right, vcc sits on top, and in_n sits
  // on the bottom where the top level ties it to the local small-signal reference.
  const topComponents: CanvasComponent[] = [
    mk({ id: "v-in", type: "V", name: "Vin", value: "0.001", subtype: "SIN", value2: "1k", x: 120, y: 276 }),
    mk({ id: "v-vcc", type: "V", name: "VCC", value: "15", subtype: "DC", x: 120, y: 120 }),
    mk({ id: "c-input", type: "C", name: "Cin", value: "1u", x: 240, y: 276 }),
    mk({ id: "c-ref", type: "C", name: "Cref", value: "1u", x: 360, y: 380, rotation: 90 }),
    mk({ id: "c-diff-mid", type: "C", name: "Cdm", value: "1u", x: 500, y: 240 }),
    mk({ id: "c-mid-out", type: "C", name: "Cmo", value: "1u", x: 760, y: 240 }),
    mk({ id: "node-vcc-src", type: "NODE", name: "vcc", value: "node", x: 120, y: 84 }),
    mk({ id: "node-vcc-diff", type: "NODE", name: "vcc", value: "node", x: 360, y: 168 }),
    mk({ id: "node-vcc-mid", type: "NODE", name: "vcc", value: "node", x: 620, y: 168 }),
    mk({ id: "node-vcc-out", type: "NODE", name: "vcc", value: "node", x: 880, y: 168 }),
    mk({ id: "node-vin", type: "NODE", name: "vin", value: "node", x: 180, y: 246 }),
    mk({ id: "node-out", type: "NODE", name: "out", value: "node", x: 1000, y: 276 }),
    mk({
      id: "sub-diff",
      type: "SUBCKT",
      name: "DiffAmp",
      value: "",
      subcircuitId: "lvl-diff",
      pins: ["in_p", "out", "vcc", "in_n"],
      x: 360,
      y: 240
    }),
    mk({
      id: "sub-mid",
      type: "SUBCKT",
      name: "MidStage",
      value: "",
      subcircuitId: "lvl-mid",
      pins: ["in", "out", "vcc"],
      x: 620,
      y: 240
    }),
    mk({
      id: "sub-out",
      type: "SUBCKT",
      name: "OutStage",
      value: "",
      subcircuitId: "lvl-out",
      pins: ["in", "out", "vcc"],
      x: 880,
      y: 240
    }),
    mk({ id: "r-load", type: "R", name: "Rload", value: "10k", x: 1080, y: 276, rotation: 90 }),
    mk({ id: "gnd-vcc", type: "GND", name: "GNDvcc", value: "0", x: 120, y: 180 }),
    mk({ id: "gnd-in", type: "GND", name: "GNDi", value: "0", x: 120, y: 360 }),
    mk({ id: "gnd-ref", type: "GND", name: "GNDref", value: "0", x: 360, y: 460 }),
    mk({ id: "gnd-load", type: "GND", name: "GNDl", value: "0", x: 1080, y: 336 })
  ];

  // The cascade carries the single-ended output forward. Inter-stage links run
  // through explicit points so the SUBCKT flattener can stitch both block pins
  // onto the same top-level net.
  const topWires: CanvasWire[] = [
    pinWire("op-w-vcc-p", "v-vcc", "p", "node-vcc-src", "n"),
    pinWire("op-w-vcc-n", "v-vcc", "n", "gnd-vcc", "g"),
    pinWire("op-w-vcc-diff", "sub-diff", "vcc", "node-vcc-diff", "n"),
    pinWire("op-w-vcc-mid", "sub-mid", "vcc", "node-vcc-mid", "n"),
    pinWire("op-w-vcc-out", "sub-out", "vcc", "node-vcc-out", "n"),
    pinWire("op-w1", "v-in", "n", "gnd-in", "g"),
    pinWire("op-w2a", "v-in", "p", "node-vin", "n"),
    pinWire("op-w2c", "node-vin", "n", "c-input", "p"),
    pinWire("op-w2b", "c-input", "n", "sub-diff", "in_p"),
    pinWire("op-w3a", "sub-diff", "in_n", "c-ref", "p"),
    pinWire("op-w3b", "c-ref", "n", "gnd-ref", "g"),
    pinWire("op-w4a", "sub-diff", "out", "c-diff-mid", "p"),
    pinWire("op-w4b", "c-diff-mid", "n", "sub-mid", "in"),
    pinWire("op-w5a", "sub-mid", "out", "c-mid-out", "p"),
    pinWire("op-w5b", "c-mid-out", "n", "sub-out", "in"),
    pinWire("op-w6", "sub-out", "out", "node-out", "n"),
    pinWire("op-w6b", "node-out", "n", "r-load", "p"),
    pinWire("op-w7", "r-load", "n", "gnd-load", "g")
  ];

  return {
    id: "three-stage-opamp",
    title: "3-Stage Op-Amp (SUBCKT)",
    description:
      "Top level shows three AC-coupled +15 V Level-1 BJT SUBCKT blocks, each with its own bias network; .op bias and nonlinear transient are used before small-signal AC linearization.",
    analysis: cloneAnalysis({ mode: "tran", tStop: "10m", tStep: "5u", probeNodes: ["V(vin)", "V(out)"] }),
    components: topComponents,
    wires: topWires,
    extraLevels: [lvlDiff, lvlMid, lvlOut]
  };
})();

const hbCloseToneMixer: SchematicPreset = (() => {
  const components: CanvasComponent[] = [
    mk({ id: "hb-node-mix", type: "NODE", name: "mix", value: "node", x: 360, y: 116 }),
    mk({ id: "hb-node-env", type: "NODE", name: "env", value: "node", x: 680, y: 116 }),
    mk({ id: "hb-i-bias", type: "I", name: "IBIAS", value: "50u", subtype: "DC", x: 120, y: 200 }),
    mk({ id: "hb-i-tone-a", type: "I", name: "I1", value: "40u", subtype: "SIN", value2: "1G", x: 230, y: 200 }),
    mk({ id: "hb-i-tone-b", type: "I", name: "I2", value: "40u", subtype: "SIN", value2: "1.005G", x: 340, y: 200 }),
    mk({ id: "hb-r-load", type: "R", name: "RLOAD", value: "10k", x: 450, y: 200, rotation: 90 }),
    mk({ id: "hb-d-mix", type: "D", name: "D1", value: "1e-12", x: 560, y: 200, rotation: 90 }),
    mk({ id: "hb-g-env", type: "VCCS", name: "GENV", value: "1m", x: 620, y: 180 }),
    mk({ id: "hb-l-env", type: "L", name: "LENV", value: "1u", x: 700, y: 210, rotation: 90 }),
    mk({ id: "hb-c-env", type: "C", name: "CENV", value: "1.013n", x: 780, y: 210, rotation: 90 }),
    mk({ id: "hb-r-q", type: "R", name: "RQ", value: "4.3k", x: 860, y: 210, rotation: 90 }),
    mk({ id: "hb-gnd", type: "GND", name: "GND", value: "0", x: 440, y: 320 })
  ];
  const wires: CanvasWire[] = [
    pinWire("hb-w-mix-bias", "hb-i-bias", "p", "hb-node-mix", "n"),
    pinWire("hb-w-mix-a", "hb-i-tone-a", "p", "hb-node-mix", "n"),
    pinWire("hb-w-mix-b", "hb-i-tone-b", "p", "hb-node-mix", "n"),
    pinWire("hb-w-mix-r", "hb-r-load", "p", "hb-node-mix", "n"),
    pinWire("hb-w-mix-d", "hb-d-mix", "p", "hb-node-mix", "n"),
    pinWire("hb-w-g-ctrl", "hb-g-env", "cp", "hb-node-mix", "n"),
    pinWire("hb-w-env-g", "hb-g-env", "p", "hb-node-env", "n"),
    pinWire("hb-w-env-l", "hb-l-env", "p", "hb-node-env", "n"),
    pinWire("hb-w-env-c", "hb-c-env", "p", "hb-node-env", "n"),
    pinWire("hb-w-env-rq", "hb-r-q", "p", "hb-node-env", "n"),
    pinWire("hb-w-g-bias", "hb-i-bias", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-a", "hb-i-tone-a", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-b", "hb-i-tone-b", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-r", "hb-r-load", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-d", "hb-d-mix", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-gout", "hb-g-env", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-gctrl", "hb-g-env", "cn", "hb-gnd", "g"),
    pinWire("hb-w-g-l", "hb-l-env", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-c", "hb-c-env", "n", "hb-gnd", "g"),
    pinWire("hb-w-g-rq", "hb-r-q", "n", "hb-gnd", "g")
  ];
  return {
    id: "hb-close-tone-mixer",
    title: "HB GHz Close-Tone Mixer",
    description:
      "Nonlinear diode mixer driven by 1 GHz and 1.005 GHz tones with a high-Q 5 MHz envelope tank that rings for 40+ beat periods; HB jumps to the steady-state beat.",
    analysis: cloneAnalysis({
      mode: "hb",
      harmonics: 205,
      hbTimeWindow: "200n",
      probeNodes: ["V(mix)", "V(env)"]
    }),
    components,
    wires
  };
})();

const largeSramCircuit: SchematicPreset = (() => {
  const rows = 10;
  const cols = 10;
  const selectedRow = 3;
  const selectedCol = 4;
  const components: CanvasComponent[] = [];
  const wires: CanvasWire[] = [];
  const sharedNodes = new Map<string, string>();
  let nodeIndex = 0;
  let wireIndex = 0;
  const vHigh = "1.2";
  const mosCap = { cgs: "2f", cgd: "1f" };
  const cellPins = ["bl", "q", "wl", "blb", "qb", "vdd", "gnd", "init_q", "init_qb"];
  const wlSelectedExpr = `Piecewise((0, t < 1e-9), (${vHigh}, t < 2.5e-9), (0, t < 4.5e-9), (${vHigh}, t < 6e-9), (0, True))`;
  const writeExpr = `Piecewise((0, t < 1e-9), (${vHigh}, t < 2.5e-9), (0, True))`;
  const writeBarExpr = `Piecewise((${vHigh}, t < 1e-9), (0, t < 2.5e-9), (${vHigh}, True))`;
  const prechargeExpr = `Piecewise((0, t < 1e-9), (${vHigh}, t < 3.5e-9), (0, t < 4.5e-9), (${vHigh}, True))`;

  function buildSramCellLevel(): SchematicLevel {
    const cellComponents: CanvasComponent[] = [];
    const cellWires: CanvasWire[] = [];
    let localWireIndex = 0;

    const addCellComponent = (
      partial: Omit<CanvasComponent, "rotation"> & Partial<Pick<CanvasComponent, "rotation">>
    ) => {
      const component = mk(partial);
      cellComponents.push(component);
      return component.id;
    };
    const addCellPort = (name: string, x: number, y: number) =>
      addCellComponent({
        id: `sram-cell-port-${name}`,
        type: "NODE",
        name,
        value: "node",
        x,
        y
      });
    const connectCell = (componentId: string, pin: string, nodeId: string) => {
      cellWires.push(pinWire(`sram-cell-w-${localWireIndex++}`, componentId, pin, nodeId, "n"));
    };
    const addCellLevel1Mos = (
      id: string,
      type: "NMOS" | "PMOS",
      name: string,
      beta: string,
      x: number,
      y: number,
      rotation: 0 | 90 | 180 | 270 = 0
    ) =>
      addCellComponent({
        id,
        type,
        name,
        value: beta,
        value2: "0.4",
        value3: "0.02",
        metadata: { model: "level1", ...mosCap },
        x,
        y,
        rotation
      });
    const addCellTwoTerminal = (
      id: string,
      type: "R" | "C",
      name: string,
      value: string,
      pNode: string,
      nNode: string,
      x: number,
      y: number,
      rotation: 0 | 90 | 180 | 270 = 0
    ) => {
      const componentId = addCellComponent({ id, type, name, value, x, y, rotation });
      connectCell(componentId, "p", pNode);
      connectCell(componentId, "n", nNode);
    };

    const bl = addCellPort("bl", 80, 300);
    const q = addCellPort("q", 330, 300);
    const wl = addCellPort("wl", 450, 460);
    const blb = addCellPort("blb", 820, 300);
    const qb = addCellPort("qb", 570, 300);
    const vdd = addCellPort("vdd", 450, 70);
    const gnd = addCellPort("gnd", 450, 530);
    const initQ = addCellPort("init_q", 240, 120);
    const initQb = addCellPort("init_qb", 660, 120);

    const pLeft = addCellLevel1Mos("sram-cell-mp-q", "PMOS", "MPQ", "0.7m", 330, 180);
    const pRight = addCellLevel1Mos("sram-cell-mp-qb", "PMOS", "MPQB", "0.7m", 570, 180);
    const nLeft = addCellLevel1Mos("sram-cell-mn-q", "NMOS", "MNQ", "1.8m", 330, 420);
    const nRight = addCellLevel1Mos("sram-cell-mn-qb", "NMOS", "MNQB", "1.8m", 570, 420);
    const accessLeft = addCellLevel1Mos("sram-cell-max-q", "NMOS", "MAXQ", "2.5m", 190, 300, 90);
    const accessRight = addCellLevel1Mos("sram-cell-max-qb", "NMOS", "MAXQB", "2.5m", 710, 300, 270);

    connectCell(pLeft, "s", vdd);
    connectCell(pLeft, "d", q);
    connectCell(pLeft, "g", qb);
    connectCell(nLeft, "d", q);
    connectCell(nLeft, "s", gnd);
    connectCell(nLeft, "g", qb);

    connectCell(pRight, "s", vdd);
    connectCell(pRight, "d", qb);
    connectCell(pRight, "g", q);
    connectCell(nRight, "d", qb);
    connectCell(nRight, "s", gnd);
    connectCell(nRight, "g", q);

    connectCell(accessLeft, "d", q);
    connectCell(accessLeft, "s", bl);
    connectCell(accessLeft, "g", wl);
    connectCell(accessRight, "d", qb);
    connectCell(accessRight, "s", blb);
    connectCell(accessRight, "g", wl);

    addCellTwoTerminal("sram-cell-c-q", "C", "CQ", "4f", q, gnd, 390, 365, 90);
    addCellTwoTerminal("sram-cell-c-qb", "C", "CQB", "4f", qb, gnd, 510, 365, 90);
    addCellTwoTerminal("sram-cell-rinit-q", "R", "RINITQ", "200Meg", q, initQ, 250, 210, 90);
    addCellTwoTerminal("sram-cell-rinit-qb", "R", "RINITQB", "200Meg", qb, initQb, 650, 210, 90);

    return {
      id: "lvl-sram-cell",
      title: "SRAM 6T Cell",
      parentId: "root",
      pins: cellPins,
      components: cellComponents,
      wires: cellWires
    };
  }

  const addComponent = (
    partial: Omit<CanvasComponent, "rotation"> & Partial<Pick<CanvasComponent, "rotation">>
  ) => {
    const component = mk(partial);
    components.push(component);
    return component.id;
  };
  const addNode = (name: string, x: number, y: number) =>
    addComponent({
      id: `node-${name.replace(/[^a-z0-9_]/gi, "_")}-${nodeIndex++}`,
      type: "NODE",
      name,
      value: "node",
      x,
      y
    });
  const addSharedNode = (name: string, x: number, y: number) => {
    const key = name.trim();
    const existing = sharedNodes.get(key);
    if (existing) return existing;
    const nodeId = addNode(name, x, y);
    sharedNodes.set(key, nodeId);
    return nodeId;
  };
  const connect = (componentId: string, pin: string, nodeId: string) => {
    wires.push(pinWire(`sram-w-${wireIndex++}`, componentId, pin, nodeId, "n"));
  };
  const addSource = (id: string, name: string, node: string, subtype: string, value: string, expr: string | undefined, x: number, y: number) => {
    const src = addComponent({ id, type: "V", name, value, subtype, value2: expr, x, y });
    const p = addSharedNode(node, x, y - 36);
    const n = addSharedNode("gnd", x, y + 36);
    connect(src, "p", p);
    connect(src, "n", n);
    return p;
  };
  const addLevel1Mos = (
    id: string,
    type: "NMOS" | "PMOS",
    name: string,
    beta: string,
    x: number,
    y: number,
    rotation: 0 | 90 | 180 | 270 = 0,
    caps: Record<string, string> = mosCap
  ) =>
    addComponent({
      id,
      type,
      name,
      value: beta,
      value2: "0.4",
      value3: "0.02",
      metadata: { model: "level1", ...caps },
      x,
      y,
      rotation
    });
  const addTwoTerminal = (
    id: string,
    type: "R" | "C",
    name: string,
    value: string,
    pNode: string,
    nNode: string,
    x: number,
    y: number,
    rotation: 0 | 90 | 180 | 270 = 0
  ) => {
    const c = addComponent({ id, type, name, value, x, y, rotation });
    connect(c, "p", pNode);
    connect(c, "n", nNode);
    return c;
  };
  const addSramCell = (row: number, col: number, x: number, y: number) => {
    const cell = `${row}_${col}`;
    const componentId = addComponent({
      id: `sram-cell-${cell}`,
      type: "SUBCKT",
      name: `Cell_${cell}`,
      value: "",
      subcircuitId: "lvl-sram-cell",
      pins: cellPins,
      x,
      y
    });
    const qStartsHigh = row === selectedRow && col === selectedCol ? false : (row + col) % 2 === 0;
    const pinNodes: Record<string, string> = {
      bl: addSharedNode(`bl_${col}`, x - 102, y - 36),
      q: addSharedNode(`q_${cell}`, x - 102, y),
      wl: addSharedNode(`wl_${row}`, x - 102, y + 36),
      blb: addSharedNode(`blb_${col}`, x + 102, y - 36),
      qb: addSharedNode(`qb_${cell}`, x + 102, y),
      vdd: addSharedNode("vdd", x + 102, y + 36),
      gnd: addSharedNode("gnd", x - 36, y - 102),
      init_q: addSharedNode(qStartsHigh ? "vdd" : "gnd", x, y - 102),
      init_qb: addSharedNode(qStartsHigh ? "gnd" : "vdd", x + 36, y - 102)
    };
    for (const pin of cellPins) {
      connect(componentId, pin, pinNodes[pin]);
    }
  };

  const vddSource = addComponent({
    id: "sram-vdd-src",
    type: "V",
    name: "VDD",
    value: vHigh,
    subtype: "DC",
    x: 100,
    y: 150
  });
  const vddNode = addSharedNode("vdd", 100, 114);
  const globalGndNode = addSharedNode("gnd", 100, 186);
  const globalGnd = addComponent({ id: "sram-gnd", type: "GND", name: "GND0", value: "0", x: 100, y: 246 });
  connect(vddSource, "p", vddNode);
  connect(vddSource, "n", globalGndNode);
  wires.push(pinWire(`sram-w-${wireIndex++}`, globalGndNode, "n", globalGnd, "g"));

  const originX = 280;
  const originY = 280;
  const cellW = 190;
  const cellH = 165;

  addSource("sram-vpch", "VPCH", "pch", "FUNC", "0", prechargeExpr, 100, 330);
  addSource("sram-vwr", "VWR", "wr", "FUNC", "0", writeExpr, 100, 430);
  addSource("sram-vwrb", "VWRB", "wr_b", "FUNC", "0", writeBarExpr, 100, 530);

  for (let col = 0; col < cols; col++) {
    const qX = originX + col * cellW;
    const qbX = qX + 80;
    const bl = addSharedNode(`bl_${col}`, qX - 72, 118);
    const blb = addSharedNode(`blb_${col}`, qbX + 72, 118);
    const vddBl = addSharedNode("vdd", qX - 72, 46);
    const vddBlb = addSharedNode("vdd", qbX + 72, 46);
    const pch = addSharedNode("pch", qX, 58);
    const pchb = addSharedNode("pch", qbX, 58);
    const gndBl = addSharedNode("gnd", qX - 32, 154);
    const gndBlb = addSharedNode("gnd", qbX + 32, 154);
    const preBl = addLevel1Mos(`sram-mpch-bl-${col}`, "PMOS", `MPCHBL${col}`, "3m", qX - 72, 82);
    const preBlb = addLevel1Mos(`sram-mpch-blb-${col}`, "PMOS", `MPCHBLB${col}`, "3m", qbX + 72, 82);
    connect(preBl, "s", vddBl);
    connect(preBl, "d", bl);
    connect(preBl, "g", pch);
    connect(preBlb, "s", vddBlb);
    connect(preBlb, "d", blb);
    connect(preBlb, "g", pchb);
    addTwoTerminal(`sram-cbl-${col}`, "C", `CBL${col}`, "20f", bl, gndBl, qX - 32, 136, 90);
    addTwoTerminal(`sram-cblb-${col}`, "C", `CBLB${col}`, "20f", blb, gndBlb, qbX + 32, 136, 90);
  }

  for (let row = 0; row < rows; row++) {
    const y = originY + row * cellH;
    if (row === selectedRow) {
      addSource(`sram-vwl-${row}`, `VWL${row}`, `wl_${row}`, "FUNC", "0", wlSelectedExpr, 100, y);
    } else {
      addSource(`sram-vwl-${row}`, `VWL${row}`, `wl_${row}`, "DC", "0", undefined, 100, y);
    }
  }

  const selectedBl = addSharedNode(`bl_${selectedCol}`, originX + selectedCol * cellW - 72, 176);
  const selectedBlb = addSharedNode(`blb_${selectedCol}`, originX + selectedCol * cellW + 152, 176);
  const writeHighVdd = addSharedNode("vdd", originX + selectedCol * cellW - 112, 176);
  const writeLowGnd = addSharedNode("gnd", originX + selectedCol * cellW + 192, 176);
  const wr = addSharedNode("wr", originX + selectedCol * cellW + 152, 226);
  const wrb = addSharedNode("wr_b", originX + selectedCol * cellW - 72, 226);
  const writeHigh = addLevel1Mos("sram-mwrite-bl", "PMOS", "MWRBL", "8m", originX + selectedCol * cellW - 72, 210);
  const writeLow = addLevel1Mos("sram-mwrite-blb", "NMOS", "MWRBLB", "8m", originX + selectedCol * cellW + 152, 210);
  connect(writeHigh, "s", writeHighVdd);
  connect(writeHigh, "d", selectedBl);
  connect(writeHigh, "g", wrb);
  connect(writeLow, "d", selectedBlb);
  connect(writeLow, "s", writeLowGnd);
  connect(writeLow, "g", wr);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x = originX + col * cellW;
      const y = originY + row * cellH;
      addSramCell(row, col, x, y);
    }
  }

  const cellLevel = buildSramCellLevel();
  return {
    id: "large-sram-circuit",
    title: "Large SRAM Circuit",
    description:
      "Hierarchical 10x10 6T Level-1 MOS SRAM transient demo that writes, holds, precharges, and reads row 3 column 4.",
    analysis: cloneAnalysis({
      mode: "tran",
      tStop: "6n",
      tStep: "0.25n",
      probeNodes: ["V(q_3_4)", "V(qb_3_4)", "V(bl_4)", "V(blb_4)", "V(wl_3)"],
      krylov: false,
      krylovRankMode: "auto"
    }),
    components,
    wires,
    extraLevels: [cellLevel]
  };
})();

const largeRlcMesh: SchematicPreset = (() => {
  const size = 23;
  const components: CanvasComponent[] = [];
  const wires: CanvasWire[] = [];
  let nodeIndex = 0;
  let wireIndex = 0;

  const addComponent = (
    partial: Omit<CanvasComponent, "rotation"> & Partial<Pick<CanvasComponent, "rotation">>
  ) => {
    const component = mk(partial);
    components.push(component);
    return component.id;
  };
  const addNode = (name: string, x: number, y: number) =>
    addComponent({
      id: `rlc-node-${name.replace(/[^a-z0-9_]/gi, "_")}-${nodeIndex++}`,
      type: "NODE",
      name,
      value: "node",
      x,
      y
    });
  const connect = (componentId: string, pin: string, nodeId: string) => {
    wires.push(pinWire(`rlc-w-${wireIndex++}`, componentId, pin, nodeId, "n"));
  };
  const addTwoTerminal = (
    id: string,
    type: "R" | "C" | "L",
    name: string,
    value: string,
    pNode: string,
    nNode: string,
    x: number,
    y: number,
    rotation: 0 | 90 | 180 | 270 = 0
  ) => {
    const componentId = addComponent({ id, type, name, value, x, y, rotation });
    connect(componentId, "p", pNode);
    connect(componentId, "n", nNode);
    return componentId;
  };

  const originX = 260;
  const originY = 170;
  const dx = 70;
  const dy = 54;
  const nodeIds: string[][] = [];

  const gndNode = addNode("gnd", 110, 210);
  const gnd = addComponent({ id: "rlc-gnd", type: "GND", name: "GND0", value: "0", x: 110, y: 250 });
  wires.push(pinWire(`rlc-w-${wireIndex++}`, gndNode, "n", gnd, "g"));

  for (let row = 0; row < size; row++) {
    nodeIds[row] = [];
    for (let col = 0; col < size; col++) {
      nodeIds[row][col] = addNode(`n_${row}_${col}`, originX + col * dx, originY + row * dy);
    }
  }

  const src = addComponent({
    id: "rlc-iin",
    type: "I",
    name: "IIN",
    value: "1m",
    subtype: "STEP",
    value2: "20n",
    x: 170,
    y: 170
  });
  connect(src, "p", nodeIds[0][0]);
  connect(src, "n", gndNode);

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const node = nodeIds[row][col];
      const x = originX + col * dx;
      const y = originY + row * dy;
      addTwoTerminal(`rlc-c-${row}-${col}`, "C", `C_${row}_${col}`, "200p", node, gndNode, x + 18, y + 18, 90);
      addTwoTerminal(`rlc-rg-${row}-${col}`, "R", `RG_${row}_${col}`, "50Meg", node, gndNode, x - 18, y + 18, 90);
      if (col + 1 < size) {
        addTwoTerminal(
          `rlc-rh-${row}-${col}`,
          "R",
          `R_H_${row}_${col}`,
          "25",
          node,
          nodeIds[row][col + 1],
          x + dx / 2,
          y,
          0
        );
      }
      if (row + 1 < size) {
        addTwoTerminal(
          `rlc-lv-${row}-${col}`,
          "L",
          `L_V_${row}_${col}`,
          "20n",
          node,
          nodeIds[row + 1][col],
          x,
          y + dy / 2,
          90
        );
      }
    }
  }

  addTwoTerminal(
    "rlc-rload",
    "R",
    "RLOAD",
    "50",
    nodeIds[size - 1][size - 1],
    gndNode,
    originX + (size - 1) * dx + 54,
    originY + (size - 1) * dy,
    90
  );

  return {
    id: "large-rlc-mesh",
    title: "Large RLC Mesh",
    description:
      "Editable 23x23 distributed RLC mesh with 1035 MNA unknowns and a sparse Backward Euler operator for Krylov/MINRES benchmarking.",
    analysis: cloneAnalysis({
      mode: "tran",
      tStop: "200n",
      tStep: "10n",
      probeNodes: ["V(n_0_0)", "V(n_11_11)", "V(n_22_22)", "I(L_V_10_11)"],
      krylov: true,
      krylovRankMode: "auto",
      krylovRank: 518
    }),
    components,
    wires
  };
})();

export const HIDDEN_DEMO_PRESETS: SchematicPreset[] = [
  ceAmplifier,
  threeStageOpamp,
  hbCloseToneMixer,
  largeSramCircuit,
  largeRlcMesh
];

export const DEMO_MENU_GROUPS: { title: string; presets: SchematicPreset[] }[] = [
  { title: "Amplifier Demos", presets: [ceAmplifier, threeStageOpamp] },
  { title: "Frequency Domain Demos", presets: [hbCloseToneMixer] },
  { title: "Large Scale Circuits", presets: [largeSramCircuit, largeRlcMesh] }
];
