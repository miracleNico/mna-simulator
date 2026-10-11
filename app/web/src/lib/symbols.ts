/**
 * Flat-design schematic symbol rendering on Canvas 2D.
 *
 * All symbols are drawn centered at (0,0); the canvas transform is set up by
 * the caller to translate/rotate into component coordinates. Every function
 * should leave the context state as it found it.
 */

import { CanvasComponent, COMPONENT_BOX, DeviceType, GND_BOX, subcktPinLocalOffset } from "./schematic";

export type SymbolTheme = {
  stroke: string;
  accent: string;
  fill: string;
  label: string;
  muted: string;
};

export const DEFAULT_THEME: SymbolTheme = {
  stroke: "#e6edf7",
  accent: "#6ea8ff",
  fill: "#1a2130",
  label: "#c9d4e6",
  muted: "#7f8fa6"
};

function strokePath(ctx: CanvasRenderingContext2D, drawer: () => void, theme: SymbolTheme, width = 1.8): void {
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  drawer();
  ctx.stroke();
  ctx.restore();
}

function drawResistor(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const w = COMPONENT_BOX;
  strokePath(
    ctx,
    () => {
      ctx.moveTo(-w / 2, 0);
      ctx.lineTo(-w / 4, 0);
      for (let i = 0; i < 6; i++) {
        const dx = -w / 4 + (i + 1) * (w / 2 / 6);
        ctx.lineTo(dx, i % 2 === 0 ? -8 : 8);
      }
      ctx.lineTo(w / 4, 0);
      ctx.lineTo(w / 2, 0);
    },
    theme
  );
}

function drawCapacitor(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const w = COMPONENT_BOX;
  strokePath(ctx, () => {
    ctx.moveTo(-w / 2, 0);
    ctx.lineTo(-6, 0);
    ctx.moveTo(-6, -14);
    ctx.lineTo(-6, 14);
    ctx.moveTo(6, -14);
    ctx.lineTo(6, 14);
    ctx.moveTo(6, 0);
    ctx.lineTo(w / 2, 0);
  }, theme);
}

function drawInductor(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const w = COMPONENT_BOX;
  strokePath(ctx, () => {
    ctx.moveTo(-w / 2, 0);
    ctx.lineTo(-w / 4, 0);
    for (let i = 0; i < 4; i++) {
      const cx = -w / 4 + 6 + i * 8;
      ctx.moveTo(cx - 4, 0);
      ctx.arc(cx, 0, 4, Math.PI, 0, false);
    }
    ctx.moveTo(w / 4 + 6, 0);
    ctx.lineTo(w / 2, 0);
  }, theme);
}

function drawVoltageSource(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const h = COMPONENT_BOX;
  strokePath(ctx, () => {
    ctx.moveTo(0, -h / 2);
    ctx.lineTo(0, -16);
    ctx.moveTo(0, 16);
    ctx.lineTo(0, h / 2);
  }, theme);
  // circle
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(0, 0, 16, 0, Math.PI * 2);
  ctx.stroke();
  // +/- labels
  ctx.fillStyle = theme.label;
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("+", 0, -6);
  ctx.fillText("−", 0, 6);
  ctx.restore();
}

function drawCurrentSource(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const h = COMPONENT_BOX;
  strokePath(ctx, () => {
    ctx.moveTo(0, -h / 2);
    ctx.lineTo(0, -16);
    ctx.moveTo(0, 16);
    ctx.lineTo(0, h / 2);
  }, theme);
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(0, 0, 16, 0, Math.PI * 2);
  ctx.stroke();
  // arrow pointing up
  ctx.beginPath();
  ctx.moveTo(0, 10);
  ctx.lineTo(0, -10);
  ctx.moveTo(-4, -5);
  ctx.lineTo(0, -10);
  ctx.lineTo(4, -5);
  ctx.stroke();
  ctx.restore();
}

function drawDiode(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  const w = COMPONENT_BOX;
  strokePath(ctx, () => {
    ctx.moveTo(-w / 2, 0);
    ctx.lineTo(-8, 0);
    ctx.moveTo(8, 0);
    ctx.lineTo(w / 2, 0);
  }, theme);
  ctx.save();
  ctx.fillStyle = theme.accent;
  ctx.strokeStyle = theme.accent;
  ctx.beginPath();
  ctx.moveTo(-8, -10);
  ctx.lineTo(8, 0);
  ctx.lineTo(-8, 10);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(8, -10);
  ctx.lineTo(8, 10);
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawGround(ctx: CanvasRenderingContext2D, theme: SymbolTheme): void {
  strokePath(ctx, () => {
    ctx.moveTo(0, -GND_BOX / 2);
    ctx.lineTo(0, 0);
    ctx.moveTo(-18, 0);
    ctx.lineTo(18, 0);
    ctx.moveTo(-12, 6);
    ctx.lineTo(12, 6);
    ctx.moveTo(-6, 12);
    ctx.lineTo(6, 12);
  }, theme, 2);
}

function drawControlledSource(
  ctx: CanvasRenderingContext2D,
  theme: SymbolTheme,
  label: string
): void {
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(-COMPONENT_BOX / 2 + 12, -COMPONENT_BOX / 4);
  ctx.lineTo(-COMPONENT_BOX / 2, -COMPONENT_BOX / 4);
  ctx.moveTo(-COMPONENT_BOX / 2 + 12, COMPONENT_BOX / 4);
  ctx.lineTo(-COMPONENT_BOX / 2, COMPONENT_BOX / 4);
  ctx.moveTo(COMPONENT_BOX / 2, -COMPONENT_BOX / 4);
  ctx.lineTo(COMPONENT_BOX / 2 - 12, -COMPONENT_BOX / 4);
  ctx.moveTo(COMPONENT_BOX / 2, COMPONENT_BOX / 4);
  ctx.lineTo(COMPONENT_BOX / 2 - 12, COMPONENT_BOX / 4);
  ctx.stroke();
  // diamond
  ctx.beginPath();
  ctx.moveTo(0, -18);
  ctx.lineTo(18, 0);
  ctx.lineTo(0, 18);
  ctx.lineTo(-18, 0);
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = theme.label;
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 0, 0);
  ctx.restore();
}

function drawBJT(
  ctx: CanvasRenderingContext2D,
  theme: SymbolTheme,
  polarity: "npn" | "pnp"
): void {
  const w = COMPONENT_BOX;
  // pins: collector(0,-w/2), base(-w/2,0), emitter(0,w/2)
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 1.8;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Base lead
  ctx.beginPath();
  ctx.moveTo(-w / 2, 0);
  ctx.lineTo(-12, 0);
  ctx.stroke();
  // Vertical "bar" (base region)
  ctx.beginPath();
  ctx.moveTo(-12, -16);
  ctx.lineTo(-12, 16);
  ctx.stroke();
  // Collector & emitter slants from bar to body pins
  ctx.beginPath();
  ctx.moveTo(-12, -8);
  ctx.lineTo(8, -16);
  ctx.lineTo(8, -w / 2);
  ctx.moveTo(8, -w / 2);
  ctx.lineTo(0, -w / 2);
  // emitter slant
  ctx.moveTo(-12, 8);
  ctx.lineTo(8, 16);
  ctx.lineTo(8, w / 2);
  ctx.moveTo(8, w / 2);
  ctx.lineTo(0, w / 2);
  ctx.stroke();
  // Arrow on emitter (NPN: out from bar; PNP: into bar)
  ctx.fillStyle = theme.accent;
  ctx.strokeStyle = theme.accent;
  ctx.beginPath();
  if (polarity === "npn") {
    // arrow pointing toward emitter pin (down-right)
    const ax = 4;
    const ay = 12;
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 6, ay - 1);
    ctx.lineTo(ax - 1, ay + 5);
    ctx.closePath();
  } else {
    // arrow pointing into base bar (up-left), drawn near upper slant
    const ax = -8;
    const ay = -8;
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + 6, ay - 1);
    ctx.lineTo(ax + 1, ay + 5);
    ctx.closePath();
  }
  ctx.fill();
  ctx.restore();
}

function drawMOSFET(
  ctx: CanvasRenderingContext2D,
  theme: SymbolTheme,
  polarity: "n" | "p"
): void {
  const w = COMPONENT_BOX;
  // pins: drain(0,-w/2), gate(-w/2,0), source(0,w/2) for n; flipped for p
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 1.8;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Gate lead
  ctx.beginPath();
  ctx.moveTo(-w / 2, 0);
  ctx.lineTo(-14, 0);
  ctx.stroke();
  // Gate vertical line
  ctx.beginPath();
  ctx.moveTo(-14, -14);
  ctx.lineTo(-14, 14);
  ctx.stroke();
  // Channel (the "T" shape on right side)
  ctx.beginPath();
  ctx.moveTo(-8, -14);
  ctx.lineTo(-8, -6);
  ctx.moveTo(-8, -2);
  ctx.lineTo(-8, 2);
  ctx.moveTo(-8, 6);
  ctx.lineTo(-8, 14);
  ctx.stroke();
  // Drain & Source leads
  ctx.beginPath();
  ctx.moveTo(-8, -14);
  ctx.lineTo(8, -14);
  ctx.lineTo(8, -w / 2);
  ctx.moveTo(8, -w / 2);
  ctx.lineTo(0, -w / 2);
  ctx.moveTo(-8, 14);
  ctx.lineTo(8, 14);
  ctx.lineTo(8, w / 2);
  ctx.moveTo(8, w / 2);
  ctx.lineTo(0, w / 2);
  ctx.stroke();
  // Arrow indicating polarity (NMOS arrow points toward channel; PMOS away)
  ctx.fillStyle = theme.accent;
  ctx.strokeStyle = theme.accent;
  ctx.beginPath();
  if (polarity === "n") {
    ctx.moveTo(-2, 0);
    ctx.lineTo(-8, -3);
    ctx.lineTo(-8, 3);
  } else {
    ctx.moveTo(-8, 0);
    ctx.lineTo(-2, -3);
    ctx.lineTo(-2, 3);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawSubckt(ctx: CanvasRenderingContext2D, theme: SymbolTheme, component: CanvasComponent): void {
  ctx.save();
  ctx.strokeStyle = theme.stroke;
  ctx.fillStyle = theme.fill;
  ctx.lineWidth = 1.8;
  const w = COMPONENT_BOX * 2;
  const h = COMPONENT_BOX * 2;
  const r = 10;
  ctx.beginPath();
  ctx.moveTo(-w / 2 + r, -h / 2);
  ctx.arcTo(w / 2, -h / 2, w / 2, h / 2, r);
  ctx.arcTo(w / 2, h / 2, -w / 2, h / 2, r);
  ctx.arcTo(-w / 2, h / 2, -w / 2, -h / 2, r);
  ctx.arcTo(-w / 2, -h / 2, w / 2, -h / 2, r);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = theme.label;
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  drawUprightText(ctx, component, component.name || "SUBCKT", 0, 0, "center");

  // Pin names just inside the body edge, so a hierarchical block reads like
  // an LTspice symbol: every port is identifiable without opening the child.
  const pins = component.pins ?? [];
  ctx.fillStyle = theme.muted;
  ctx.font = "500 9px Inter, system-ui, sans-serif";
  for (const pin of pins) {
    const off = subcktPinLocalOffset(pins, pin);
    const inset = 8;
    let tx = off.dx;
    let ty = off.dy;
    let align: CanvasTextAlign = "center";
    if (off.dx <= -w / 2 + 1) {
      tx += inset;
      align = "left";
    } else if (off.dx >= w / 2 - 1) {
      tx -= inset;
      align = "right";
    } else if (off.dy <= -h / 2 + 1) {
      ty += inset;
    } else {
      ty -= inset;
    }
    drawUprightText(ctx, component, pin, tx, ty, align);
  }
  ctx.restore();
}

/**
 * Draw text at a local symbol coordinate while cancelling the symbol's
 * rotation/mirror so labels never render upside down or mirrored.
 */
function drawUprightText(
  ctx: CanvasRenderingContext2D,
  component: CanvasComponent,
  text: string,
  x: number,
  y: number,
  align: CanvasTextAlign
): void {
  ctx.save();
  ctx.translate(x, y);
  if (component.mirrored) ctx.scale(-1, 1);
  if (component.rotation) ctx.rotate((-component.rotation * Math.PI) / 180);
  let finalAlign = align;
  if (component.mirrored && align !== "center") finalAlign = align === "left" ? "right" : "left";
  if (component.rotation === 180 && align !== "center") finalAlign = finalAlign === "left" ? "right" : "left";
  ctx.textAlign = finalAlign;
  ctx.textBaseline = "middle";
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawLabel(ctx: CanvasRenderingContext2D, theme: SymbolTheme, component: CanvasComponent): void {
  const text = component.name || component.value || "label";
  ctx.save();
  ctx.fillStyle = theme.label;
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawNodeMarker(ctx: CanvasRenderingContext2D, theme: SymbolTheme, component: CanvasComponent): void {
  const text = component.name || component.value || "node";
  ctx.save();
  ctx.fillStyle = theme.accent;
  ctx.strokeStyle = theme.accent;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(8, 0);
  ctx.lineTo(22, 0);
  ctx.stroke();
  // Name above-right of the flag stub so neither a horizontal nor a
  // vertical wire through the marker runs through the text.
  ctx.fillStyle = theme.label;
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  drawUprightText(ctx, component, text, 10, -10, "left");
  ctx.restore();
}

export function drawComponent(
  ctx: CanvasRenderingContext2D,
  component: CanvasComponent,
  theme: SymbolTheme = DEFAULT_THEME
): void {
  ctx.save();
  ctx.translate(component.x, component.y);
  if (component.rotation) {
    ctx.rotate((component.rotation * Math.PI) / 180);
  }
  if (component.mirrored) {
    ctx.scale(-1, 1);
  }
  switch (component.type) {
    case "R":
      drawResistor(ctx, theme);
      break;
    case "C":
      drawCapacitor(ctx, theme);
      break;
    case "L":
      drawInductor(ctx, theme);
      break;
    case "V":
      drawVoltageSource(ctx, theme);
      break;
    case "I":
      drawCurrentSource(ctx, theme);
      break;
    case "D":
      drawDiode(ctx, theme);
      break;
    case "GND":
      drawGround(ctx, theme);
      break;
    case "VCVS":
      drawControlledSource(ctx, theme, "E");
      break;
    case "VCCS":
      drawControlledSource(ctx, theme, "G");
      break;
    case "CCCS":
      drawControlledSource(ctx, theme, "F");
      break;
    case "CCVS":
      drawControlledSource(ctx, theme, "H");
      break;
    case "SUBCKT":
      drawSubckt(ctx, theme, component);
      break;
    case "LABEL":
      drawLabel(ctx, theme, component);
      break;
    case "NODE":
      drawNodeMarker(ctx, theme, component);
      break;
    case "QNPN":
      drawBJT(ctx, theme, "npn");
      break;
    case "QPNP":
      drawBJT(ctx, theme, "pnp");
      break;
    case "NMOS":
      drawMOSFET(ctx, theme, "n");
      break;
    case "PMOS":
      drawMOSFET(ctx, theme, "p");
      break;
  }
  ctx.restore();
}

/** Draws a small palette preview of a device type, centered at (0,0). */
export function drawPalettePreview(
  ctx: CanvasRenderingContext2D,
  type: DeviceType,
  theme: SymbolTheme = DEFAULT_THEME
): void {
  // Text-based symbols are illegible at palette scale, so use dedicated glyphs.
  if (type === "LABEL") {
    ctx.save();
    ctx.fillStyle = theme.label;
    ctx.font = "700 44px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("T", 0, 2);
    ctx.restore();
    return;
  }
  if (type === "NODE") {
    ctx.save();
    ctx.strokeStyle = theme.stroke;
    ctx.fillStyle = theme.accent;
    ctx.lineWidth = 4;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(-30, 0);
    ctx.lineTo(-6, 0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-6, -14);
    ctx.lineTo(18, -14);
    ctx.lineTo(30, 0);
    ctx.lineTo(18, 14);
    ctx.lineTo(-6, 14);
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(-30, 0, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }
  if (type === "SUBCKT") {
    ctx.save();
    ctx.strokeStyle = theme.stroke;
    ctx.fillStyle = theme.fill;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.rect(-20, -22, 40, 44);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-34, -10);
    ctx.lineTo(-20, -10);
    ctx.moveTo(-34, 10);
    ctx.lineTo(-20, 10);
    ctx.moveTo(20, 0);
    ctx.lineTo(34, 0);
    ctx.stroke();
    ctx.fillStyle = theme.label;
    ctx.font = "700 22px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("X", 0, 1);
    ctx.restore();
    return;
  }
  const fake: CanvasComponent = {
    id: "preview",
    type,
    name: type,
    value: "",
    x: 0,
    y: 0,
    rotation: 0
  };
  drawComponent(ctx, fake, theme);
}
