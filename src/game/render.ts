// Neon night look, top-down. Reads the game, never changes it.
import type { Action } from "@/shared/types";
import { BANDS, CAR_LEN, CAR_X, Game, GameObject, OBJ_PX, OWN_LANE_Y, PX_PER_M, VIEW_H, VIEW_W, toScreenX } from "./engine";

export const ACTION_COLORS: Record<Action, string> = {
  continue: "#4ade80",
  go: "#4ade80",
  slow_down: "#facc15",
  stop: "#ff3b6b",
  stop_then_go: "#ff6b9d",
  wait_for_green: "#ff8a3d",
  change_speed: "#b18cff",
};
export const ACTION_LABELS: Record<Action, string> = {
  continue: "CONTINUE",
  go: "GO",
  slow_down: "SLOW DOWN",
  stop: "STOP",
  stop_then_go: "STOP, THEN GO",
  wait_for_green: "WAIT FOR GREEN",
  change_speed: "NEW SPEED",
};
const SEEING = "#c084fc";
const THINKING = "#22d3ee";

const ROAD_TOP = BANDS.oncoming_lane[0];
const ROAD_BOTTOM = BANDS.own_lane[1];
const CENTER_Y = BANDS.own_lane[0];

// same building for the same slot every time, so the city doesn't flicker
function rand(n: number) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function glowLine(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, width: number, blur: number) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

// rooftops along a strip; `edgeY` is the side facing the street
function drawRooftops(ctx: CanvasRenderingContext2D, g: Game, top: number, bottom: number, edgeY: number, seed: number) {
  const offset = g.carX * PX_PER_M;
  const slot = 150;
  const first = Math.floor(offset / slot) - 1;
  const neon = ["#ff2e97", "#22d3ee", "#b18cff", "#facc15"];
  for (let i = first; i < first + VIEW_W / slot + 3; i++) {
    const r = rand(i + seed);
    const x = i * slot - offset + 6;
    const w = slot - 12 - r * 30;
    const shade = 18 + Math.floor(r * 12);
    ctx.fillStyle = `rgb(${shade},${shade - 4},${shade + 18})`;
    ctx.fillRect(x, top + 6, w, bottom - top - 12);
    // roof details: vents and a blinking beacon
    ctx.fillStyle = "rgba(255,255,255,0.05)";
    ctx.fillRect(x + 12, top + 18, 22, 14);
    ctx.fillRect(x + w - 40, top + 22 + r * 10, 16, 16);
    if (rand(i * 3 + seed) > 0.5 && Math.sin(g.time * 3 + i) > 0) {
      ctx.fillStyle = "#ff3b3b";
      ctx.shadowColor = "#ff3b3b";
      ctx.shadowBlur = 8;
      ctx.fillRect(x + w / 2, top + 14, 3, 3);
      ctx.shadowBlur = 0;
    }
    // neon sign strip on the street side
    if (r > 0.35) {
      const c = neon[Math.floor(rand(i + seed * 7) * neon.length)];
      const sx = x + 14 + r * 30;
      glowLine(ctx, sx, edgeY, sx + w * 0.45, edgeY, c, 3, 14);
    }
  }
}

function drawSidewalk(ctx: CanvasRenderingContext2D, g: Game, top: number, bottom: number) {
  ctx.fillStyle = "#17142a";
  ctx.fillRect(0, top, VIEW_W, bottom - top);
  const offset = (g.carX * PX_PER_M) % 32;
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let x = -offset; x < VIEW_W; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
  }
}

// pools of warm light from street lamps every 30 m on both sides
function drawStreetLights(ctx: CanvasRenderingContext2D, g: Game) {
  const spacing = 30 * PX_PER_M;
  const offset = (g.carX * PX_PER_M) % spacing;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let x = -offset; x < VIEW_W + spacing; x += spacing) {
    for (const [y, shift] of [
      [BANDS.far_sidewalk[0] + 14, 0],
      [BANDS.near_sidewalk[1] - 14, spacing / 2],
    ] as const) {
      const lx = x + shift;
      const pool = ctx.createRadialGradient(lx, y, 2, lx, y, 120);
      pool.addColorStop(0, "rgba(255,190,120,0.30)");
      pool.addColorStop(1, "rgba(255,190,120,0)");
      ctx.fillStyle = pool;
      ctx.fillRect(lx - 120, y - 120, 240, 240);
      ctx.fillStyle = "#ffe2b8";
      ctx.beginPath();
      ctx.arc(lx, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawRoad(ctx: CanvasRenderingContext2D, g: Game) {
  const grad = ctx.createLinearGradient(0, ROAD_TOP, 0, ROAD_BOTTOM);
  grad.addColorStop(0, "#0f1020");
  grad.addColorStop(0.5, "#13142a");
  grad.addColorStop(1, "#0f1020");
  ctx.fillStyle = grad;
  ctx.fillRect(0, ROAD_TOP, VIEW_W, ROAD_BOTTOM - ROAD_TOP);
  // curbs
  glowLine(ctx, 0, ROAD_TOP, VIEW_W, ROAD_TOP, "#ff2e97", 2, 12);
  glowLine(ctx, 0, ROAD_BOTTOM, VIEW_W, ROAD_BOTTOM, "#ff2e97", 2, 12);
  // dashed center line
  const dash = 6 * PX_PER_M;
  const offset = (g.carX * PX_PER_M) % (dash * 2);
  ctx.save();
  ctx.strokeStyle = "#22d3ee";
  ctx.shadowColor = "#22d3ee";
  ctx.shadowBlur = 10;
  ctx.lineWidth = 3;
  ctx.setLineDash([dash, dash]);
  ctx.lineDashOffset = offset;
  ctx.beginPath();
  ctx.moveTo(0, CENTER_Y);
  ctx.lineTo(VIEW_W, CENTER_Y);
  ctx.stroke();
  ctx.restore();
}

function drawCar(ctx: CanvasRenderingContext2D, g: Game) {
  const len = CAR_LEN * PX_PER_M;
  const wid = 18;
  const x = CAR_X;
  const y = OWN_LANE_Y;

  // headlight beams
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const side of [-1, 1]) {
    const hy = y + side * 6;
    const beam = ctx.createLinearGradient(x + len / 2, hy, x + len / 2 + 220, hy);
    beam.addColorStop(0, "rgba(220,240,255,0.42)");
    beam.addColorStop(1, "rgba(220,240,255,0)");
    ctx.fillStyle = beam;
    ctx.beginPath();
    ctx.moveTo(x + len / 2, hy - 2);
    ctx.lineTo(x + len / 2 + 220, hy - 34);
    ctx.lineTo(x + len / 2 + 220, hy + 34);
    ctx.lineTo(x + len / 2, hy + 2);
    ctx.fill();
  }
  ctx.restore();

  // tail lights flare when braking
  const tail = g.braking || g.speed < 0.2 ? 22 : 8;
  ctx.save();
  ctx.fillStyle = "#ff2244";
  ctx.shadowColor = "#ff2244";
  ctx.shadowBlur = tail;
  ctx.fillRect(x - len / 2 - 1, y - wid / 2 + 2, 3, 5);
  ctx.fillRect(x - len / 2 - 1, y + wid / 2 - 7, 3, 5);
  ctx.restore();

  // body
  ctx.save();
  const body = ctx.createLinearGradient(x, y - wid / 2, x, y + wid / 2);
  body.addColorStop(0, "#f5f7ff");
  body.addColorStop(1, "#b8c2e0");
  ctx.fillStyle = body;
  ctx.shadowColor = "#22d3ee";
  ctx.shadowBlur = 16;
  ctx.beginPath();
  ctx.roundRect(x - len / 2, y - wid / 2, len, wid, 6);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#1b1e3a"; // glass
  ctx.beginPath();
  ctx.roundRect(x + 2, y - wid / 2 + 3, 9, wid - 6, 3);
  ctx.roundRect(x - 13, y - wid / 2 + 3, 6, wid - 6, 2);
  ctx.fill();
  ctx.fillStyle = "#ff2e97"; // racing stripe
  ctx.fillRect(x - len / 2 + 2, y - 1.5, len - 4, 3);
  ctx.restore();
}

function chip(ctx: CanvasRenderingContext2D, text: string, cx: number, cy: number, color: string, filled: boolean) {
  ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
  const w = ctx.measureText(text).width + 14;
  ctx.save();
  ctx.fillStyle = filled ? color : "rgba(8,8,20,0.85)";
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = filled ? 12 : 0;
  ctx.beginPath();
  ctx.roundRect(cx - w / 2, cy - 9, w, 18, 9);
  ctx.fill();
  if (!filled) ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = filled ? "#0a0a16" : "#e7e9ff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy + 0.5);
  ctx.restore();
}

function drawObject(ctx: CanvasRenderingContext2D, g: Game, o: GameObject) {
  const sx = toScreenX(g, o.x);
  const half = OBJ_PX / 2;
  const act = o.decision?.action.choice;
  const pulse = 0.55 + 0.45 * Math.sin(g.time * 6);
  const color = act ? ACTION_COLORS[act] : o.status === "seeing" ? SEEING : THINKING;
  const done = act === "change_speed" ? o.applied : (act === "stop_then_go" || act === "wait_for_green") && o.cleared;

  ctx.save();
  if (done) ctx.globalAlpha = 0.45;
  ctx.shadowColor = color;
  ctx.shadowBlur = act ? 16 : 10 + 14 * pulse;
  ctx.fillStyle = "#0a0a16";
  ctx.beginPath();
  ctx.roundRect(sx - half - 3, o.y - half - 3, OBJ_PX + 6, OBJ_PX + 6, 10);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.roundRect(sx - half, o.y - half, OBJ_PX, OBJ_PX, 8);
  ctx.save();
  ctx.clip();
  // cover-fit the photo into the square
  const s = Math.max(OBJ_PX / o.img.naturalWidth, OBJ_PX / o.img.naturalHeight);
  const w = o.img.naturalWidth * s;
  const h = o.img.naturalHeight * s;
  ctx.drawImage(o.img, sx - w / 2, o.y - h / 2, w, h);
  ctx.restore();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.globalAlpha *= act ? 1 : pulse;
  ctx.stroke();
  ctx.restore();

  // status chip under it, caption above it
  const label = act ? ACTION_LABELS[act] + (act === "change_speed" ? ` ${o.decision!.speedLimit.choice}` : "") : o.status === "seeing" ? "LOOKING…" : "ASKING JEV…";
  chip(ctx, label, sx, o.y + half + 13, color, !!act);
  if (o.caption) {
    const cap = o.caption.length > 42 ? o.caption.slice(0, 40) + "…" : o.caption;
    chip(ctx, cap, sx, o.y - half - 13, "#4b4f7a", false);
  }

  // remove button
  ctx.save();
  ctx.fillStyle = "rgba(8,8,20,0.9)";
  ctx.strokeStyle = "#8a8fc7";
  ctx.beginPath();
  ctx.arc(sx + half, o.y - half, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "#e7e9ff";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(sx + half - 3.5, o.y - half - 3.5);
  ctx.lineTo(sx + half + 3.5, o.y - half + 3.5);
  ctx.moveTo(sx + half + 3.5, o.y - half - 3.5);
  ctx.lineTo(sx + half - 3.5, o.y - half + 3.5);
  ctx.stroke();
  ctx.restore();
}

export function drawGame(ctx: CanvasRenderingContext2D, g: Game, dropY: number | null) {
  ctx.fillStyle = "#07071a";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);

  drawRooftops(ctx, g, 0, BANDS.far_sidewalk[0], BANDS.far_sidewalk[0] - 3, 11);
  drawRooftops(ctx, g, BANDS.near_sidewalk[1], VIEW_H, BANDS.near_sidewalk[1] + 3, 77);
  drawSidewalk(ctx, g, ...BANDS.far_sidewalk);
  drawSidewalk(ctx, g, ...BANDS.near_sidewalk);
  drawRoad(ctx, g);
  drawStreetLights(ctx, g);

  // highlight the band under a dragged image
  if (dropY !== null) {
    const band = Object.values(BANDS).find(([t, b]) => dropY >= t && dropY < b);
    if (band) {
      ctx.fillStyle = "rgba(34,211,238,0.08)";
      ctx.fillRect(0, band[0], VIEW_W, band[1] - band[0]);
    }
  }

  if (g.stopLine !== null) {
    const x = toScreenX(g, g.stopLine) + 2;
    glowLine(ctx, x, BANDS.own_lane[0] + 6, x, BANDS.own_lane[1] - 6, "#ff3b6b", 3, 16);
  }

  drawCar(ctx, g);
  for (const o of g.objects) drawObject(ctx, g, o);
}

/** What's under the pointer: an object, or its remove button. */
export function hitTest(g: Game, sx: number, sy: number): { obj: GameObject; remove: boolean } | null {
  const half = OBJ_PX / 2;
  for (let i = g.objects.length - 1; i >= 0; i--) {
    const o = g.objects[i];
    const ox = toScreenX(g, o.x);
    if (Math.hypot(sx - (ox + half), sy - (o.y - half)) < 12) return { obj: o, remove: true };
    if (Math.abs(sx - ox) < half && Math.abs(sy - o.y) < half) return { obj: o, remove: false };
  }
  return null;
}
