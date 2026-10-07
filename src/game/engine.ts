// Game state and driving behavior. Jev picks an action per object; this code carries it out
// (speed, braking, timers). No browser APIs except the image handle, so it stays easy to read.
import type { Action, DecideResponse, Zone } from "@/shared/types";

export const VIEW_W = 1280;
export const VIEW_H = 520;
export const PX_PER_M = 8;
export const CAR_X = 230; // screen x of the car's center; the world scrolls past it
export const CAR_LEN = 4.5; // m
export const OBJ_PX = 64;
export const BANDS: Record<Zone, [number, number]> = {
  far_sidewalk: [70, 140],
  oncoming_lane: [140, 250],
  own_lane: [250, 360],
  near_sidewalk: [360, 430],
};
export const OWN_LANE_Y = (BANDS.own_lane[0] + BANDS.own_lane[1]) / 2;

const STOP_GAP = OBJ_PX / 2 / PX_PER_M + 2; // stop 2 m before the object's edge
const ACCEL = 3; // m/s^2
const COMFORT_BRAKE = 4;
const MAX_BRAKE = 9;
const STOP_SIGN_WAIT = 2; // s

export type GameObject = {
  id: number;
  url: string;
  img: HTMLImageElement;
  x: number; // world m
  y: number; // screen px
  zone: Zone;
  status: "seeing" | "thinking" | "decided";
  caption?: string;
  visionMs?: number;
  decision?: DecideResponse;
  seq: number; // bumps on every new decision request, so stale answers are dropped
  waited: number; // seconds stopped at a stop sign
  cleared: boolean; // stop sign done / light turned green
  applied: boolean; // speed sign already used
};

export type Game = {
  carX: number; // world m
  speed: number; // m/s
  cruiseKmh: number;
  braking: boolean;
  objects: GameObject[];
  nextId: number;
  status: string;
  stopLine: number | null; // world m, for drawing
  time: number;
  dragging: number | null; // id of object being moved by the user
};

export function createGame(): Game {
  return { carX: 0, speed: 0, cruiseKmh: 50, braking: false, objects: [], nextId: 1, status: "Cruising", stopLine: null, time: 0, dragging: null };
}

export const toScreenX = (g: Game, x: number) => CAR_X + (x - g.carX) * PX_PER_M;
export const toWorldX = (g: Game, sx: number) => g.carX + (sx - CAR_X) / PX_PER_M;

export function zoneAt(y: number): Zone {
  for (const [zone, [top, bottom]] of Object.entries(BANDS) as [Zone, [number, number]][]) {
    if (y >= top && y < bottom) return zone;
  }
  return y < BANDS.far_sidewalk[0] ? "far_sidewalk" : "near_sidewalk";
}

// keep the object fully inside its band
export function clampToZone(y: number): number {
  const [top, bottom] = BANDS[zoneAt(y)];
  return Math.max(top + 18, Math.min(bottom - 18, y));
}

export function addObject(g: Game, img: HTMLImageElement, url: string, sx: number, sy: number): GameObject {
  const y = clampToZone(sy);
  const o: GameObject = { id: g.nextId++, url, img, x: toWorldX(g, sx), y, zone: zoneAt(y), status: "seeing", seq: 0, waited: 0, cleared: false, applied: false };
  g.objects.push(o);
  return o;
}

export function moveObject(g: Game, o: GameObject, sx: number, sy: number) {
  o.x = toWorldX(g, sx);
  o.y = clampToZone(sy);
  o.zone = zoneAt(o.y);
}

export function removeObject(g: Game, id: number) {
  g.objects = g.objects.filter((o) => o.id !== id);
}

/** Start a new decision for this object; returns the ticket number. */
export function beginDecision(o: GameObject): number {
  o.status = "thinking";
  o.waited = 0;
  o.cleared = false;
  o.applied = false;
  return ++o.seq;
}

export function applyDecision(g: Game, o: GameObject, seq: number, d: DecideResponse) {
  if (seq !== o.seq) return; // a newer request replaced this one
  o.decision = d;
  o.status = "decided";
  // a green light releases every red light the car is waiting at
  if (d.action.choice === "go") {
    for (const other of g.objects) if (other.decision?.action.choice === "wait_for_green") other.cleared = true;
  }
}

export const shortLabel = (o: GameObject) => (o.caption ? o.caption.split(" ").slice(0, 5).join(" ").replace(/[.,]$/, "") : "something");

const blocks = (o: GameObject, act: Action | undefined) =>
  (o.status !== "decided" && o.zone === "own_lane") || // unknown thing in my lane: be careful until Jev answers
  act === "stop" ||
  ((act === "stop_then_go" || act === "wait_for_green") && !o.cleared);

export function update(g: Game, dt: number) {
  g.time += dt;
  const front = g.carX + CAR_LEN / 2;
  const cruise = g.cruiseKmh / 3.6;
  let target = cruise;
  let stopLine: number | null = null;
  let stopObj: GameObject | null = null;
  let slowObj: GameObject | null = null;

  for (const o of g.objects) {
    if (o.id === g.dragging) continue;
    const ahead = o.x - front;
    const act = o.decision?.action.choice;
    if (blocks(o, act) && ahead > -1) {
      const line = o.x - STOP_GAP;
      if (stopLine === null || line < stopLine) {
        stopLine = line;
        stopObj = o;
      }
    }
    if (act === "slow_down" && ahead < 25 && ahead > -CAR_LEN - 3) {
      target = Math.min(target, cruise / 2);
      slowObj = o;
    }
    // speed signs apply once the car reaches them
    if (act === "change_speed" && !o.applied && ahead <= 0) {
      o.applied = true;
      const limit = Number(o.decision!.speedLimit.choice);
      if (limit) g.cruiseKmh = limit;
    }
  }

  // fastest speed that still lets us stop at the line with comfortable braking
  let vmax = target;
  if (stopLine !== null) {
    const d = stopLine - front;
    vmax = d < 0.3 ? 0 : Math.min(vmax, Math.sqrt(2 * COMFORT_BRAKE * d));
  }
  const dv = vmax - g.speed;
  g.braking = dv < -0.05;
  g.speed = Math.max(0, g.speed + (dv > 0 ? Math.min(dv, ACCEL * dt) : Math.max(dv, -MAX_BRAKE * dt)));
  g.carX += g.speed * dt;
  g.stopLine = stopLine;

  // stop sign: count the wait once we're standing at it
  const stopAct = stopObj?.decision?.action.choice;
  if (stopObj && stopAct === "stop_then_go" && g.speed < 0.3 && stopLine! - front < 3) {
    stopObj.waited += dt;
    if (stopObj.waited >= STOP_SIGN_WAIT) stopObj.cleared = true;
  }

  // what the car is doing, in words
  if (stopObj) {
    const label = shortLabel(stopObj);
    if (stopObj.status === "seeing") g.status = "Something in my lane, looking at it…";
    else if (stopObj.status === "thinking") g.status = `Asking Jev about "${label}"…`;
    else if (stopAct === "wait_for_green") g.status = g.speed < 0.3 ? "Red light, waiting for green" : "Red light ahead, stopping";
    else if (stopAct === "stop_then_go")
      g.status = g.speed < 0.3 ? `Stop sign, going in ${Math.max(0, STOP_SIGN_WAIT - stopObj.waited).toFixed(1)} s` : "Stop sign ahead";
    else g.status = g.speed < 0.3 ? `Stopped for "${label}". Remove it to go on` : `Stopping for "${label}"`;
  } else if (slowObj) g.status = `Slowing down for "${shortLabel(slowObj)}"`;
  else g.status = "Cruising";

  // forget objects that scrolled off the left edge
  g.objects = g.objects.filter((o) => o.id === g.dragging || toScreenX(g, o.x) > -OBJ_PX * 2);
}
