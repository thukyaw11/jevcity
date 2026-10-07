// Offline rules, used when there is no API key or Jev can't be reached.
// Plain keyword matching on the caption, so the game is still playable.
import type { Action, Answer, Category, DecideResponse, SpeedLimit, Zone } from "@/shared/types";

// a limit number not glued to other digits, so "80km/h" counts but "1800" doesn't
const LIMIT = /(?<!\d)(20|30|40|50|60|70|80|90|100|110|120)(?!\d)/;

function category(t: string): Category {
  if (/stop sign/.test(t)) return "stop_sign";
  if (/traffic (light|signal)|signal light/.test(t)) return "traffic_light";
  if (/speed limit|km\/h|mph/.test(t) || LIMIT.test(t)) return "speed_limit_sign";
  if (/\b(person|people|m[ae]n|wom[ae]n|boy|girl|child|children|kid|baby|pedestrian|player)s?\b/.test(t)) return "person";
  if (/\b(dog|cat|deer|bird|horse|cow|sheep|animal|puppy|kitten|fox)s?\b/.test(t)) return "animal";
  if (/\b(car|truck|bus|van|bicycle|bike|motorcycle|scooter|vehicle)s?\b/.test(t)) return "vehicle";
  if (/\b(bag|leaves|leaf|paper|litter|plastic)\b/.test(t)) return "harmless_debris";
  if (/\bsign\b/.test(t)) return "other_sign";
  return t ? "obstacle" : "unclear";
}

function action(cat: Category, t: string, zone: Zone): Action {
  const inLane = zone === "own_lane";
  switch (cat) {
    case "stop_sign":
      return "stop_then_go";
    case "traffic_light":
      return /green/.test(t) && !/red|amber|yellow/.test(t) ? "go" : "wait_for_green";
    case "speed_limit_sign":
      return "change_speed";
    case "person":
    case "animal":
      return inLane ? "stop" : zone === "far_sidewalk" ? "continue" : "slow_down";
    case "harmless_debris":
    case "other_sign":
      return "continue";
    default:
      return inLane ? "stop" : zone === "oncoming_lane" ? "slow_down" : "continue";
  }
}

const sure = <T extends string>(choice: T): Answer<T> => ({ choice, confidence: 1, probabilities: { [choice]: 1 } });

export function offlineDecide(caption: string, zone: Zone): DecideResponse {
  const t = caption.toLowerCase();
  const cat = category(t);
  const limit = (t.match(LIMIT)?.[1] ?? "none") as SpeedLimit;
  return {
    action: sure(action(cat, t, zone)),
    category: sure(cat),
    speedLimit: sure(limit),
    source: "offline",
    latencyMs: 0,
  };
}
