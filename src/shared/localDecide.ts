// Turns our own model's raw output into the same answer shape Jev gives.
// Model was trained on driving_dataset_new/: 4 zones, 3 base actions, 8 categories (see labels.json).
import type { Action, Answer, Category, DecideResponse, SpeedLimit, Zone } from "./types";

// must match the order in driving_dataset_new/labels.json (heads.json in the exported model)
const BASE_ACTIONS = ["go", "stop", "slow_down"] as const;
const BASE_CATEGORIES = ["animal", "person", "vehicle", "traffic_sign", "traffic_light", "obstacle", "harmless", "clear_road"] as const;
const BASE_LIMITS = ["none", "20", "30", "40", "50", "60", "80"] as const;
export const NUM_LOGITS = BASE_ACTIONS.length + BASE_CATEGORIES.length + BASE_LIMITS.length;

export const modelText = (caption: string, zone: Zone) => `[${zone}] ${caption}`;

function softmax(xs: number[]) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function head<T extends string>(names: readonly T[], logits: number[]) {
  const p = softmax(logits);
  const best = p.indexOf(Math.max(...p));
  return { choice: names[best], confidence: p[best], probabilities: Object.fromEntries(names.map((n, i) => [n, p[i]])) };
}

// same rules as driving_dataset_new/action_map.py, plus "go" for a green light
function gameAction(action: string, category: string, limit: string): Action {
  if (category === "traffic_light") return action === "go" ? "go" : "wait_for_green";
  if (category === "traffic_sign" && action === "stop") return "stop_then_go";
  if (category === "traffic_sign" && limit !== "none") return "change_speed"; // numbers on shirts or buses aren't limits
  if (action === "go") return "continue";
  return action as Action;
}

function appCategory(category: string, action: string, limit: string): Category {
  if (category === "traffic_sign") return limit !== "none" ? "speed_limit_sign" : action === "stop" ? "stop_sign" : "other_sign";
  if (category === "harmless" || category === "clear_road") return "harmless_debris";
  return category as Category;
}

export function localDecide(logits: number[], latencyMs: number): DecideResponse {
  // an old model with a different output size would give wrong answers, so fail and use offline rules
  if (logits.length !== NUM_LOGITS) throw new Error(`model gave ${logits.length} outputs, expected ${NUM_LOGITS}: retrain it`);
  const nA = BASE_ACTIONS.length;
  const nC = BASE_CATEGORIES.length;
  const a = head(BASE_ACTIONS, logits.slice(0, nA));
  const c = head(BASE_CATEGORIES, logits.slice(nA, nA + nC));
  const l = head(BASE_LIMITS, logits.slice(nA + nC));
  if (c.choice !== "traffic_sign") l.choice = "none"; // only signs carry a speed limit
  const act = gameAction(a.choice, c.choice, l.choice);
  const answer = <T extends string>(choice: T, from: { confidence: number; probabilities: Record<string, number> }): Answer<T> => ({
    choice,
    confidence: from.confidence,
    probabilities: from.probabilities,
  });
  return {
    action: answer(act, a),
    category: answer(appCategory(c.choice, a.choice, l.choice), c),
    speedLimit: answer(l.choice as SpeedLimit, l),
    source: "local",
    latencyMs,
  };
}
