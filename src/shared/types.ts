// Contract for POST /api/decide, used by both the browser and the server.

// Road bands from top to bottom of the screen. The car drives left to right in own_lane.
export const ZONES = ["far_sidewalk", "oncoming_lane", "own_lane", "near_sidewalk"] as const;
export type Zone = (typeof ZONES)[number];

export const ACTIONS = ["continue", "slow_down", "stop", "stop_then_go", "wait_for_green", "go", "change_speed"] as const;
export type Action = (typeof ACTIONS)[number];

export const CATEGORIES = [
  "person",
  "animal",
  "vehicle",
  "traffic_light",
  "stop_sign",
  "speed_limit_sign",
  "other_sign",
  "obstacle",
  "harmless_debris",
  "unclear",
] as const;
export type Category = (typeof CATEGORIES)[number];

export const SPEED_LIMITS = ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "none"] as const;
export type SpeedLimit = (typeof SPEED_LIMITS)[number];

export const CAPTION_MAX = 300;

export type DecideRequest = { caption: string; zone: Zone };

export type Answer<T extends string> = { choice: T; confidence: number; probabilities: Record<string, number> };

export type DecideResponse = {
  action: Answer<Action>;
  category: Answer<Category>;
  speedLimit: Answer<SpeedLimit>;
  source: "jev" | "local" | "offline";
  latencyMs: number;
  inputTokens?: number;
};
