// What Jev sees and is asked. Server-only, so the browser can't send its own prompts.
import type { Action, Category, SpeedLimit, Zone } from "@/shared/types";

const WHERE: Record<Zone, string> = {
  own_lane: "on the road, in the car's own lane, ahead of the car",
  oncoming_lane: "on the road, in the oncoming lane beside the car's lane",
  near_sidewalk: "on the sidewalk right next to the car's lane, off the road",
  far_sidewalk: "on the sidewalk on the far side of the road, past the oncoming lane, off the road",
};

export function buildState(caption: string, zone: Zone) {
  return { object_seen: caption, location: WHERE[zone] };
}

export const QUESTIONS = {
  action: {
    type: "choice",
    instructions: "What should the car do about `object_seen`, which is `location`?",
    criteria: {
      continue: "Keep driving at the same speed: it cannot get in the car's way, or it is soft and light enough to drive over, like a bag, paper or leaves",
      slow_down: "Drive on at half speed until past it: a person or animal that is near the car's lane but not in it, and could step into it",
      stop: "Stop before it and wait until it is gone: a person, animal, vehicle or solid object is blocking the car's lane",
      stop_then_go: "Stop at it, wait two seconds, then drive on: a stop sign",
      wait_for_green: "Stop at it and wait for a green light: a red or amber traffic light",
      go: "Drive on, and start again if waiting at a light: a green traffic light",
      change_speed: "Set the car's speed to the number shown: a speed limit sign",
    } satisfies Record<Action, string>,
  },
  category: {
    type: "choice",
    instructions: "What kind of thing is `object_seen`?",
    criteria: {
      person: "One or more people of any age",
      animal: "An animal such as a dog, cat, deer or bird",
      vehicle: "A vehicle such as a car, truck, bus, motorbike or bicycle",
      traffic_light: "A traffic light",
      stop_sign: "A stop sign",
      speed_limit_sign: "A speed limit sign",
      other_sign: "Any other sign",
      obstacle: "A solid object that could damage the car or block the road, like a box, rock, toy or furniture",
      harmless_debris: "Something light the car can drive over, like a plastic bag, paper or leaves",
      unclear: "`object_seen` is too vague to tell",
    } satisfies Record<Category, string>,
  },
  speed_limit: {
    type: "choice",
    instructions: "Which speed limit number is in `object_seen`?",
    criteria: {
      "20": "The number 20",
      "30": "The number 30",
      "40": "The number 40",
      "50": "The number 50",
      "60": "The number 60",
      "70": "The number 70",
      "80": "The number 80",
      "90": "The number 90",
      "100": "The number 100",
      "110": "The number 110",
      "120": "The number 120",
      none: "No speed limit number is in `object_seen`",
    } satisfies Record<SpeedLimit, string>,
  },
} as const;
