"""
Synthetic Florence-style caption dataset for a multi-head driving classifier.

Input text  = "[zone] caption"   e.g. "[own_lane] A dog sitting on the grass."
Zone is where the photo was dropped (own_lane | oncoming_lane | near_sidewalk | far_sidewalk). Captions are written the way
Florence-2 describes a photo -- they do NOT say where the object is on the road.
Every caption is emitted once per zone, so the model must use the zone token.

Some captions end with the app's OCR text: ' The text on it reads "...".' Numbers there only
count as a speed limit on a traffic sign; on shirts, buses or boxes they are labeled 'none'.

Heads trained: action (go/stop/slow_down), category, speedLimit.
game_action is derived with action_map.to_game_action (not a training target).

Output: train / val / synth_test  (.jsonl + .csv), labels.json, stats.json
NOTE: synth_test comes from the same templates as train -> its score is NOT a real test.
      Use a hand-labeled set of real Florence captions (see caption_real.py) instead.

Run:  python generate_dataset.py [n_captions]   (default 40000 captions -> 160000 rows)
"""
import json, random, csv, sys
from collections import Counter
from action_map import to_game_action, GAME_ACTIONS

random.seed(42)
N_CAPTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000

ZONES = ["own_lane", "oncoming_lane", "near_sidewalk", "far_sidewalk"]   # must match the app's zone tokens

def zmap(own, oncoming, near, far):
    return dict(zip(ZONES, [own, oncoming, near, far]))
ACTIONS = ["go", "stop", "slow_down"]
CATEGORIES = ["animal", "person", "vehicle", "traffic_sign", "traffic_light", "obstacle", "harmless", "clear_road"]
SPEED_LIMITS = ["none", "20", "30", "40", "50", "60", "80"]

c = random.choice

# ---------- Florence-style wording ----------
def florence(core, background=""):
    """Wrap a caption core in one of Florence's output styles."""
    style = random.random()
    if style < 0.6:                                   # <CAPTION>
        s = core
    elif style < 0.85:                                # <DETAILED_CAPTION>
        s = "The image shows " + core + background
    else:
        s = c(["In this image we can see ", "This is a picture of ", "In this picture we can see "]) + core + background
    return s[0].upper() + s[1:] + "."

BACKGROUND = ["", "", " with trees in the background", " with a building in the background",
              " under a blue sky", " with a fence behind it", " with houses in the background",
              " with grass around it", " on a sunny day", " at night", " in the rain",
              " with mountains in the distance", " with cars in the background"]
PLACES = ["on the grass", "in a field", "on the ground", "on a dirt road", "on the sidewalk",
          "in front of a building", "next to a fence", "on the street", "on the beach",
          "in the snow", "on a road", "in a park", "on a concrete floor", "in a parking lot",
          "on a gravel path", "near a tree", "in the yard", "on a brick road", "in the mud", ""]
COLORS = ["", "", "brown ", "black ", "white ", "gray ", "small ", "large ", "young "]

ANIMALS = ["dog", "cat", "cow", "deer", "horse", "goat", "chicken", "duck", "sheep", "buffalo",
           "elephant", "monkey", "fox", "pig", "goose", "rabbit", "camel", "donkey", "puppy",
           "calf", "turkey", "bird", "squirrel", "kitten"]
ANIMAL_POSE = ["sitting", "standing", "lying", "walking", "running", "sleeping",
               "looking at the camera", "eating grass", "laying down"]

PEOPLE = ["a man", "a woman", "an older man", "an old woman",
          "a group of people", "a man in a red shirt", "a woman wearing a hat", "a police officer",
          "a construction worker", "two people", "a man wearing a backpack",
          "a woman in a white dress", "a person", "a young man"]
CHILDREN = ["a child", "a young boy", "a little girl", "a small child", "a toddler", "two kids",
            "a boy", "a girl", "a group of children", "a child holding a ball"]
PEOPLE_ACT = ["walking", "standing", "running", "holding an umbrella", "talking on a phone",
              "carrying a bag", "riding a bicycle", "pushing a stroller", "waving",
              "looking at the camera", "jogging", "smiling", "sitting"]

VEHICLES = ["car", "truck", "bus", "motorcycle", "van", "taxi", "tuk tuk", "suv", "pickup truck",
            "scooter", "school bus", "ambulance", "police car", "sedan", "minivan", "fire truck",
            "tractor", "delivery truck", "jeep", "sports car"]
VCOLORS = ["", "red ", "white ", "black ", "blue ", "silver ", "yellow ", "green ", "gray ", "old "]
V_PARKED = ["parked on the side of the road", "parked in a parking lot", "parked on the street",
            "parked in front of a building", "stopped on a road", "sitting in a driveway", "parked next to a curb"]
V_MOVING = ["driving down a road", "driving down the street", "on a highway", "driving on a city street",
            "traveling down a country road", "driving through an intersection"]

OBST_BLOCK = ["a large rock", "a fallen tree", "a cardboard box", "an old tire", "a pile of bricks",
              "a wooden pallet", "a broken branch", "a metal barrier", "a pile of sand", "a trash can",
              "a large log", "a plastic crate", "a wheelbarrow", "a mattress"]
OBST_MINOR = ["a pothole", "a puddle of water", "an orange traffic cone", "a row of traffic cones",
              "a crack in the asphalt", "a manhole cover", "a speed bump",
              "a small stone", "some gravel", "a muddy patch"]
OBST_PLACE = ["on the ground", "on the road", "in the street", "on the asphalt", "on the pavement", "in the woods", "on the floor", ""]

SCENES = ["an empty road", "a road with trees on both sides", "a city street with buildings",
          "a green field under a blue sky", "a park with trees", "a long straight highway",
          "a country road through the fields", "a quiet residential street", "a road leading to the mountains",
          "a view of the sky", "a forest", "a bridge over a river", "a sunset over the ocean",
          "a grassy hill", "an empty parking lot", "a road in the desert", "a wooden bench in a park",
          "a garden with flowers", "a building with windows", "a beach with palm trees"]

LIGHT_PLACE = ["", "", "at an intersection", "on a city street", "in front of a building",
               "against a cloudy sky", "next to a street lamp", "above a crosswalk", "on a busy road",
               "at a crossroads", "near some trees", "with a street sign below it"]
SIGN_PLACE = ["", "", "on the side of a road", "in front of a building", "on a street corner",
              "against a blue sky", "next to a tree", "on a metal pole", "near a fence",
              "on a sidewalk", "by a highway", "in a residential area"]
SCENE_MOD = ["", "", "", "on a sunny day", "in the evening", "with clouds in the sky",
             "in the morning", "at sunset", "covered in snow", "after the rain", "in autumn",
             "with no people", "in black and white"]

# Each generator returns (caption, {zone: action}, category, speedLimit)
def animal():
    core = f"a {c(COLORS)}{c(ANIMALS)} {c(ANIMAL_POSE)} {c(PLACES)}".strip()
    return florence(core, c(BACKGROUND)), zmap("stop", "slow_down", "slow_down", "slow_down"), "animal", "none"

def person():
    if random.random() < 0.3:   # children can dart out: slow down even on the far sidewalk
        who, zones = c(CHILDREN), zmap("stop", "slow_down", "slow_down", "slow_down")
    else:
        who, zones = c(PEOPLE), zmap("stop", "slow_down", "slow_down", "go")
    core = f"{who} {c(PEOPLE_ACT)} {c(PLACES)}".strip()
    return florence(core, c(BACKGROUND)), zones, "person", "none"

def vehicle():
    v = f"a {c(VCOLORS)}{c(VEHICLES)}".replace("a a", "an a").replace("a o", "an o")
    r = random.random()
    if r < 0.45:   core, own = f"{v} {c(V_PARKED)}", "stop"
    elif r < 0.8:  core, own = f"{v} {c(V_MOVING)}", "slow_down"
    else:          core, own = c([f"a close up of {v}", f"{v}", f"the front of {v}", f"{v} with its headlights on"]), "stop"
    return florence(core, c(BACKGROUND)), zmap(own, "go", "go", "go"), "vehicle", "none"

def stop_sign():
    core = c(["a red stop sign", "a stop sign", "a stop sign on a pole", "a red and white stop sign"]) + " " + \
           c(SIGN_PLACE)
    return florence(core.strip(), c(BACKGROUND)), {z: "stop" for z in ZONES}, "traffic_sign", "none"

def yield_sign():
    core = c(["a yield sign", "a red and white yield sign", "a triangular yield sign", "a give way sign"]) + " " + \
           c(SIGN_PLACE)
    return florence(core.strip(), c(BACKGROUND)), {z: "slow_down" for z in ZONES}, "traffic_sign", "none"

def speed_sign():
    s = c(SPEED_LIMITS[1:])
    core = c([f"a speed limit sign that says {s}", f"a white sign with the number {s} on it",
              f"a speed limit {s} sign", f"a round red and white sign with {s} on it",
              f"a sign that reads speed limit {s}", f"a {s} km/h speed limit sign"]) + " " + c(SIGN_PLACE)
    act = "slow_down" if int(s) <= 40 else "go"
    return florence(core.strip(), c(BACKGROUND)), {z: act for z in ZONES}, "traffic_sign", s

def school_sign():
    s = c(["20", "30"])
    core = c([f"a yellow school zone sign with a speed limit of {s}", f"a school crossing sign and a {s} sign",
              f"a sign that says school zone {s}"]) + " " + c(SIGN_PLACE)
    return florence(core.strip(), c(BACKGROUND)), {z: "slow_down" for z in ZONES}, "traffic_sign", s

def info_sign():
    core = c(["a no parking sign", "a blue street sign", "a green street name sign", "a one way sign",
              "a parking sign", "a bus stop sign", "a hospital sign"]) + " " + c(SIGN_PLACE)
    return florence(core.strip(), c(BACKGROUND)), {z: "go" for z in ZONES}, "traffic_sign", "none"

# warning signs: (caption core, OCR words) -> slow down in every zone
WARN_SIGNS = [
    ("a yellow sign that says slow down", ["SLOW DOWN", "SLOWDOWN", "SLOW"]),
    ("a slow sign", ["SLOW"]),
    ("an orange slow sign held by a worker", ["SLOW"]),
    ("a yellow diamond shaped warning sign", ["CAUTION", "WARNING", ""]),
    ("a caution sign", ["CAUTION"]),
    ("a pedestrian crossing sign", ["PED XING", "PEDESTRIAN CROSSING", ""]),
    ("a children crossing sign", ["CHILDREN", "SCHOOL", "CHILDREN CROSSING"]),
    ("a road work ahead sign", ["ROAD WORK AHEAD", "ROAD WORK"]),
    ("a men at work sign", ["MEN AT WORK"]),
    ("a sharp curve ahead sign", ["CURVE AHEAD", ""]),
    ("a speed bump sign", ["BUMP", "SPEED BUMP", "HUMP"]),
    ("a deer crossing sign", ["DEER CROSSING", ""]),
    ("a yellow sign with a picture of a person walking", ["", "XING"]),
    ("a narrow road sign", ["ROAD NARROWS", ""]),
]

def warning_sign():
    core, _ = c(WARN_SIGNS)
    return florence(f"{core} {c(SIGN_PLACE)}".strip(), c(BACKGROUND)), {z: "slow_down" for z in ZONES}, "traffic_sign", "none"

def traffic_light():
    color = random.choices(["red", "amber", "green"], [4, 3, 4])[0]
    word = {"red": "red", "amber": c(["yellow", "amber", "orange"]), "green": "green"}[color]
    core = c([f"a traffic light with the {word} light on", f"a {word} traffic light",
              f"a traffic light showing {word}", f"a {word} traffic light hanging over a street",
              f"a {word} traffic signal on a pole", f"a traffic light that is {word}"])
    core += " " + c(LIGHT_PLACE)
    act = {"red": "stop", "amber": "slow_down", "green": "go"}[color]
    return florence(core.strip(), c(BACKGROUND)), {z: act for z in ZONES}, "traffic_light", "none"

def obstacle():
    if random.random() < 0.55:
        core, own = f"{c(OBST_BLOCK)} {c(OBST_PLACE)}", "stop"
    else:
        core, own = f"{c(OBST_MINOR)} {c(OBST_PLACE)}", "slow_down"
    return florence(core.strip(), c(BACKGROUND)), zmap(own, "go", "go", "go"), "obstacle", "none"

HARMLESS = ["a plastic bag", "a white plastic bag", "a black plastic bag", "a crumpled piece of paper",
            "a newspaper", "a few dry leaves", "a pile of leaves", "a single leaf", "a paper cup",
            "an empty plastic bottle", "a candy wrapper", "a feather", "a piece of cloth", "a sock",
            "a napkin", "a flat piece of cardboard", "a receipt", "a flyer", "a crushed soda can",
            "a drinking straw", "a paper bag", "a tissue", "some fallen petals", "a balloon",
            "a piece of string", "a bottle cap", "a rubber band", "a plastic wrapper", "a cigarette butt"]
CLOTHES = ["a red t-shirt", "a white t-shirt", "a blue shirt", "a black hoodie", "a baseball cap",
           "a scarf", "a glove", "a pair of jeans", "a yellow shirt", "a towel"]
PRINTS = ["", "", " with the word stop on it", " with a stop sign printed on it", " with the word go on it",
          " with a logo on it", " with writing on it", " with the word slow on it", " with a picture of a car on it",
          " with the word danger on it"]
HARMLESS_PLACE = ["on the ground", "on the road", "on the asphalt", "on the grass", "blowing in the wind",
                  "floating in the air", "on a white background", "on the pavement", "on a wooden floor",
                  "on a table", "lying flat", ""]

def harmless():
    # light objects the car can drive over or through -> go in every zone
    item = c(CLOTHES) + c(PRINTS) if random.random() < 0.3 else c(HARMLESS)
    core = f"{item} {c(HARMLESS_PLACE)}".strip()
    return florence(core, c(BACKGROUND)), {z: "go" for z in ZONES}, "harmless", "none"

def scene():
    core = f"{c(SCENES)} {c(SCENE_MOD)}".strip()
    return florence(core, c(BACKGROUND)), {z: "go" for z in ZONES}, "clear_road", "none"

# ---------- OCR text, same ending the app adds (vision.worker.ts) ----------
LIMIT_LIKE = ["20", "30", "40", "50", "60", "80"]   # the hard case: these numbers off a sign
def any_number():
    return c(LIMIT_LIKE) if random.random() < 0.6 else str(random.randint(1, 199))

OCR_WORDS = {
    "person": lambda: c([any_number(), f"NO. {any_number()}", f"MARATHON {any_number()}", "POLICE", "NIKE",
                         f"TEAM {any_number()}", "STOP", "STAFF", f"RUN {any_number()}K"]),
    "vehicle": lambda: c([any_number(), f"Route {any_number()}", f"BUS {any_number()}", "TAXI",
                          f"ABC {random.randint(100, 999)}", f"{random.randint(1000, 9999)}", "SCHOOL BUS", "POLICE"]),
    "animal": lambda: c([any_number(), "ZOO", f"TAG {any_number()}"]),
    "obstacle": lambda: c([any_number(), "DANGER", "FRAGILE", "ROAD WORK", f"{any_number()} KG", "KEEP OUT"]),
    "harmless": lambda: c([any_number(), "STOP", "GO", "SALE", "SLOW", f"{any_number()}% OFF", "COCA COLA"]),
    "traffic_light": lambda: c([any_number(), "WAIT", "WALK"]),
    "clear_road": lambda: c([any_number(), "OPEN", "Main St", f"EXIT {any_number()}", "HOTEL", f"{any_number()} m"]),
}

def ocr_words(cap, cat, spd):
    """Text OCR would read on this thing, or '' for none."""
    if cat == "traffic_sign":
        if spd != "none": return c([f"SPEED LIMIT {spd}", spd, f"{spd} km/h", f"MAX {spd}", f"LIMIT {spd}", f"SCHOOL ZONE {spd}" if "school" in cap.lower() else spd])
        for core, words in WARN_SIGNS:
            if core in cap.lower(): return c(words)
        if "stop" in cap.lower(): return "STOP"
        if "yield" in cap.lower() or "give way" in cap.lower(): return c(["YIELD", "GIVE WAY"])
        return c(["NO PARKING", "ONE WAY", "BUS STOP", "HOSPITAL", "P", "Main St"])
    return OCR_WORDS[cat]()

def add_ocr(cap, cat, spd):
    # signs usually have readable text, other things only sometimes
    if random.random() >= (0.6 if cat == "traffic_sign" else 0.3): return cap
    words = ocr_words(cap, cat, spd)[:80]   # app keeps at most 80 chars
    return f'{cap} The text on it reads "{words}".' if words else cap

GENS = [(animal, 14), (person, 14), (vehicle, 14), (stop_sign, 5), (yield_sign, 3),
        (speed_sign, 8), (school_sign, 2), (info_sign, 3), (warning_sign, 5), (traffic_light, 10),
        (obstacle, 12), (harmless, 12), (scene, 10)]
funcs, weights = zip(*GENS)

captions, seen, tries = [], set(), 0
while len(captions) < N_CAPTIONS and tries < N_CAPTIONS * 60:
    tries += 1
    cap, zone_actions, cat, spd = random.choices(funcs, weights)[0]()
    cap = add_ocr(cap, cat, spd)
    if cap in seen:
        continue
    seen.add(cap)
    captions.append((cap, zone_actions, cat, spd))

# split by caption so both zone rows of a caption stay in the same split (no leakage)
random.shuffle(captions)
n = len(captions); a, b = int(n * .8), int(n * .9)
splits = {"train": captions[:a], "val": captions[a:b], "synth_test": captions[b:]}

FIELDS = ["id", "zone", "caption", "text", "action", "category", "speedLimit", "game_action"]
all_rows = []
for name, caps in splits.items():
    rows = []
    for i, (cap, zone_actions, cat, spd) in enumerate(caps):
        for z in ZONES:
            act = zone_actions[z]
            rows.append({"id": f"{name}-{i:05d}-{z}", "zone": z, "caption": cap,
                         "text": f"[{z}] {cap}", "action": act, "category": cat,
                         "speedLimit": spd, "game_action": to_game_action(act, cat, spd)})
    random.shuffle(rows)
    all_rows += rows
    with open(f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

json.dump({"zones": ZONES, "action": ACTIONS, "category": CATEGORIES,
           "speedLimit": SPEED_LIMITS, "game_action": GAME_ACTIONS},
          open("labels.json", "w"), indent=2)
stats = {"captions": n, "rows": len(all_rows),
         **{f"rows_{k}": len(v) * len(ZONES) for k, v in splits.items()},
         "action": Counter(r["action"] for r in all_rows),
         "action_by_zone": {z: Counter(r["action"] for r in all_rows if r["zone"] == z) for z in ZONES},
         "category": Counter(r["category"] for r in all_rows),
         "speedLimit": Counter(r["speedLimit"] for r in all_rows),
         "game_action": Counter(r["game_action"] for r in all_rows)}
json.dump(stats, open("stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
