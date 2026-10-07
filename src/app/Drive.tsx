"use client";

// drop image -> Florence-2 caption (in this browser) -> ob-road (or Jev) decides with caption + zone -> game acts
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import {
  OWN_LANE_Y,
  VIEW_H,
  VIEW_W,
  addObject,
  applyDecision,
  beginDecision,
  createGame,
  moveObject,
  removeObject,
  toScreenX,
  update,
  type Game,
  type GameObject,
} from "@/game/engine";
import { offlineDecide } from "@/game/offline";
import { ACTION_COLORS, ACTION_LABELS, drawGame, hitTest } from "@/game/render";
import { localDecide, modelText } from "@/shared/localDecide";
import type { DecideResponse, Zone } from "@/shared/types";
import { Vision, type VisionStatus } from "@/vision/client";
import Gauge from "./Gauge";

const SAMPLES = [
  // people and animals
  "child.jpeg",
  "dog.jpeg",
  "cat.avif",
  "snake.jpg",
  "bison.webp",
  // vehicles
  "trishaw.jpeg",
  "van.jpeg",
  // signs
  "crosswalk.jpeg",
  "crossing-sign.avif",
  "caution.webp",
  "stop.webp",
  "limit20.jpeg",
  "limit60.webp",
  "limit80.avif",
  // traffic lights
  "red-light.jpg",
  "red-light-street.webp",
  "amber-light.webp",
  "green-light.webp",
  "traffic-green.jpg",
  // harmless
  "tshirt.jpeg",
  "plastic-bags.jpg",
  "leaves.jpg",
].map((file) => ({ name: file.replace(/\.\w+$/, ""), url: `/samples/${file}` }));

const ZONE_LABELS: Record<Zone, string> = {
  far_sidewalk: "far sidewalk",
  oncoming_lane: "oncoming lane",
  own_lane: "my lane",
  near_sidewalk: "near sidewalk",
};

const CRUISE_MIN = 20;
const CRUISE_MAX = 120;
const CRUISE_STEP = 10;

const PRICE_PER_TOKEN = 0.042 / 1_000_000;
const DT = 1 / 60;

type Thought = { key: string; url: string; caption: string; zone: Zone; visionMs?: number; roundTripMs: number; d: DecideResponse; error?: string };
type JevMode = "waiting" | "local" | "jev" | "offline";
type Brain = "local" | "jev";

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = document.createElement("img");
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load image"));
    img.src = url;
  });
}

export default function Drive() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameRef = useRef<Game>(createGame());
  const visionRef = useRef<Vision | null>(null);
  const dropYRef = useRef<number | null>(null);
  const aborts = useRef(new Map<number, AbortController>());

  const [vision, setVision] = useState<VisionStatus>({ state: "loading", loaded: 0, total: 0 });
  const [hud, setHud] = useState({ speedKmh: 0, cruiseKmh: 50 });
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [jevMode, setJevMode] = useState<JevMode>("waiting");
  const [brain, setBrain] = useState<Brain>("local");
  const brainRef = useRef<Brain>("local"); // read inside async decide calls
  // user's own Jev token, set only after a test call works: kept in memory, gone on reload, never saved
  const jevTokenRef = useRef("");
  const [uploads, setUploads] = useState<{ name: string; url: string }[]>([]);
  const [trayOver, setTrayOver] = useState(false);

  useEffect(() => {
    const ctx = canvasRef.current!.getContext("2d")!;
    const v = new Vision(setVision);
    visionRef.current = v;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let frame = 0;
    const loop = (now: number) => {
      const g = gameRef.current;
      acc += Math.min(0.1, (now - last) / 1000);
      last = now;
      while (acc >= DT) {
        acc -= DT;
        update(g, DT);
      }
      drawGame(ctx, g, dropYRef.current);
      if (++frame % 6 === 0) setHud({ speedKmh: g.speed * 3.6, cruiseKmh: g.cruiseKmh });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      v.dispose();
    };
  }, []);

  // ask ob-road (in the browser) or Jev (through our server) what to do about one object
  async function decideFor(o: GameObject) {
    const g = gameRef.current;
    const seq = beginDecision(o);
    aborts.current.get(o.id)?.abort(); // a newer question replaces the old one
    const ctrl = new AbortController();
    aborts.current.set(o.id, ctrl);
    const caption = o.caption ?? "";
    const zone = o.zone;
    const started = performance.now();
    let d: DecideResponse;
    let error: string | undefined;

    if (!caption) {
      d = offlineDecide("", zone); // vision failed: fall back to zone rules
      error = "no caption";
    } else if (brainRef.current === "local") {
      try {
        const r = await visionRef.current!.classify(modelText(caption, zone));
        if (ctrl.signal.aborted) return;
        d = localDecide(r.logits, r.ms);
        setJevMode("local");
      } catch (err) {
        if (ctrl.signal.aborted) return;
        error = (err as Error).message;
        setJevMode("offline");
        d = offlineDecide(caption, zone);
      }
    } else if (!jevTokenRef.current) {
      d = offlineDecide(caption, zone);
      error = "enter your Jev token";
      setJevMode("offline");
    } else {
      try {
        const res = await fetch("/api/decide", {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-jev-token": jevTokenRef.current },
          body: JSON.stringify({ caption, zone }),
          signal: ctrl.signal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error === "no_key" ? "enter your Jev token" : (data.error ?? `HTTP ${res.status}`));
        d = data as DecideResponse;
        setJevMode("jev");
      } catch (err) {
        if (ctrl.signal.aborted) return;
        error = (err as Error).message;
        setJevMode("offline");
        d = offlineDecide(caption, zone);
      }
    }
    if (aborts.current.get(o.id) === ctrl) aborts.current.delete(o.id);
    if (!g.objects.includes(o) || seq !== o.seq) return;
    applyDecision(g, o, seq, d);
    const thought: Thought = {
      key: `${o.id}-${seq}`,
      url: o.url,
      caption,
      zone,
      visionMs: o.visionMs,
      roundTripMs: Math.round(performance.now() - started),
      d,
      error,
    };
    setThoughts((t) => [thought, ...t].slice(0, 40));
  }

  async function spawn(url: string, sx: number, sy: number) {
    let img: HTMLImageElement;
    try {
      img = await loadImage(url);
    } catch {
      return;
    }
    const g = gameRef.current;
    const o = addObject(g, img, url, sx, sy);
    try {
      const c = await visionRef.current!.caption(url);
      o.caption = c.text;
      o.visionMs = c.ms;
    } catch (err) {
      console.error("caption failed", err);
    }
    if (g.objects.includes(o)) await decideFor(o);
  }

  function remove(id: number) {
    aborts.current.get(id)?.abort();
    aborts.current.delete(id);
    removeObject(gameRef.current, id);
  }

  // client coords -> canvas coords (the canvas is scaled by CSS)
  function toCanvas(e: { clientX: number; clientY: number }) {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * VIEW_W, y: ((e.clientY - r.top) / r.height) * VIEW_H, inside: e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom };
  }

  // move objects already on the road
  const drag = useRef<{ obj: GameObject; dx: number; dy: number } | null>(null);
  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = toCanvas(e);
    const hit = hitTest(gameRef.current, p.x, p.y);
    if (!hit) return;
    if (hit.remove) return remove(hit.obj.id);
    const sx = toScreenX(gameRef.current, hit.obj.x);
    drag.current = { obj: hit.obj, dx: p.x - sx, dy: p.y - hit.obj.y };
    gameRef.current.dragging = hit.obj.id;
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!drag.current) return;
    const p = toCanvas(e);
    moveObject(gameRef.current, drag.current.obj, p.x - drag.current.dx, p.y - drag.current.dy);
    dropYRef.current = p.inside ? p.y : null;
  }
  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    dropYRef.current = null;
    gameRef.current.dragging = null;
    if (!toCanvas(e).inside) return remove(d.obj.id); // dragged off the road
    if (d.obj.caption !== undefined) void decideFor(d.obj); // moved: ask again
  }

  // drops from the sample tray or from the desktop
  function onDrop(e: React.DragEvent<HTMLCanvasElement>) {
    e.preventDefault();
    dropYRef.current = null;
    const p = toCanvas(e);
    const added = addUploads(e.dataTransfer.files);
    if (added.length) {
      added.forEach((a, i) => void spawn(a.url, p.x + i * 80, p.y));
      return;
    }
    const url = e.dataTransfer.getData("text/x-sample");
    if (url) void spawn(url, p.x, p.y);
  }

  // put image files in the tray, return them
  function addUploads(files: FileList | null) {
    const added = [...(files ?? [])].filter((f) => f.type.startsWith("image/")).map((f) => ({ name: f.name, url: URL.createObjectURL(f) }));
    if (added.length) setUploads((u) => [...added, ...u]);
    return added;
  }

  function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    addUploads(e.target.files);
    e.target.value = "";
  }

  // only desktop files light up the tray, not samples dragged out of it
  const hasFiles = (e: React.DragEvent) => e.dataTransfer.types.includes("Files");

  function setCruise(kmh: number) {
    kmh = Math.min(CRUISE_MAX, Math.max(CRUISE_MIN, kmh));
    gameRef.current.cruiseKmh = kmh;
    setHud((h) => ({ ...h, cruiseKmh: kmh }));
  }

  // taps queue up along my lane instead of stacking on one spot
  function nextTapX() {
    const g = gameRef.current;
    const xs = g.objects.filter((o) => o.zone === "own_lane").map((o) => toScreenX(g, o.x) + 110);
    return Math.max(VIEW_W - 140, ...xs);
  }

  const tray = [...uploads, ...SAMPLES];

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,#1a1040_0%,#07071a_55%)] text-[#e7e9ff]">
      <header className="flex flex-wrap items-center gap-3 border-b border-white/5 px-5 py-2">
        <h1 className="flex items-center gap-3">
          <Image src="/logo/jevcity.png" alt="JevCity" width={128} height={64} priority className="h-14 w-auto drop-shadow-[0_0_12px_#ff2e9766]" />
          <span className="text-sm text-[#8a8fc7]">night drive</span>
        </h1>
        <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
          <VisionPill s={vision} />
          <JevPill mode={jevMode} />
          <div className="flex overflow-hidden rounded-full border border-white/10" role="group" aria-label="Decision model">
            {(["local", "jev"] as const).map((b) => (
              <button
                key={b}
                onClick={() => {
                  brainRef.current = b;
                  setBrain(b);
                }}
                aria-pressed={brain === b}
                className={`px-3 py-1 ${brain === b ? "bg-white/15 text-white" : "text-[#8a8fc7] hover:text-[#c7cbf5]"}`}
              >
                {b === "local" ? "ob-road" : "Jev"}
              </button>
            ))}
          </div>
          {/* stays mounted so the token survives switching to ob-road and back */}
          <div className={brain === "jev" ? "contents" : "hidden"}>
            <JevTokenField onChange={(t) => (jevTokenRef.current = t)} />
          </div>
        </div>
      </header>

      <main className="grid gap-4 p-4 xl:grid-cols-[1fr_360px]">
        <section className="flex min-w-0 flex-col gap-3">
          <div className="relative overflow-hidden rounded-2xl border border-[#2a2a55] shadow-[0_0_40px_-10px_#ff2e9766]">
            <canvas
              ref={canvasRef}
              width={VIEW_W}
              height={VIEW_H}
              className="block h-auto w-full touch-none"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onDragOver={(e) => {
                e.preventDefault();
                dropYRef.current = toCanvas(e).y;
              }}
              onDragLeave={() => (dropYRef.current = null)}
              onDrop={onDrop}
            />
            {/* speedometer on the road view; only the buttons take clicks */}
            <div className="pointer-events-none absolute right-3 top-3 flex flex-col items-center gap-1 rounded-2xl bg-[#0c0b22]/70 px-3 py-2 backdrop-blur-sm">
              <Gauge kmh={hud.speedKmh} cruiseKmh={hud.cruiseKmh} />
              <div className="pointer-events-auto flex items-center gap-2 text-xs text-[#8a8fc7]">
                <CruiseButton label="−" disabled={hud.cruiseKmh <= CRUISE_MIN} onClick={() => setCruise(hud.cruiseKmh - CRUISE_STEP)} />
                <span className="w-24 text-center">
                  Cruise <span className="font-mono text-[#e7e9ff]">{hud.cruiseKmh}</span>
                </span>
                <CruiseButton label="+" disabled={hud.cruiseKmh >= CRUISE_MAX} onClick={() => setCruise(hud.cruiseKmh + CRUISE_STEP)} />
              </div>
            </div>
          </div>

          {/* sample dock */}
          <div
            onDragOver={(e) => {
              if (!hasFiles(e)) return;
              e.preventDefault();
              setTrayOver(true);
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setTrayOver(false);
            }}
            onDrop={(e) => {
              if (!hasFiles(e)) return;
              e.preventDefault();
              setTrayOver(false);
              addUploads(e.dataTransfer.files);
            }}
            className={`rounded-2xl border bg-[#0c0b22]/80 p-3 transition ${trayOver ? "border-dashed border-[#ff2e97] bg-[#ff2e97]/10" : "border-[#2a2a55]"}`}
          >
            <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-[#8a8fc7]">
              <span className="font-semibold uppercase tracking-widest text-[#c7cbf5]">Drop things on the road</span>
              <span>{trayOver ? "Drop to add your images" : "or drop image files here"}</span>
              <label className="ml-auto cursor-pointer rounded-full border border-dashed border-[#ff2e97] px-3 py-1 text-[#ff8cc4] hover:bg-[#ff2e97]/10">
                + Your images
                <input type="file" accept="image/*" multiple onChange={onUpload} className="hidden" />
              </label>
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(4.5rem,1fr))] gap-2">
              {tray.map((s) => (
                <button
                  key={s.url}
                  type="button"
                  title={s.name}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData("text/x-sample", s.url);
                    void visionRef.current?.caption(s.url).catch(() => {}); // start captioning while dragging
                  }}
                  onClick={() => void spawn(s.url, nextTapX(), OWN_LANE_Y)}
                  className="group flex cursor-grab flex-col items-center gap-1 rounded-lg border border-[#2a2a55] bg-black/30 p-1.5 transition hover:border-[#22d3ee] hover:shadow-[0_0_12px_#22d3ee]"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={s.url} alt={s.name} draggable={false} className="aspect-square w-full rounded-md bg-white/90 object-contain transition group-hover:scale-105" />
                  <span className="w-full truncate text-center text-[10px] text-[#8a8fc7]">{s.name}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Jev's thoughts */}
        <aside className="flex max-h-[calc(100vh-110px)] min-h-64 flex-col rounded-2xl border border-[#2a2a55] bg-[#0c0b22]/80">
          <h2 className="border-b border-white/5 px-4 py-3 text-xs font-semibold uppercase tracking-widest text-[#c7cbf5]">Radio log · what the car thinks</h2>
          <ol className="flex flex-1 flex-col gap-2 overflow-y-auto p-3">
            {thoughts.length === 0 && <li className="p-2 text-sm text-[#8a8fc7]">Nothing yet. Drop a photo on the road.</li>}
            {thoughts.map((t) => (
              <ThoughtCard key={t.key} t={t} />
            ))}
          </ol>
          <p className="border-t border-white/5 px-4 py-2 text-[11px] text-[#6b6f9e]">
            Vision: Florence-2 in your browser. Decisions: TypeSafe Jev.
          </p>
        </aside>
      </main>
    </div>
  );
}

function CruiseButton({ label, disabled, onClick }: { label: string; disabled: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-label={label === "+" ? "Faster cruise" : "Slower cruise"}
      disabled={disabled}
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-full border border-[#ff2e97] text-lg leading-none text-[#ff8cc4] transition hover:bg-[#ff2e97]/20 disabled:opacity-30"
    >
      {label}
    </button>
  );
}

function VisionPill({ s }: { s: VisionStatus }) {
  if (s.state === "ready")
    return <span className="rounded-full border border-[#4ade80]/40 bg-[#4ade80]/10 px-3 py-1 text-[#86efac]"> {s.backend === "webgpu" ? "WebGPU" : "CPU (slower)"}</span>;
  if (s.state === "error") return <span className="rounded-full border border-[#ff3b6b]/40 bg-[#ff3b6b]/10 px-3 py-1 text-[#ff9db5]">Vision failed: {s.message}</span>;
  const pct = s.total ? (s.loaded / s.total) * 100 : 0;
  return (
    <span className="flex items-center gap-2 rounded-full border border-[#c084fc]/40 bg-[#c084fc]/10 px-3 py-1 text-[#e1c6ff]">
      Loading Florence-2 {s.total ? `${Math.round(s.loaded / 1e6)} / ${Math.round(s.total / 1e6)} MB` : "…"}
      <span className="h-1.5 w-20 overflow-hidden rounded bg-white/10">
        <span className="block h-full bg-[#c084fc]" style={{ width: `${pct}%` }} />
      </span>
    </span>
  );
}

// token box: test it with one real Jev call, collapse when it works, Edit to change it
function JevTokenField({ onChange }: { onChange: (token: string) => void }) {
  const [value, setValue] = useState("");
  const [state, setState] = useState<"edit" | "testing" | "ok">("edit");
  const [error, setError] = useState("");

  async function test() {
    const token = value.trim();
    if (!token) return;
    setState("testing");
    setError("");
    try {
      const res = await fetch("/api/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-jev-token": token },
        body: JSON.stringify({ caption: "A dog sitting on the grass.", zone: "own_lane" }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      onChange(token);
      setState("ok");
    } catch (err) {
      setError((err as Error).message);
      setState("edit");
    }
  }

  if (state === "ok")
    return (
      <span className="flex items-center gap-2 rounded-full border border-[#22d3ee]/40 bg-[#22d3ee]/10 py-1 pl-3 pr-1 text-[#a5f3fc]">
        Jev connected · ••••{value.trim().slice(-4)}
        <button
          onClick={() => {
            onChange(""); // stop using it until the new one is tested
            setState("edit");
          }}
          className="rounded-full px-2 text-[#c7cbf5] hover:bg-white/10"
        >
          Edit
        </button>
      </span>
    );

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void test();
      }}
      className="flex items-center gap-1"
    >
      <input
        type="password"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          setError("");
        }}
        placeholder="Your Jev token"
        title="Only kept in this tab's memory and sent with each Jev request. Never saved."
        autoComplete="off"
        spellCheck={false}
        disabled={state === "testing"}
        className="w-44 rounded-full border border-white/10 bg-black/30 px-3 py-1 text-[#e7e9ff] placeholder:text-[#6b6f9e] focus:border-[#22d3ee]/60 focus:outline-none"
      />
      <button
        type="submit"
        disabled={!value.trim() || state === "testing"}
        className="rounded-full border border-white/10 px-3 py-1 text-[#c7cbf5] hover:border-white/30 disabled:opacity-40"
      >
        {state === "testing" ? "Testing…" : "Test"}
      </button>
      {error && <span className="text-[#ff9db5]">{error === "no_key" ? "enter a token" : error}</span>}
    </form>
  );
}

// segmented meter: thin rounded ticks, filled ones show the value
const TICKS = 30;
function Ticks({ value, color }: { value: number; color: string }) {
  const on = Math.round(value * TICKS);
  return (
    <span className="flex h-3 flex-1 items-stretch justify-between" aria-hidden>
      {Array.from({ length: TICKS }, (_, i) => (
        <span key={i} className="w-0.5 rounded-full" style={{ background: i < on ? color : "rgba(255,255,255,0.12)" }} />
      ))}
    </span>
  );
}

function JevPill({ mode }: { mode: JevMode }) {
  if (mode === "local") return <span className="rounded-full border border-[#4ade80]/40 bg-[#4ade80]/10 px-3 py-1 text-[#bbf7d0]">ob-road (in browser)</span>;
  if (mode === "jev") return <span className="rounded-full border border-[#22d3ee]/40 bg-[#22d3ee]/10 px-3 py-1 text-[#a5f3fc]">Jev live</span>;
  if (mode === "offline")
    return <span className="rounded-full border border-[#facc15]/40 bg-[#facc15]/10 px-3 py-1 text-[#fde68a]">Offline rules (model unavailable)</span>;
  return null;
}

function ThoughtCard({ t }: { t: Thought }) {
  const act = t.d.action.choice;
  const color = ACTION_COLORS[act];
  const top = Object.entries(t.d.action.probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  return (
    <li className="rounded-xl border border-white/5 bg-black/30 p-3">
      <div className="flex gap-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={t.url} alt="" className="h-11 w-11 shrink-0 rounded-md object-cover" />
        <div className="min-w-0 flex-1">
          <p className="text-sm italic text-[#c7cbf5]">&ldquo;{t.caption || "(no caption)"}&rdquo;</p>
          <p className="mt-1 text-[11px] text-[#8a8fc7]">
            {ZONE_LABELS[t.zone]} · looks like {t.d.category.choice.replaceAll("_", " ")}
          </p>
        </div>
      </div>
      <div className="mt-2 flex items-baseline justify-between">
        <span className="text-sm font-bold tracking-wide" style={{ color, textShadow: `0 0 10px ${color}` }}>
          {ACTION_LABELS[act]}
          {act === "change_speed" && ` ${t.d.speedLimit.choice} km/h`}
        </span>
        <span className="font-mono text-xs text-[#8a8fc7]">{Math.round(t.d.action.confidence * 100)}% sure</span>
      </div>
      {t.d.source !== "offline" && (
        <ul className="mt-1.5 flex flex-col gap-1">
          {top.map(([id, p], i) => (
            <li key={id} className="flex items-center gap-2 text-[10px] text-[#8a8fc7]">
              <span className="w-16 truncate">{id.replaceAll("_", " ")}</span>
              <Ticks value={p} color={i === 0 ? color : "#6b6f9e"} />
              <span className="w-8 text-right font-mono">{Math.round(p * 100)}%</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 font-mono text-[10px] text-[#6b6f9e]">
        {t.visionMs !== undefined && `vision ${t.visionMs} ms · `}
        {t.d.source === "offline" ? "offline rules" : `${t.d.source === "jev" ? "jev" : "ob-road"} ${t.d.latencyMs} ms · round trip ${t.roundTripMs} ms`}
        {t.d.inputTokens !== undefined && ` · ${t.d.inputTokens} tok ≈ $${(t.d.inputTokens * PRICE_PER_TOKEN).toFixed(6)}`}
      </p>
      {t.error && <p className="mt-1 text-[10px] text-[#ff9db5]">Decision model unavailable ({t.error}), used offline rules</p>}
    </li>
  );
}
