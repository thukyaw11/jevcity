// Server proxy to TypeSafe Jev. Uses the user's own Jev token from the request; never stored or logged.
// Schema: https://docs.typesafe.ai/api
import { CAPTION_MAX, ZONES, type Answer, type DecideResponse, type Zone } from "@/shared/types";
import { QUESTIONS, buildState } from "./policy";

const TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone";

type JevChoice = { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> };

export async function POST(request: Request) {
  const key = request.headers.get("x-jev-token")?.trim();
  if (!key) return Response.json({ error: "no_key" }, { status: 401 });

  const body = await request.json().catch(() => null);
  const caption = typeof body?.caption === "string" ? body.caption.trim() : "";
  const zone = body?.zone as Zone;
  if (!caption || caption.length > CAPTION_MAX || !ZONES.includes(zone)) {
    return Response.json({ error: "Expected { caption: 1-300 chars, zone }" }, { status: 400 });
  }

  const started = Date.now();
  let upstream: Response;
  try {
    upstream = await fetch(TYPESAFE_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ state: buildState(caption, zone), model: "jev-latest", questions: QUESTIONS }),
      signal: AbortSignal.timeout(8000),
    });
  } catch (err) {
    return Response.json({ error: `Could not reach TypeSafe: ${(err as Error).message}` }, { status: 504 });
  }
  const data = await upstream.json().catch(() => null);
  if (!upstream.ok) return Response.json({ error: `TypeSafe returned ${upstream.status}`, detail: data }, { status: 502 });

  const a = data?.answers ?? {};
  if (a.action?.type !== "choice" || a.category?.type !== "choice" || a.speed_limit?.type !== "choice") {
    return Response.json({ error: "Unexpected TypeSafe response", detail: data }, { status: 502 });
  }

  const pick = <T extends string>(x: JevChoice): Answer<T> => ({
    choice: x.choice as T,
    confidence: x.confidence,
    probabilities: x.probabilities,
  });
  const result: DecideResponse = {
    action: pick(a.action),
    category: pick(a.category),
    speedLimit: pick(a.speed_limit),
    source: "jev",
    latencyMs: Date.now() - started,
    inputTokens: data.usage?.input_tokens,
  };
  return Response.json(result);
}
