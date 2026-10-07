// Main-thread side of the vision worker: load once, caption images, cache by URL.
import type { WorkerIn, WorkerOut } from "./vision.worker";

export type VisionStatus =
  | { state: "loading"; loaded: number; total: number }
  | { state: "ready"; backend: "webgpu" | "wasm"; loadMs: number }
  | { state: "error"; message: string };

export type Caption = { text: string; ms: number };
export type Classified = { logits: number[]; ms: number };

export class Vision {
  private worker: Worker;
  private nextId = 0;
  private waiting = new Map<number, { resolve: (c: Caption) => void; reject: (e: Error) => void }>();
  private classifying = new Map<number, { resolve: (c: Classified) => void; reject: (e: Error) => void }>();
  // same image always gets the same caption, so reuse it
  private cache = new Map<string, Promise<Caption>>();

  constructor(onStatus: (s: VisionStatus) => void) {
    this.worker = new Worker(new URL("./vision.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<WorkerOut>) => {
      const m = e.data;
      if (m.type === "progress") onStatus({ state: "loading", loaded: m.loaded, total: m.total });
      else if (m.type === "ready") onStatus({ state: "ready", backend: m.backend, loadMs: m.loadMs });
      else if (m.type === "caption") {
        this.waiting.get(m.id)?.resolve({ text: m.text, ms: m.ms });
        this.waiting.delete(m.id);
      } else if (m.type === "classified") {
        this.classifying.get(m.id)?.resolve({ logits: m.logits, ms: m.ms });
        this.classifying.delete(m.id);
      } else if (m.id === undefined) onStatus({ state: "error", message: m.message });
      else {
        const err = new Error(m.message);
        (this.waiting.get(m.id) ?? this.classifying.get(m.id))?.reject(err);
        this.waiting.delete(m.id);
        this.classifying.delete(m.id);
      }
    };
    onStatus({ state: "loading", loaded: 0, total: 0 });
    this.send({ type: "load" });
  }

  private send(m: WorkerIn) {
    this.worker.postMessage(m);
  }

  caption(url: string): Promise<Caption> {
    let p = this.cache.get(url);
    if (!p) {
      const id = this.nextId++;
      p = new Promise<Caption>((resolve, reject) => this.waiting.set(id, { resolve, reject }));
      this.send({ type: "caption", id, url: new URL(url, location.href).href });
      this.cache.set(url, p);
      p.catch(() => this.cache.delete(url)); // let failures retry
    }
    return p;
  }

  // run our own decision model on "[zone] caption" text
  classify(text: string): Promise<Classified> {
    const id = this.nextId++;
    const p = new Promise<Classified>((resolve, reject) => this.classifying.set(id, { resolve, reject }));
    this.send({ type: "classify", id, text });
    return p;
  }

  dispose() {
    this.worker.terminate();
  }
}
