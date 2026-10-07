// Runs Florence-2 in a Web Worker so captioning never freezes the game.
// Florence-2-base-ft, <CAPTION> task, 384 px.
// Also runs <OCR>, since captions skip the numbers on signs.
import {
  AutoModelForSequenceClassification,
  AutoProcessor,
  AutoTokenizer,
  env,
  Florence2ForConditionalGeneration,
  RawImage,
  type PreTrainedModel,
  type Processor,
} from "@huggingface/transformers";

export type WorkerIn = { type: "load" } | { type: "caption"; id: number; url: string } | { type: "classify"; id: number; text: string };
export type WorkerOut =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; backend: "webgpu" | "wasm"; loadMs: number }
  | { type: "caption"; id: number; text: string; ms: number }
  | { type: "classified"; id: number; logits: number[]; ms: number }
  | { type: "error"; id?: number; message: string };

const MODEL = "onnx-community/Florence-2-base-ft";
const REVISION = "e88a44eaf3791a35eae0c5a47b3dbcd36e67eb6f"; // pinned so answers don't drift
const DTYPES = {
  webgpu: { embed_tokens: "fp16", vision_encoder: "fp16", encoder_model: "q4", decoder_model_merged: "q4" },
  wasm: { embed_tokens: "q8", vision_encoder: "q8", encoder_model: "q4", decoder_model_merged: "q4" },
} as const;

const post = (m: WorkerOut) => self.postMessage(m);

// our own decision model, trained by driving_dataset_new/train_model.py, served from public/models/
env.allowLocalModels = true; // Florence still comes from the Hub when not found here
env.localModelPath = "/models/";
const CLASSIFIER = "driving-clf";
type Classifier = (text: string) => Promise<number[]>;
let classifier: Promise<Classifier> | null = null;

// drop our model from the browser cache so a retrained one is picked up (Florence stays cached)
async function clearClassifierCache() {
  try {
    const cache = await caches.open(env.cacheKey);
    const keys = await cache.keys();
    await Promise.all(keys.filter((r) => r.url.includes(`/models/${CLASSIFIER}/`)).map((r) => cache.delete(r)));
  } catch {}
}

async function loadClassifier(): Promise<Classifier> {
  await clearClassifierCache();
  const [model, tokenizer] = await Promise.all([
    AutoModelForSequenceClassification.from_pretrained(CLASSIFIER, { dtype: "q8", device: "wasm", local_files_only: true }),
    AutoTokenizer.from_pretrained(CLASSIFIER, { local_files_only: true }),
  ]);
  return async (text) => {
    const inputs = tokenizer(text, { truncation: true, max_length: 64 });
    const { logits } = await model(inputs);
    return Array.from(logits.data as Float32Array);
  };
}

type Captioner = (img: RawImage) => Promise<string>;
let captioner: Promise<Captioner> | null = null;
let queue: Promise<unknown> = Promise.resolve(); // one image at a time

async function pickBackend(): Promise<"webgpu" | "wasm"> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (await gpu?.requestAdapter()) return "webgpu";
  } catch {}
  return "wasm";
}

async function build(backend: "webgpu" | "wasm"): Promise<Captioner> {
  // sum progress across all files being downloaded
  const files = new Map<string, { loaded: number; total: number }>();
  const progress_callback = (e: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (e.status !== "progress" || !e.file) return;
    files.set(e.file, { loaded: e.loaded ?? 0, total: e.total ?? 0 });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: "progress", loaded, total });
  };

  const [model, processor] = await Promise.all([
    Florence2ForConditionalGeneration.from_pretrained(MODEL, {
      revision: REVISION,
      device: backend,
      dtype: DTYPES[backend],
      progress_callback,
    }) as Promise<PreTrainedModel>,
    AutoProcessor.from_pretrained(MODEL, { revision: REVISION, progress_callback }) as Promise<
      Processor & {
        construct_prompts(t: string): string[];
        post_process_generation(t: string, task: string, size: [number, number]): Record<string, unknown>;
        image_processor: { size: unknown };
      }
    >,
  ]);
  processor.image_processor.size = { height: 384, width: 384 };

  const run = async (img: RawImage, task: string) => {
    const inputs = await processor(img, processor.construct_prompts(task));
    const ids = await model.generate({ ...inputs, max_new_tokens: 60, do_sample: false });
    const text = processor.batch_decode(ids as never, { skip_special_tokens: false })[0] ?? "";
    const out = processor.post_process_generation(text, task, [img.width, img.height])[task];
    return typeof out === "string" ? out.replace(/<\/?s>|<pad>/g, "").trim() : "";
  };

  return async (img) => {
    const caption = await run(img, "<CAPTION>");
    const words = (await run(img, "<OCR>")).slice(0, 80); // keep under the server caption limit
    return words ? `${caption} The text on it reads "${words}".` : caption;
  };
}

async function load(): Promise<Captioner> {
  const started = performance.now();
  let backend = await pickBackend();
  let fn: Captioner;
  try {
    fn = await build(backend);
  } catch (err) {
    if (backend !== "webgpu") throw err;
    console.warn("WebGPU load failed, falling back to Wasm", err);
    backend = "wasm";
    fn = await build(backend);
  }
  await fn(new RawImage(new Uint8ClampedArray([127, 127, 127]), 1, 1, 3)); // warm up shaders
  post({ type: "ready", backend, loadMs: Math.round(performance.now() - started) });
  return fn;
}

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  if (msg.type === "load") {
    captioner ??= load();
    captioner.catch((err: Error) => post({ type: "error", message: err.message }));
    return;
  }
  if (msg.type === "classify") {
    classifier ??= loadClassifier();
    classifier.catch(() => (classifier = null)); // let a failed load retry
    void (async () => {
      try {
        const fn = await classifier!;
        const started = performance.now();
        const logits = await fn(msg.text);
        post({ type: "classified", id: msg.id, logits, ms: Math.round(performance.now() - started) });
      } catch (err) {
        post({ type: "error", id: msg.id, message: (err as Error).message });
      }
    })();
    return;
  }
  if (msg.type === "caption") {
    captioner ??= load();
    queue = queue.then(async () => {
      try {
        const fn = await captioner!;
        const started = performance.now();
        const img = await RawImage.fromURL(msg.url);
        const text = await fn(img);
        post({ type: "caption", id: msg.id, text, ms: Math.round(performance.now() - started) });
      } catch (err) {
        post({ type: "error", id: msg.id, message: (err as Error).message });
      }
    });
  }
};
