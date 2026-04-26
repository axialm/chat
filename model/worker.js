/**
 * worker.js — AxiaLM-1 inference worker
 * Runs entirely off the main thread so the browser never freezes.
 * Uses @xenova/transformers for tokenization (correct HF BPE implementation)
 * and ONNX Runtime Web for inference.
 *
 * Host this file alongside index.html in your GitHub Pages repo.
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.all.min.js');

// ── Config ────────────────────────────────────────────────────────────────────
const MODEL_URL  = './model/axialm1_int8.onnx';
const TOK_URL    = './model/tokenizer.json';
const CONFIG_URL = './model/config.json';

const SYSTEM_PROMPT =
    "You are AxiaLM-1, a helpful and friendly AI assistant. " +
    "You are a small language model with limitations — you may not always be " +
    "accurate, and you should say so when unsure rather than guessing confidently. " +
    "Be concise, honest, and conversational. " +
    "Format your responses using simple HTML: use <p> for paragraphs, " +
    "<ol> with <li> for numbered lists, <ul> with <li> for bullet lists, " +
    "and <strong> for bold emphasis. Always wrap your response in at least " +
    "one <p> tag. Do not use markdown.";

const GEN_MAX_NEW = 200;
const GEN_TEMP    = 0.75;
const GEN_TOP_K   = 40;
const GEN_TOP_P   = 0.90;
const MAX_CTX     = 1024;

// ── State ─────────────────────────────────────────────────────────────────────
let ortSession   = null;
let tokData      = null;
let modelConfig  = null;
let vocab        = null;   // token -> id
let invVocab     = null;   // id -> token
let mergeRank    = null;   // "a b" -> priority
let specialToks  = {};
let TOK_BOS, TOK_EOS, TOK_SEP, TOK_SYS, TOK_PAD;

// ── Byte-level BPE tokenizer ──────────────────────────────────────────────────
// Built directly from the tokenizer.json — same spec as Python tokenizers lib.

function buildByteSymbols() {
    // GPT-2 byte-to-unicode mapping
    const bs = [], cs = [];
    let n = 0;
    for (let b = 0; b < 256; b++) {
        const inRange = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        bs.push(b);
        cs.push(inRange ? b : 256 + n++);
    }
    const fwd = {}, inv = {};
    bs.forEach((b, i) => {
        fwd[b] = String.fromCodePoint(cs[i]);
        inv[String.fromCodePoint(cs[i])] = b;
    });
    return { fwd, inv };
}

const { fwd: BYTE_FWD, inv: BYTE_INV } = buildByteSymbols();

function byteEncode(str) {
    const bytes = new TextEncoder().encode(str);
    return Array.from(bytes).map(b => BYTE_FWD[b]).join('');
}

function bpeMerge(chars) {
    let word = [...chars];
    while (word.length > 1) {
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < word.length - 1; i++) {
            const pair = word[i] + ' ' + word[i + 1];
            const rank = mergeRank[pair];
            if (rank !== undefined && rank < bestRank) {
                bestRank = rank; bestIdx = i;
            }
        }
        if (bestIdx === -1) break;
        const merged = word[bestIdx] + word[bestIdx + 1];
        word = [...word.slice(0, bestIdx), merged, ...word.slice(bestIdx + 2)];
    }
    return word;
}

function encode(text, addPrefixSpace = true) {
    if (!text) return [];
    const prefixed = addPrefixSpace ? ' ' + text : text;
    const byteStr  = byteEncode(prefixed);
    const chars    = [...byteStr];
    const tokens   = bpeMerge(chars);
    const ids = [];
    for (const t of tokens) {
        const id = vocab[t];
        if (id !== undefined) ids.push(id);
        else ids.push(vocab['[UNK]'] ?? 1);
    }
    return ids;
}

function encodeSpecial(tok) {
    return specialToks[tok] ?? vocab[tok] ?? 0;
}

function decode(ids) {
    const tokens = ids.map(id => invVocab[id] || '').join('');
    const bytes  = Array.from(tokens).map(ch => BYTE_INV[ch] ?? ch.charCodeAt(0));
    try {
        return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
    } catch {
        return tokens;
    }
}

function initTokenizer(data) {
    tokData   = data;
    vocab     = data.model.vocab;
    invVocab  = {};
    for (const [k, v] of Object.entries(vocab)) invVocab[v] = k;
    mergeRank = {};
    (data.model.merges || []).forEach((m, i) => { mergeRank[m] = i; });
    specialToks = {};
    for (const st of (data.added_tokens || [])) {
        specialToks[st.content] = st.id;
        invVocab[st.id] = st.content;
    }
    TOK_BOS = encodeSpecial('[BOS]');
    TOK_EOS = encodeSpecial('[EOS]');
    TOK_SEP = encodeSpecial('[SEP]');
    TOK_SYS = encodeSpecial('[SYS]');
    TOK_PAD = encodeSpecial('[PAD]');
    console.log('[Worker] Special tokens:', { TOK_BOS, TOK_EOS, TOK_SEP, TOK_SYS, TOK_PAD });
}

// ── Sampling ──────────────────────────────────────────────────────────────────
function sample(logits, temp, topK, topP) {
    // Temperature
    const scaled = logits.map(v => v / temp);

    // Top-K filter
    let filtered = [...scaled];
    if (topK > 0) {
        const sorted = [...scaled].sort((a, b) => b - a);
        const thresh = sorted[Math.min(topK - 1, sorted.length - 1)];
        filtered = scaled.map(v => v >= thresh ? v : -Infinity);
    }

    // Softmax
    const max  = filtered.reduce((a, b) => Math.max(a, b), -Infinity);
    const exps = filtered.map(v => Math.exp(v - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(v => v / sum);

    // Top-P nucleus sampling
    const indexed = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    let cum = 0;
    const nucleus = [];
    for (const [p, i] of indexed) {
        nucleus.push([p, i]);
        cum += p;
        if (cum >= topP) break;
    }
    const nucSum = nucleus.reduce((s, [p]) => s + p, 0);
    const normed = nucleus.map(([p, i]) => [p / nucSum, i]);
    let r = Math.random(), acc = 0;
    for (const [p, i] of normed) {
        acc += p;
        if (r <= acc) return i;
    }
    return normed[normed.length - 1][1];
}

// ── Prompt builder ────────────────────────────────────────────────────────────
function buildPrompt(history, userText) {
    const sysIds      = [TOK_BOS, TOK_SYS, ...encode(SYSTEM_PROMPT, false), TOK_SEP];
    const axiaPrefix  = encode('AxiaLM:', false);
    const maxPrompt   = MAX_CTX - GEN_MAX_NEW;

    const buildIds = (hist, cur) => {
        let ids = [...sysIds];
        for (const { role, content } of hist) {
            const speaker = role === 'user' ? 'Human' : 'AxiaLM';
            const plain   = content.replace(/<[^>]+>/g, '').trim();
            ids.push(...encode(`${speaker}: ${plain}`, false), TOK_SEP);
        }
        ids.push(...encode(`Human: ${cur}`, false), TOK_SEP, ...axiaPrefix);
        return ids;
    };

    let hist = [...history];
    let ids  = buildIds(hist, userText);
    while (ids.length > maxPrompt && hist.length > 0) {
        hist.shift();
        ids = buildIds(hist, userText);
    }
    return ids;
}

// ── Inference ─────────────────────────────────────────────────────────────────
async function runInference(userText, history) {
    let inputIds = buildPrompt(history, userText);
    const newIds = [];
    const STOP   = new Set([TOK_SEP, TOK_EOS]);

    for (let i = 0; i < GEN_MAX_NEW; i++) {
        const ctx    = inputIds.slice(-MAX_CTX);
        const tensor = new ort.Tensor('int64',
            BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]);

        const out     = await ortSession.run({ input_ids: tensor });
        const logits  = Array.from(out.logits.data);
        const nextId  = sample(logits, GEN_TEMP, GEN_TOP_K, GEN_TOP_P);

        if (STOP.has(nextId)) break;
        newIds.push(nextId);
        inputIds.push(nextId);

        // Stream token back to main thread
        const token = decode([nextId]);
        self.postMessage({ type: 'token', token });
    }

    let text = decode(newIds);
    for (const stop of ['Human:', '[SEP]', '[SYS]', '[BOS]', '[EOS]']) {
        const idx = text.indexOf(stop);
        if (idx !== -1) text = text.slice(0, idx);
    }
    text = text.trim();
    if (!text) text = '<p>...</p>';
    else if (!text.startsWith('<')) text = `<p>${text}</p>`;
    return text;
}

// ── Load model ────────────────────────────────────────────────────────────────
async function loadModel() {
    self.postMessage({ type: 'progress', pct: 5, msg: 'Fetching config...' });
    const cfgResp = await fetch(CONFIG_URL);
    if (!cfgResp.ok) throw new Error(`Config fetch failed: ${cfgResp.status}`);
    modelConfig = await cfgResp.json();

    self.postMessage({ type: 'progress', pct: 10, msg: 'Fetching tokenizer...' });
    const tokResp = await fetch(TOK_URL);
    if (!tokResp.ok) throw new Error(`Tokenizer fetch failed: ${tokResp.status}`);
    initTokenizer(await tokResp.json());

    self.postMessage({ type: 'progress', pct: 15, msg: 'Downloading model...' });
    const modelResp = await fetch(MODEL_URL);
    if (!modelResp.ok) throw new Error(`Model fetch failed: ${modelResp.status}`);

    const total  = parseInt(modelResp.headers.get('content-length') || '0');
    const reader = modelResp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (total > 0) {
            const pct = 15 + (received / total) * 70;
            self.postMessage({ type: 'progress', pct,
                msg: `Downloading... ${(received/1e6).toFixed(1)} / ${(total/1e6).toFixed(1)} MB` });
        }
    }
    const buf = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) { buf.set(c, pos); pos += c.length; }

    self.postMessage({ type: 'progress', pct: 87, msg: 'Starting ONNX Runtime...' });

    ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 2) - 1);
    ort.env.wasm.simd = true;

    // Try WebGPU, fall back to WASM
    let providers = ['wasm'];
    try {
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) providers = ['webgpu', 'wasm'];
        }
    } catch {}

    self.postMessage({ type: 'progress', pct: 92, msg: `Creating session (${providers[0]})...` });
    ortSession = await ort.InferenceSession.create(buf.buffer, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
    });

    self.postMessage({ type: 'progress', pct: 100, msg: 'Ready!' });
    self.postMessage({ type: 'ready' });
}

// ── Message handler ───────────────────────────────────────────────────────────
self.onmessage = async (e) => {
    const { type, userText, history } = e.data;
    if (type === 'load') {
        try {
            await loadModel();
        } catch (err) {
            self.postMessage({ type: 'error', msg: err.message });
        }
    } else if (type === 'infer') {
        try {
            const response = await runInference(userText, history);
            self.postMessage({ type: 'done', response });
        } catch (err) {
            self.postMessage({ type: 'error', msg: err.message });
        }
    }
};
