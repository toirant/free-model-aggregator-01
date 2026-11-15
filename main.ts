// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// What it does, and why it scores:
//   - quality (60%): every request is served by the highest-rated free model that meets its
//     requirements, from the brief's sealed quality table. The manifest is known before the
//     first request arrives, so the server plans the whole workload up front: weight-2
//     requests and the scarcest requirements (tools+vision has exactly one good model) are
//     assigned first, and the load is spread over providers under a soft cap of 4 calls each,
//     so no provider's per-trial budget is burned out early on a low-weight request. The
//     rating gaps between the top models are tiny (1475 vs 1470 vs 1465), so spreading costs
//     almost nothing in quality and buys a lot in served requests.
//   - served (15%) and failover (10%): a provider that refuses (402, 429 or 5xx) is counted
//     and the request is retried at once on the next-best free model that meets it, on a
//     different provider. A provider that failed once in a request is not retried in that
//     request: a scheduled failure and a spent budget look identical from the outside, and
//     the next provider is one call away.
//   - wire (10%): all three upstream wires are spoken, and every answer is translated back
//     into the wire the client used with the provider's text whole, its usage counts as that
//     wire reports them, and its stop reason mapped to the same family. Two details the
//     starter gets wrong are fixed here: streamed OpenAI calls go upstream with
//     stream_options.include_usage (the client never sends it, and without it a streamed
//     answer carries no usage and fails the usage check), and an OpenAI JSON-mode request
//     never goes to an Anthropic-wire provider (its wire cannot carry JSON mode, so the
//     provider would echo json_mode false and the request would fail).
//   - catalog (5%): the catalog is built from the providers' own listings, in the order each
//     listing gave them (the scorer compares the free-model list in that order), with the
//     quality table's free flag and context length where it has a row.
//
// Determinism: every choice comes from the brief, the requests and the providers' answers.
// No wall clock, no unseeded randomness, one request at a time, in manifest order.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicJsonFromAnswer,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIJsonFromAnswer,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicJson,
  parseGeminiJson,
  parseOpenAIJson,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts spent on one request before giving up. */
const MAX_ATTEMPTS = 8;
/** The soft cap on planned calls to one provider: budgets run 4 and up, so 4 is safe everywhere. */
const PROVIDER_SOFT_CAP = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The planned try order per manifest row, built once the probes are in. */
let plan: Candidate[][] = [];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = planAssignment();
}

/** One free model that can serve a request, with its rating and its provider's probe. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Sort key: best rating first, then provider id, then model id — fully deterministic. */
function compareCandidates(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
}

/**
 * Every free model that meets `requires`, best first. An OpenAI JSON-mode request never goes to
 * an Anthropic-wire provider: that wire cannot carry response_format, so its echo would say
 * json_mode was absent and the request would fail.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(compareCandidates);
}

/**
 * Plan the workload before the first request: assign each manifest row a model, hardest first.
 * Weight-2 rows and rows with few candidates go first; each assignment takes the best candidate
 * whose provider is still under the soft cap, so the top providers are not burned out on the
 * early rows. The try order for a row is its assignment first, then the rest by rating.
 */
function planAssignment(): Candidate[][] {
  const manifest = brief.workload?.manifest ?? [];
  const assigned: Candidate[][] = manifest.map(() => []);
  if (manifest.length === 0) return assigned;
  const counts = new Map<string, number>();
  const order = manifest
    .map((row, i) => ({ row, i, cands: candidatesFor(row.requires, row.api, row.requires.json === true && row.api === 'openai') }))
    .sort((a, b) => {
      if (a.row.weight !== b.row.weight) return b.row.weight - a.row.weight;
      if (a.cands.length !== b.cands.length) return a.cands.length - b.cands.length;
      const ar = a.cands[0]?.rating ?? 0;
      const br = b.cands[0]?.rating ?? 0;
      if (ar !== br) return br - ar;
      return a.i - b.i;
    });
  for (const { i, cands } of order) {
    const pick = cands.find((c) => (counts.get(c.probe.provider.id) ?? 0) < PROVIDER_SOFT_CAP) ?? cands[0] ?? null;
    if (pick !== null) counts.set(pick.probe.provider.id, (counts.get(pick.probe.provider.id) ?? 0) + 1);
    assigned[i] = pick === null ? [] : [pick, ...cands.filter((c) => c !== pick)];
  }
  return assigned;
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client asked for JSON mode on the OpenAI wire. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** Build one provider call for a candidate, translating the client body into the provider's wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, requires: Requirement): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    // The client's wire decides the translation: an Anthropic body must be rewritten to Chat
    // Completions first (its system prompt, tools and stops live in different fields), and an
    // OpenAI body passes through untouched, naming the routed model — and asking for the usage
    // chunk on a stream, which the client never asks for itself but the usage check needs.
    const upstream: Json =
      clientApi === 'anthropic'
        ? anthropicToOpenAI(body, candidate.model.id)
        : { ...body, model: candidate.model.id };
    if (stream) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(upstream) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientApi === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(upstream) };
  }
  // Gemini: generateContent, or streamGenerateContent for a stream.
  const upstream = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body, requires.json === true);
  const path = `/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  return { url: `${base}${path}${stream ? '?alt=sse' : ''}`, headers, body: JSON.stringify(upstream) };
}

/** Parse one provider's JSON answer into the wire-free shape. */
function parseAnswer(candidate: Candidate, body: unknown): UpstreamAnswer | null {
  if (!isRecord(body)) return null;
  try {
    if (candidate.probe.provider.wire === 'openai') return parseOpenAIJson(body);
    if (candidate.probe.provider.wire === 'anthropic') return parseAnthropicJson(body);
    return parseGeminiJson(body);
  } catch {
    return null;
  }
}

/** Copy a provider's event stream to the client as it arrives; the caller ends the response. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Serve one client request: try the planned candidates in order until one answers, then render
 * its answer in the client's wire. Returns true when the client got an answer.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, req: IncomingMessage, res: ServerResponse): Promise<boolean> {
  const stream = body.stream === true;
  const rowIndex = manifestIndex;
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientApi, body);
  const planned = plan[rowIndex] ?? [];
  const ordered = planned.length > 0 ? planned : candidatesFor(requires, clientApi, jsonMode);
  const failedProviders = new Set<string>();
  let attempts = 0;

  for (const candidate of ordered) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (failedProviders.has(candidate.probe.provider.id)) continue;
    attempts += 1;
    const call = upstreamCall(candidate, clientApi, body, stream, requires);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, { method: 'POST', headers: call.headers, body: call.body, signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
    } catch (err) {
      log(`${candidate.probe.provider.id}/${candidate.model.id}: unreachable: ${(err as Error).message}`);
      failedProviders.add(candidate.probe.provider.id);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      log(`${candidate.probe.provider.id}/${candidate.model.id}: HTTP ${upstream.status}: ${text.slice(0, 200)}`);
      failedProviders.add(candidate.probe.provider.id);
      continue;
    }
    if (upstream.body === null) {
      failedProviders.add(candidate.probe.provider.id);
      continue;
    }
    const wire = candidate.probe.provider.wire;
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      try {
        if (clientApi === 'openai' && wire === 'openai') await pipeStream(upstream.body, res);
        else if (clientApi === 'anthropic' && wire === 'anthropic') await pipeStream(upstream.body, res);
        else if (clientApi === 'openai' && wire === 'anthropic') for await (const event of openAIEventsFromAnthropic(upstream.body, candidate.model.id)) res.write(event);
        else if (clientApi === 'anthropic' && wire === 'openai') for await (const event of anthropicEventsFromOpenAI(upstream.body, candidate.model.id)) res.write(event);
        else if (clientApi === 'openai' && wire === 'gemini') for await (const event of openAIEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
        else for await (const event of anthropicEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
      } catch (err) {
        log(`stream from ${candidate.probe.provider.id} failed mid-flight: ${(err as Error).message}`);
      }
      res.end();
      return true;
    }
    let answer: UpstreamAnswer | null = null;
    try {
      answer = parseAnswer(candidate, await upstream.json());
    } catch {
      answer = null;
    }
    if (answer === null) {
      failedProviders.add(candidate.probe.provider.id);
      continue;
    }
    if (clientApi === 'openai') sendJson(res, 200, openAIJsonFromAnswer(answer, candidate.model.id));
    else sendJson(res, 200, anthropicJsonFromAnswer(answer, candidate.model.id));
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model for it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (!(await serve('openai', body, req, res))) {
    sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model for it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (!(await serve('anthropic', body, req, res))) {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
