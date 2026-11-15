// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. Each request
// is answered by one provider, on that provider's own wire, and the answer is translated back
// into the wire the client used:
//
//   client openai    -> upstream openai:    body passed through (model replaced)
//   client openai    -> upstream anthropic: rewritten to Messages, answer rewritten back
//   client openai    -> upstream gemini:    rewritten to generateContent, answer rewritten back
//   client anthropic -> upstream openai:    rewritten to Chat Completions, answer rewritten back
//   client anthropic -> upstream anthropic: body passed through (model replaced)
//   client anthropic -> upstream gemini:    rewritten to generateContent, answer rewritten back
//
// What the client checks, and so what every translation must preserve exactly:
//   - the provider's answer text whole (it is an echo of what the client sent: system prompt,
//     user text, tool names, max_tokens, stop sequences, image and JSON mode);
//   - the provider's usage counts, as that wire reports them;
//   - the stop reason, mapped between the wires' names for the same family.
// The simulated providers answer plain text only, so tool calls in an answer never occur here,
// but the translators handle them anyway: a real provider's tool_calls become tool_use blocks
// and input_json_delta events, and back.
//
// Everything here is a pure function of its arguments (or a generator over a stream): no clock
// and no randomness, so the same body and the same provider answer give the same bytes on a
// replay. The model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Any wire's stop reason as an OpenAI finish_reason (same family, OpenAI's name). */
export function stopReasonToOpenAI(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Any wire's stop reason as an Anthropic stop_reason (same family, Anthropic's name). */
export function stopReasonToAnthropic(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the 13-character pieces the simulated providers stream in. */
export function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One tool call as the wires carry it: id, name and the arguments as a JSON string. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** One provider answer, in no wire's shape: text, usage, stop reason and tool calls. */
export interface UpstreamAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
  toolCalls: ToolCall[];
}

/** A Chat Completions answer as an UpstreamAnswer. */
export function parseOpenAIJson(body: Json): UpstreamAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

/** An Anthropic Messages answer as an UpstreamAnswer. */
export function parseAnthropicJson(body: Json): UpstreamAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls,
  };
}

/** A Gemini generateContent answer as an UpstreamAnswer. */
export function parseGeminiJson(body: Json): UpstreamAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) },
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    toolCalls: [],
  };
}

/** An UpstreamAnswer as a Chat Completions object naming `model`. */
export function openAIJsonFromAnswer(answer: UpstreamAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: stopReasonToOpenAI(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** An UpstreamAnswer as an Anthropic Messages object naming `model`. */
export function anthropicJsonFromAnswer(answer: UpstreamAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  return anthropicJsonFromAnswer(parseOpenAIJson(body), model);
}

/* --- Gemini --- */

/** An OpenAI image_url part as a Gemini part: inline data for a data URL, a file reference otherwise. */
function geminiImagePart(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini part. */
function geminiImageBlock(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** The tool definitions of a request as Gemini function declarations. */
function geminiFunctionDeclarations(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    // OpenAI shape: {type:'function', function:{name, description, parameters}}
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      out.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
      continue;
    }
    // Anthropic shape: {name, description, input_schema}
    if (typeof raw.name === 'string') {
      out.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

/** A Gemini request body from the shared pieces every client wire carries. */
function geminiBody(
  systemText: string,
  contents: Json[],
  declarations: Json[],
  maxTokens: number | null,
  temperature: unknown,
  topP: unknown,
  stops: string[],
  jsonMode: boolean,
): Json {
  const body: Json = { contents };
  if (systemText !== '') body.systemInstruction = { parts: [{ text: systemText }] };
  if (declarations.length > 0) body.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let systemText = '';
  const contents: Json[] = [];
  const callNames = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          const id = typeof call.id === 'string' ? call.id : '';
          if (id !== '') callNames.set(id, call.function.name);
          parts.push({ functionCall: { name: call.function.name, args: argumentsOf(call.function.arguments) } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: callNames.get(id) ?? '', response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const part = geminiImagePart(block.image_url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  return geminiBody(systemText, contents, geminiFunctionDeclarations(body.tools), maxTokens, body.temperature, body.top_p, stops, format === 'json_object' || format === 'json_schema');
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, jsonRequired: boolean): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    const responses: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const part = geminiImageBlock(block.source);
          if (part !== null) parts.push(part);
        } else if (block.type === 'tool_result') {
          responses.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { result: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (responses.length > 0) contents.push({ role: 'function', parts: responses });
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : null;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return geminiBody(systemText, contents, geminiFunctionDeclarations(body.tools), maxTokens, body.temperature, body.top_p, stops, jsonRequired);
}

/* --- SSE plumbing --- */

/** One SSE frame read from a stream: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Read every SSE frame of a stream, in order. */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield frameOf(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameOf(buffer);
  } finally {
    reader.releaseLock();
  }
}

function frameOf(text: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of text.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** A Chat Completions SSE stream as the Anthropic event sequence. */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of readSseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const out: string[] = [...startEvents()];
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
    // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
    // output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    yield* out;
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/** An Anthropic event stream as a Chat Completions event stream. */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  const toolCalls: { index: number; id: string; name: string; args: string }[] = [];
  const toolByIndex = new Map<number, number>();
  let sawRole = false;

  const roleChunk = (): string[] => {
    if (sawRole) return [];
    sawRole = true;
    return [sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })];
  };

  for await (const frame of readSseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      yield* roleChunk();
    } else if (frame.event === 'content_block_start') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const block = isRecord(data.content_block) ? data.content_block : {};
      yield* roleChunk();
      if (block.type === 'tool_use') {
        toolByIndex.set(index, toolCalls.length);
        toolCalls.push({ index: toolCalls.length, id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', args: '' });
        yield sseEvent(null, {
          id: 'chatcmpl-aggregate',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [{ index: 0, delta: { tool_calls: [{ index: toolCalls.length - 1, id: toolCalls[toolCalls.length - 1].id, type: 'function', function: { name: toolCalls[toolCalls.length - 1].name, arguments: '' } }] }, finish_reason: null }],
        });
      }
    } else if (frame.event === 'content_block_delta') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const delta = isRecord(data.delta) ? data.delta : {};
      yield* roleChunk();
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        text += delta.text;
        for (const piece of splitText(delta.text)) {
          yield sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
        }
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const at = toolByIndex.get(index);
        if (at !== undefined) {
          toolCalls[at].args += delta.partial_json;
          yield sseEvent(null, {
            id: 'chatcmpl-aggregate',
            object: 'chat.completion.chunk',
            created: FIXED_OPENAI_CREATED,
            model,
            choices: [{ index: 0, delta: { tool_calls: [{ index: at, function: { arguments: delta.partial_json } }] }, finish_reason: null }],
          });
        }
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') stopReason = delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      break;
    }
  }
  yield* roleChunk();
  yield sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: stopReasonToOpenAI(stopReason) }] });
  yield sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  yield 'data: [DONE]\n\n';
  void text;
}

/** The text and usage a Gemini streamGenerateContent stream carries, in order. */
export async function* geminiStreamParts(upstream: ReadableStream<Uint8Array>): AsyncGenerator<{ text: string; usage: { input: number; output: number } | null; finish: string | null }> {
  for await (const frame of readSseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    let text = '';
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
    const usage = isRecord(data.usageMetadata)
      ? { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) }
      : null;
    const finish = typeof candidate.finishReason === 'string' ? candidate.finishReason : null;
    yield { text, usage, finish };
  }
}

/** A Gemini stream as a Chat Completions event stream. */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  yield sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for await (const part of geminiStreamParts(upstream)) {
    if (part.usage !== null) {
      inputTokens = part.usage.input;
      outputTokens = part.usage.output;
    }
    if (part.finish !== null) stop = part.finish;
    for (const piece of splitText(part.text)) {
      yield sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
    }
  }
  yield sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: stopReasonToOpenAI(stop) }] });
  yield sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  yield 'data: [DONE]\n\n';
}

/** A Gemini stream as an Anthropic event stream. */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  let started = false;
  let textOpen = false;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  for await (const part of geminiStreamParts(upstream)) {
    if (part.usage !== null) {
      inputTokens = part.usage.input;
      outputTokens = part.usage.output;
    }
    if (part.finish !== null) stop = part.finish;
    if (part.text !== '') {
      const out: string[] = [...start()];
      if (!textOpen) {
        textOpen = true;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
      }
      for (const piece of splitText(part.text)) {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
      }
      yield* out;
    }
  }
  const out: string[] = [...start()];
  if (textOpen) {
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
    textOpen = false;
  }
  out.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonToAnthropic(stop), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  yield* out;
  void inputTokens;
}

/* --- OpenAI request -> Anthropic request --- */

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`:
 * system messages become the system prompt, image_url parts become image blocks, assistant
 * tool_calls become tool_use blocks, tool messages become tool_result blocks, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through. (The reverse direction is anthropicToOpenAI, above.)
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  const callNames = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          const id = typeof call.id === 'string' ? call.id : '';
          if (id !== '') callNames.set(id, call.function.name);
          blocks.push({ type: 'tool_use', id, name: call.function.name, input: argumentsOf(call.function.arguments) });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: id, content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) {
            blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          } else {
            blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required' || choice === 'none') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}
