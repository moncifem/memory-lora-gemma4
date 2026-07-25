import { NextRequest, NextResponse } from "next/server";
import { activeEndpoint } from "@/lib/jobs";
import {
  anthropicToOpenAI,
  openAIToAnthropic,
  openAIStreamToAnthropic,
} from "@/lib/anthropic";

export const dynamic = "force-dynamic";

const SERVED_MODEL = "memory-lora";
const API_KEY = process.env.MLORA_API_KEY || "";

/** True when the caller explicitly asked for the un-adapted base model. */
function isBaseModel(model: unknown): boolean {
  return typeof model === "string" && model.trim().toLowerCase() === "base";
}

function authOk(req: NextRequest): boolean {
  if (!API_KEY) return true;
  const auth = req.headers.get("authorization");
  if (auth === `Bearer ${API_KEY}`) return true;
  if (req.headers.get("x-api-key") === API_KEY) return true;
  return false;
}

function resolveTarget(req: NextRequest, modelField?: string): string | null {
  // Job selection precedence: ?job= , x-mlora-job header, model "memory-lora:<job>",
  // else most-recently-ready job.
  const url = new URL(req.url);
  let jobId =
    url.searchParams.get("job") || req.headers.get("x-mlora-job") || undefined;
  if (!jobId && modelField && modelField.includes(":")) {
    jobId = modelField.split(":")[1];
  }
  const ep = activeEndpoint(jobId || undefined);
  return ep ? `http://127.0.0.1:${ep.port}` : null;
}

async function proxyPassthrough(
  req: NextRequest,
  subpath: string,
  rawBody: string | null
): Promise<Response> {
  let modelField: string | undefined;
  if (rawBody) {
    try {
      modelField = JSON.parse(rawBody).model;
    } catch {
      /* ignore */
    }
  }
  const target = resolveTarget(req, modelField);
  if (!target) {
    return NextResponse.json(
      { error: { message: "no ready model — build a repo first", type: "not_ready" } },
      { status: 503 }
    );
  }

  // Normalize the model field so arbitrary client model strings work — but
  // NEVER collapse "base" into the served name. The upstream server selects
  // frozen-vs-adapted from this field, so rewriting it unconditionally made the
  // side-by-side comparison silently return the adapted model on BOTH sides.
  let body = rawBody;
  if (rawBody) {
    try {
      const j = JSON.parse(rawBody);
      if (!isBaseModel(j.model)) j.model = SERVED_MODEL;
      body = JSON.stringify(j);
    } catch {
      /* leave as-is */
    }
  }

  const upstream = await fetch(`${target}/v1/${subpath}`, {
    method: req.method,
    headers: { "content-type": "application/json" },
    body: req.method === "GET" ? undefined : body,
  });

  // Stream passthrough (SSE or JSON).
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") || "application/json",
      "cache-control": "no-cache",
    },
  });
}

async function handleMessages(req: NextRequest, rawBody: string): Promise<Response> {
  let body: any;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ error: "invalid json" }, { status: 400 });
  }
  const target = resolveTarget(req, body.model);
  if (!target) {
    return NextResponse.json(
      { type: "error", error: { type: "not_ready", message: "no ready model — build a repo first" } },
      { status: 503 }
    );
  }

  const oaiReq = anthropicToOpenAI(
    body, isBaseModel(body.model) ? "base" : SERVED_MODEL);
  const upstream = await fetch(`${target}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(oaiReq),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return NextResponse.json(
      { type: "error", error: { type: "upstream_error", message: text } },
      { status: upstream.status }
    );
  }

  if (body.stream) {
    const stream = openAIStreamToAnthropic(upstream.body!, SERVED_MODEL);
    return new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
    });
  }

  const oai = await upstream.json();
  return NextResponse.json(openAIToAnthropic(oai, SERVED_MODEL));
}

// Rough token estimate (~4 chars/token) over all text in an Anthropic request.
// Clients use this for budgeting only, so an approximation is acceptable and
// avoids loading a tokenizer in the Next.js process.
function estimateTokens(rawBody: string): number {
  let body: any;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return 0;
  }
  const chunks: string[] = [];
  const collect = (c: any) => {
    if (typeof c === "string") chunks.push(c);
    else if (Array.isArray(c))
      for (const b of c) {
        if (typeof b?.text === "string") chunks.push(b.text);
        else if (typeof b?.content === "string") chunks.push(b.content);
      }
  };
  collect(body.system);
  for (const m of body.messages || []) collect(m.content);
  for (const t of body.tools || [])
    chunks.push(t.name || "", t.description || "", JSON.stringify(t.input_schema || {}));
  return Math.ceil(chunks.join(" ").length / 4);
}

async function handle(req: NextRequest, path: string[]): Promise<Response> {
  if (!authOk(req)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const subpath = path.join("/");
  const rawBody = req.method === "GET" ? null : await req.text();

  if (subpath === "messages") {
    return handleMessages(req, rawBody || "{}");
  }
  // Claude Code calls this before sending a request; answer it locally rather
  // than 404-ing (the upstream OpenAI server has no equivalent route).
  if (subpath === "messages/count_tokens") {
    return NextResponse.json({ input_tokens: estimateTokens(rawBody || "{}") });
  }
  return proxyPassthrough(req, subpath, rawBody);
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return handle(req, path);
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return handle(req, path);
}
