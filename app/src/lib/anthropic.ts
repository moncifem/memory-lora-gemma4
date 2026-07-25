// Translation between the Anthropic Messages API (/v1/messages, what Claude
// Code speaks) and the OpenAI Chat Completions API (what vLLM serves).
// Supports system prompts, multi-turn, tool definitions, tool calls and tool
// results, in both non-streaming and streaming (SSE) modes.

/* eslint-disable @typescript-eslint/no-explicit-any */

const STOP_MAP: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
};

function textFromContent(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b) => b && b.type === "text")
      .map((b) => b.text)
      .join("");
  }
  return "";
}

// Anthropic request -> OpenAI Chat Completions request.
export function anthropicToOpenAI(body: any, servedModel: string): any {
  const messages: any[] = [];

  if (body.system) {
    messages.push({ role: "system", content: textFromContent(body.system) });
  }

  for (const m of body.messages || []) {
    const content = m.content;
    if (typeof content === "string") {
      messages.push({ role: m.role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;

    if (m.role === "assistant") {
      const text = textFromContent(content);
      const toolUses = content.filter((b: any) => b.type === "tool_use");
      const msg: any = { role: "assistant", content: text || null };
      if (toolUses.length) {
        msg.tool_calls = toolUses.map((t: any) => ({
          id: t.id,
          type: "function",
          function: { name: t.name, arguments: JSON.stringify(t.input ?? {}) },
        }));
      }
      messages.push(msg);
    } else {
      // user role: may carry tool_result blocks (-> OpenAI tool messages) and
      // plain text / image blocks (-> a user message).
      const toolResults = content.filter((b: any) => b.type === "tool_result");
      for (const tr of toolResults) {
        messages.push({
          role: "tool",
          tool_call_id: tr.tool_use_id,
          content:
            typeof tr.content === "string"
              ? tr.content
              : textFromContent(tr.content),
        });
      }
      const text = textFromContent(content);
      if (text) messages.push({ role: "user", content: text });
    }
  }

  const out: any = {
    model: servedModel,
    messages,
    max_tokens: body.max_tokens ?? 1024,
    stream: !!body.stream,
  };
  if (body.temperature != null) out.temperature = body.temperature;
  if (body.top_p != null) out.top_p = body.top_p;
  if (body.stop_sequences) out.stop = body.stop_sequences;

  if (Array.isArray(body.tools) && body.tools.length) {
    out.tools = body.tools.map((t: any) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description || "",
        parameters: t.input_schema || { type: "object", properties: {} },
      },
    }));
  }
  if (body.tool_choice) {
    const tc = body.tool_choice;
    if (tc.type === "auto") out.tool_choice = "auto";
    else if (tc.type === "any") out.tool_choice = "required";
    else if (tc.type === "tool" && tc.name)
      out.tool_choice = { type: "function", function: { name: tc.name } };
  }
  return out;
}

// OpenAI (non-streaming) response -> Anthropic Messages response.
export function openAIToAnthropic(oai: any, model: string): any {
  const choice = oai.choices?.[0] || {};
  const msg = choice.message || {};
  const content: any[] = [];
  if (msg.content) content.push({ type: "text", text: msg.content });
  for (const tc of msg.tool_calls || []) {
    let input: any = {};
    try {
      input = JSON.parse(tc.function?.arguments || "{}");
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: tc.id, name: tc.function?.name, input });
  }
  if (!content.length) content.push({ type: "text", text: "" });

  return {
    id: oai.id || `msg_${Math.random().toString(16).slice(2)}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: STOP_MAP[choice.finish_reason] || "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: oai.usage?.prompt_tokens ?? 0,
      output_tokens: oai.usage?.completion_tokens ?? 0,
    },
  };
}

function sse(event: string, data: any): Uint8Array {
  return new TextEncoder().encode(
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  );
}

// Transform an OpenAI SSE stream (ReadableStream of bytes) into an Anthropic
// Messages SSE stream.
export function openAIStreamToAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string
): ReadableStream<Uint8Array> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const msgId = `msg_${Math.random().toString(16).slice(2)}`;

  let buffer = "";
  let started = false;
  let textOpen = false;
  let nextIndex = 0;
  // OpenAI tool_call index -> { anthropicIndex }
  const toolBlocks = new Map<number, number>();
  let finish = "end_turn";

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const openMessage = () => {
        if (started) return;
        started = true;
        controller.enqueue(
          sse("message_start", {
            type: "message_start",
            message: {
              id: msgId,
              type: "message",
              role: "assistant",
              model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          })
        );
      };
      const ensureText = () => {
        openMessage();
        if (!textOpen) {
          textOpen = true;
          controller.enqueue(
            sse("content_block_start", {
              type: "content_block_start",
              index: nextIndex,
              content_block: { type: "text", text: "" },
            })
          );
        }
      };
      const closeText = () => {
        if (textOpen) {
          controller.enqueue(
            sse("content_block_stop", { type: "content_block_stop", index: nextIndex })
          );
          textOpen = false;
          nextIndex++;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // finalize
          closeText();
          for (const idx of toolBlocks.values()) {
            controller.enqueue(
              sse("content_block_stop", { type: "content_block_stop", index: idx })
            );
          }
          if (started) {
            controller.enqueue(
              sse("message_delta", {
                type: "message_delta",
                delta: { stop_reason: finish, stop_sequence: null },
                usage: { output_tokens: 0 },
              })
            );
            controller.enqueue(sse("message_stop", { type: "message_stop" }));
          }
          controller.close();
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const t = line.trim();
          if (!t.startsWith("data:")) continue;
          const payload = t.slice(5).trim();
          if (payload === "[DONE]") continue;
          let chunk: any;
          try {
            chunk = JSON.parse(payload);
          } catch {
            continue;
          }
          const choice = chunk.choices?.[0];
          if (!choice) continue;
          const delta = choice.delta || {};

          if (delta.content) {
            ensureText();
            controller.enqueue(
              sse("content_block_delta", {
                type: "content_block_delta",
                index: nextIndex,
                delta: { type: "text_delta", text: delta.content },
              })
            );
          }

          for (const tc of delta.tool_calls || []) {
            const oaiIdx = tc.index ?? 0;
            if (!toolBlocks.has(oaiIdx)) {
              closeText();
              const aidx = nextIndex++;
              toolBlocks.set(oaiIdx, aidx);
              openMessage();
              controller.enqueue(
                sse("content_block_start", {
                  type: "content_block_start",
                  index: aidx,
                  content_block: {
                    type: "tool_use",
                    id: tc.id || `toolu_${Math.random().toString(16).slice(2)}`,
                    name: tc.function?.name || "",
                    input: {},
                  },
                })
              );
            }
            const aidx = toolBlocks.get(oaiIdx)!;
            if (tc.function?.arguments) {
              controller.enqueue(
                sse("content_block_delta", {
                  type: "content_block_delta",
                  index: aidx,
                  delta: { type: "input_json_delta", partial_json: tc.function.arguments },
                })
              );
            }
          }

          if (choice.finish_reason) {
            finish = STOP_MAP[choice.finish_reason] || "end_turn";
          }
        }
      }
    },
    cancel() {
      reader.cancel();
    },
  });
}
