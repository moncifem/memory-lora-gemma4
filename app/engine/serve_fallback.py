#!/usr/bin/env python3
"""OpenAI-compatible inference server backed by transformers, for machines
where vLLM has no usable backend (notably Apple Silicon / MPS).

Exposes the same surface the app's proxy expects from vLLM:
    GET  /v1/models
    POST /v1/completions           (streaming + non-streaming)
    POST /v1/chat/completions      (streaming + non-streaming)
    GET  /health

Serves either a self-contained MERGED model directory, or the base model with
a generated LoRA adapter loaded on top (``--adapter``). Uses stdlib http.server
only (no FastAPI/uvicorn dependency) so it runs from the training repo's venv.

Usage:
    python serve_fallback.py --model /path/to/merged --port 8000
    python serve_fallback.py --base google/gemma-4-E2B --adapter /path/to/adapter --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import config  # noqa: F401  (sets sys.path to the training repo root)

SERVED_NAME = "memory-lora"
_LOCK = threading.Lock()  # transformers generate is not reentrant-safe here


class Engine:
    def __init__(self, model_dir: str | None, base: str | None,
                 adapter: str | None, device: str):
        from transformers import (AutoModelForCausalLM,
                                  AutoModelForImageTextToText, AutoTokenizer)

        self.device = config.resolve_device(device)
        src = model_dir or base
        print(f"[serve] loading {src} on {self.device} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        # Gemma-4 is a conditional-generation (image-text-to-text) architecture,
        # but nothing else here is multimodal-specific — fall back to the plain
        # causal-LM class so a text-only base model also serves correctly.
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                src, torch_dtype=dtype, low_cpu_mem_usage=True,
            )
        except (ValueError, KeyError, OSError):
            self.model = AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=dtype, low_cpu_mem_usage=True,
            )
        self.has_adapter = bool(adapter)
        if adapter:
            from peft import PeftModel
            print(f"[serve] attaching adapter {adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(self.device)
        self.model.eval()
        print("[serve] ready.", flush=True)

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Render chat messages into a prompt string.

        ``google/gemma-4-E2B`` is a *base* (pretrained, non-instruction-tuned)
        checkpoint and genuinely ships no chat template — and it is
        deliberately the model this project targets, since the hypernetwork was
        trained to emit adapters for it. So we fall back to a plain role-tagged
        transcript, which is what base models handle best.

        The chat-template path is still tried first so that pointing this
        server at an instruction-tuned variant (``…-it``) just works.
        """
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 -- expected on base checkpoints
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):  # tolerate block-style content
                    content = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                label = {"system": "System", "user": "User",
                         "assistant": "Assistant", "tool": "Tool"}.get(role, role)
                parts.append(f"{label}: {content}")
            return "\n\n".join(parts) + "\n\nAssistant:"

    @staticmethod
    def _strip_thought(text: str) -> str:
        """Drop a Gemma-4 ``<|channel>thought … <channel|>`` block if the model
        emits one, so clients receive only the final answer."""
        end = text.rfind("<channel|>")
        if end != -1 and "<|channel>" in text[:end]:
            return text[end + len("<channel|>"):].lstrip()
        return text

    def _stops(self, extra: list[str] | None) -> list[str]:
        """Stop sequences. A base model has no turn structure and will happily
        keep going past its answer and hallucinate the *next* turn of the
        transcript we handed it, so the role labels used by
        ``_apply_chat_template`` are always treated as stops — otherwise every
        response trails a fabricated conversation."""
        stops = ["\nUser:", "\nSystem:", "\nAssistant:", "\nTool:"]
        for s in extra or []:
            if s:
                stops.append(s)
        return stops

    @staticmethod
    def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
        cut = min((i for i in (text.find(s) for s in stops) if i != -1),
                  default=-1)
        return (text[:cut], True) if cut != -1 else (text, False)

    def _maybe_off(self, use_adapter: bool):
        """Context manager that disables the adapter for base-model requests.
        Toggling one resident model is what makes the side-by-side demo fit in
        memory -- serving base and adapted separately would need 2x10GB."""
        import contextlib
        if use_adapter or not self.has_adapter:
            return contextlib.nullcontext()
        return self.model.disable_adapter()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, stop: list[str] | None = None,
                 use_adapter: bool = True):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature and temperature > 0
        with self._maybe_off(use_adapter):
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self._strip_thought(self.tok.decode(gen, skip_special_tokens=True))
        text, _ = self._truncate_at_stop(text, self._stops(stop))
        return text.strip(), int(enc["input_ids"].shape[1]), int(gen.shape[0])

    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int, temperature: float,
               top_p: float, stop: list[str] | None = None,
               use_adapter: bool = True):
        """Stream tokens, with the adapter toggled for the WHOLE operation.

        The enable/disable must wrap the entire generator -- both the worker
        thread and the consumption of the streamer -- and the thread must be
        joined before the context exits. PEFT's disable_adapter() flips state on
        the shared model, so if the context closed while the next request was
        already starting, that request would silently run with the wrong
        adapter state. Putting the context inside the worker thread did exactly
        that: base and adapted both came back as base.
        """
        from transformers import TextIteratorStreamer
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True)
        do_sample = temperature and temperature > 0
        kwargs = dict(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            streamer=streamer,
        )
        stops = self._stops(stop)
        hold = max(len(s) for s in stops)

        with self._maybe_off(use_adapter):
            worker = threading.Thread(
                target=self.model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            try:
                # A stop sequence can straddle two streamed pieces, so emit only
                # the part of the buffer that can no longer become part of one.
                buf, emitted, hit = "", 0, False
                for piece in streamer:
                    buf += piece
                    cut, hit = self._truncate_at_stop(buf, stops)
                    if hit:
                        if len(cut) > emitted:
                            yield cut[emitted:]
                        break
                    safe = max(0, len(buf) - hold)
                    if safe > emitted:
                        yield buf[emitted:safe]
                        emitted = safe
                if not hit and len(buf) > emitted:
                    yield buf[emitted:]
            finally:
                # Drain and join so the model is idle before adapter state flips.
                for _ in streamer:
                    pass
                worker.join(timeout=120)


ENGINE: Engine | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quieter logs
        pass

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_open(self):
        """Open an SSE response.

        The body length is unknown up front, and BaseHTTPRequestHandler does
        not apply chunked transfer-encoding automatically. Under HTTP/1.1 a
        response with neither Content-Length nor chunked framing has no way to
        signal its end, so clients block forever after the last event. Closing
        the connection at end-of-stream makes EOF the terminator, which is
        valid framing and what SSE clients handle natively.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path.rstrip("/") == "/v1/models":
            data = [{"id": SERVED_NAME, "object": "model", "owned_by": "memory-lora"}]
            if ENGINE is not None and ENGINE.has_adapter:
                data.append({"id": "base", "object": "model", "owned_by": "memory-lora"})
            return self._json(200, {"object": "list", "data": data})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        path = self.path.rstrip("/")
        is_chat = path == "/v1/chat/completions"
        if path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": "not found"})

        max_new = int(req.get("max_tokens") or 512)
        temperature = float(req.get("temperature", 0.0) or 0.0)
        top_p = float(req.get("top_p", 1.0) or 1.0)
        stream = bool(req.get("stream"))
        # The demo asks the SAME server for both sides of the comparison; the
        # model field selects which. "base" -> frozen model, anything else ->
        # repo-adapted.
        req_model = str(req.get("model") or "")
        use_adapter = "base" not in req_model.lower()
        stop = req.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = []
        if is_chat:
            prompt = ENGINE._apply_chat_template(req.get("messages", []))
        else:
            prompt = req.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "".join(map(str, prompt))

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            self._sse_open()
            with _LOCK:
                for piece in ENGINE.stream(prompt, max_new, temperature, top_p, stop, use_adapter):
                    if not piece:
                        continue
                    delta = ({"content": piece} if is_chat else None)
                    choice = ({"index": 0, "delta": delta, "finish_reason": None}
                              if is_chat else
                              {"index": 0, "text": piece, "finish_reason": None})
                    self._sse({
                        "id": cid, "object": "chat.completion.chunk" if is_chat
                        else "text_completion", "created": created,
                        "model": SERVED_NAME, "choices": [choice]})
            end_choice = ({"index": 0, "delta": {}, "finish_reason": "stop"}
                          if is_chat else
                          {"index": 0, "text": "", "finish_reason": "stop"})
            self._sse({"id": cid, "object": "chat.completion.chunk" if is_chat
                       else "text_completion", "created": created,
                       "model": SERVED_NAME, "choices": [end_choice]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        with _LOCK:
            text, n_in, n_out = ENGINE.generate(prompt, max_new, temperature, top_p, stop, use_adapter)
        usage = {"prompt_tokens": n_in, "completion_tokens": n_out,
                 "total_tokens": n_in + n_out}
        if is_chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": "stop"}
            obj = {"id": cid, "object": "chat.completion", "created": created,
                   "model": SERVED_NAME, "choices": [choice], "usage": usage}
        else:
            choice = {"index": 0, "text": text, "finish_reason": "stop"}
            obj = {"id": cid, "object": "text_completion", "created": created,
                   "model": SERVED_NAME, "choices": [choice], "usage": usage}
        return self._json(200, obj)


def main() -> None:
    global ENGINE, SERVED_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="merged model directory")
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--adapter", default="", help="LoRA adapter dir (base+adapter mode)")
    ap.add_argument("--served-name", default=SERVED_NAME)
    ap.add_argument("--port", type=int, default=config.SERVE_PORT)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--job", default="",
                    help="job id to re-attach to: rewrites that job's "
                         "status.json with this server's pid/port so the app "
                         "routes to it again after a manual restart")
    args = ap.parse_args()
    SERVED_NAME = args.served_name

    ENGINE = Engine(
        model_dir=args.model or None,
        base=args.base,
        adapter=args.adapter or None,
        device=args.device,
    )
    if args.job:
        # Re-attach: the app decides a job is usable by checking that the pid
        # in status.json is alive, so a manually restarted server has to
        # publish its own pid or the job keeps reporting a dead endpoint.
        import json as _json
        sp = config.workspace(args.job) / "status.json"
        st = _json.loads(sp.read_text()) if sp.exists() else {"job_id": args.job}
        st["server"] = {"backend": "fallback", "pid": os.getpid(),
                        "port": args.port,
                        "mode": "merged" if args.model else "lora"}
        st["state"] = "ready"
        st["stage"] = "ready"
        st["error"] = None
        st["endpoint"] = f"http://127.0.0.1:{args.port}"
        st["updated_at"] = time.time()
        sp.write_text(_json.dumps(st, indent=2))
        print(f"[serve] re-attached to job {args.job}", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] OpenAI-compatible server on http://0.0.0.0:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
