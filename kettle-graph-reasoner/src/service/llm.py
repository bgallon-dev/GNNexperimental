r"""Tiny OpenAI-compatible LLM client for the KGR explorer.

The downstream consumer in the KGR architecture: KGR selects a high-signal
subgraph, this calls a LOCAL small LLM to answer over it. No new deps
(urllib only); auto-detects a running local server.

Detection order: $KGR_LLM_BASE, then LM Studio (:1234), then Ollama
(:11434). Both speak the OpenAI /v1 API. Embedding-only models are
skipped when picking a default.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

CANDIDATES = [
    os.getenv("KGR_LLM_BASE", "").rstrip("/") or None,
    "http://localhost:1234/v1",     # LM Studio
    "http://localhost:11434/v1",    # Ollama
]
_STATE: dict = {}


def _get(url: str, timeout: float = 4.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def discover() -> dict:
    """Return {base, models, default} for the first reachable server."""
    if _STATE.get("base"):
        return _STATE
    for base in [c for c in CANDIDATES if c]:
        try:
            data = _get(base + "/models")
        except Exception:
            continue
        ids = [m.get("id", "") for m in data.get("data", [])]
        chat = [m for m in ids if m and "embed" not in m.lower()]
        if chat:
            # prefer the smallest-looking model as the "small LLM" default
            default = sorted(chat, key=_size_hint)[0]
            _STATE.update(base=base, models=chat, default=default)
            return _STATE
    _STATE.update(base=None, models=[], default=None)
    return _STATE


def _size_hint(name: str) -> float:
    """Rough param-count guess from the model id, for picking a small one."""
    import re
    m = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    if m:
        return float(m[-1])
    if "e4b" in name.lower() or "mini" in name.lower():
        return 4.0
    return 999.0


def available() -> bool:
    return bool(discover().get("base"))


def chat(messages, model: str | None = None, temperature: float = 0.2,
         max_tokens: int = 2048, timeout: float = 240.0) -> dict:
    """Call /v1/chat/completions. Returns {text, model, usage} or raises.
    max_tokens is generous because some small local models (e.g. gemma-4)
    are REASONING models that spend most tokens thinking before answering;
    too small a budget yields an empty content field."""
    st = discover()
    if not st.get("base"):
        raise RuntimeError("no local LLM server found (start LM Studio on "
                           ":1234 or Ollama on :11434 with a model loaded)")
    model = model or st["default"]
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(st["base"] + "/chat/completions", body,
                                 {"content-type": "application/json"})
    # local single-model servers (LM Studio) can 5xx when hit while busy;
    # retry once after a short pause before surfacing the error.
    import time
    d = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            code, snippet = e.code, e.read().decode("utf-8", "ignore")[:200]
            if 500 <= code < 600 and attempt == 0:
                time.sleep(1.2)
                continue
            raise RuntimeError(f"LLM error {code} (the local model server "
                               f"returned an error — often transient/busy): "
                               f"{snippet}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 0:
                time.sleep(1.2)
                continue
            raise RuntimeError(f"could not reach the local LLM server: {e}")
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    # strip an inline <think>...</think> block if the model embeds reasoning
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # some reasoning models leave content empty and put prose in reasoning_*
    if not text:
        text = (msg.get("reasoning_content") or msg.get("reasoning")
                or "").strip()
    usage = d.get("usage", {})
    if not text:
        rt = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        text = (f"(the model used its full token budget reasoning"
                + (f" — {rt} reasoning tokens" if rt else "")
                + " — and did not emit an answer; try a smaller top-k or the "
                "12B model)")
    return {"text": text, "model": d.get("model", model), "usage": usage}
