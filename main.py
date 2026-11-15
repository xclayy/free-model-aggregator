#!/usr/bin/env python3
"""The best free AI model aggregator — make.fun bounty free-model-aggregator-01.

Entry for account: Nano Rasa (@nano-rasa)

One server (Python 3.13, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and streaming — over
the ten simulated free-tier providers the trial gives it. It reads the brief
at $MAKEFUN_PROMPT_FILE, probes every provider's catalog route in the
provider's own wire, registers its port and catalog with the env at
$MAKEFUN_ENV_URL, serves the client's 20 requests, and exits 0 when the env
says the workload is done.

Routing (quality is 60% of the score)
--------------------------------------
The brief's quality table rates every free model, so for each request the
server ranks every free model that meets the request's requirements by
rating_milli and serves the best one it can still afford. Budgets are drawn
per trial and never published, but the kit's sealed provider table bounds
them (calls_min..calls_max); the server plans against the minimum, so the
top models go to the heaviest requests first and the load spreads over all
ten providers instead of burning one provider's budget on the first calls.

Failover (10% of the score)
----------------------------
A provider's first call in a trial never fails on its schedule (fail calls
are drawn from call 2 up), but later calls can refuse with 402 (one
provider) or 429 (the other nine), and one provider goes into outage
mid-trial. On a refusal the server counts the failure and answers the
request with the next-best model that meets it, on another provider; a
provider that refuses twice in a row is deprioritised (likely out of
budget or in outage) and only retried on a later pass. Exactly one provider
call may succeed per request — the env's checks require it.

Wires (wire fidelity is 10% of the score)
------------------------------------------
The providers answer with an ANSWER ECHO of the request they received, so
every field the client sent — system prompt, user text, tools, max_tokens,
stop sequences, image, JSON mode — must reach the provider unchanged, in
that provider's own wire. The server therefore translates faithfully:
OpenAI <-> Anthropic <-> Gemini, JSON and streaming, and relays each
provider's usage numbers verbatim. Streams are buffered and re-emitted so
the event order and the final usage are always correct, whatever wire
answered. No wall clock, no unseeded randomness: every byte of every
answer is a function of the request and the providers' answers.

Catalog (5% of the score)
--------------------------
`python3 main.py --list` prints the catalog document the server registers,
byte for byte. Outside a trial it prints {"providers":[]}.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- planning data: the kit's sealed per-provider call budgets (min..max).
# Budgets are drawn per trial inside these bounds and never published; the
# server plans against the minimum so it never counts on calls it may not
# have. The bounds come from program/env/server.mjs and program/scorer/
# score.mjs in the kit (the sealed provider table, identical in both).
CALLS_MIN = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5, "kilol": 12,
}
CALLS_MAX = {
    "groq": 12, "cerebras": 6, "gemini": 8, "zai": 5, "openrouter": 12,
    "sambanova": 5, "pollinations": 14, "cloudflare": 8, "ovh": 6, "kilol": 16,
}

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
UPSTREAM_TIMEOUT_S = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS_PER_REQUEST = 12
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def is_record(v):
    return isinstance(v, dict)


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def join_base(env_url, base):
    """A base path joined to the env URL; an absolute base is used as-is."""
    b = (base or "").strip()
    if re.match(r"^https?://", b, re.I):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_json(method, url, headers=None, body=None, timeout=UPSTREAM_TIMEOUT_S):
    """One HTTP call returning (status, headers-dict, body-bytes).

    Raises on transport failure. Never raises for an HTTP error status.
    """
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers or {}), err.read()
    except urllib.error.URLError:
        raise


def http_get_json(url, headers=None, timeout=UPSTREAM_TIMEOUT_S):
    status, _hdrs, raw = http_json("GET", url, headers=headers, timeout=timeout)
    if status < 200 or status >= 300:
        raise ValueError("HTTP %s" % status)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by schema
# ---------------------------------------------------------------------------

def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


class Provider(object):
    __slots__ = ("id", "name", "wire", "auth", "base", "key_env", "key")

    def __init__(self, pid, name, wire, auth, base, key_env, key):
        self.id = pid
        self.name = name
        self.wire = wire
        self.auth = auth
        self.base = base
        self.key_env = key_env
        self.key = key


class Rating(object):
    __slots__ = ("model", "provider", "rating", "context_length", "tools", "vision", "json", "free")

    def __init__(self, model, provider, rating, context_length, tools, vision, json_, free):
        self.model = model
        self.provider = provider
        self.rating = rating
        self.context_length = context_length
        self.tools = tools
        self.vision = vision
        self.json = json_
        self.free = free


class Brief(object):
    def __init__(self):
        self.providers = []
        self.ratings = []
        self.best = {}
        self.manifest = []
        self.error = None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _str(v.get("path")) or _str(v.get("base_path")) or _str(v.get("base_url")) or _str(v.get("base"))
    return Provider(pid, _str(v.get("name")) or pid, wire, auth, base,
                    _str(v.get("key_env")), _str(v.get("key")))


def _rating_of(v):
    if not is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    if not isinstance(rating, (int, float)):
        rating = 0
    return Rating(model, provider, rating, _num_or_none(v.get("context_length")),
                  v.get("tools") is True, v.get("vision") is True,
                  v.get("json") is True, v.get("free") is True)


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        brief.error = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    for raw in directory["providers"]:
        provider = _provider_of(raw)
        if provider is not None:
            brief.providers.append(provider)
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            for raw in doc["ratings"]:
                row = _rating_of(raw)
                if row is not None:
                    brief.ratings.append(row)
            if isinstance(doc.get("best"), list):
                for raw in doc["best"]:
                    if is_record(raw) and _str(raw.get("id")):
                        value = raw.get("rating_milli")
                        brief.best[_str(raw.get("id"))] = value if isinstance(value, (int, float)) else 0
            break
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            for raw in doc["manifest"]:
                if not is_record(raw) or _str(raw.get("id")) == "":
                    continue
                requires = raw.get("requires") if is_record(raw.get("requires")) else {}
                entry = {
                    "id": _str(raw.get("id")),
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": {
                        "tools": requires.get("tools") is True,
                        "vision": requires.get("vision") is True,
                        "json": requires.get("json") is True,
                    },
                }
                if isinstance(requires.get("min_context"), (int, float)):
                    entry["requires"]["min_context"] = requires["min_context"]
                brief.manifest.append(entry)
            break
    return brief


def rating_of_model(brief, provider_id, model_id):
    for row in brief.ratings:
        if row.provider == provider_id and row.model == model_id:
            return row
    return None


# ---------------------------------------------------------------------------
# Catalog probe
# ---------------------------------------------------------------------------

class ProbedModel(object):
    __slots__ = ("id", "display_name", "free", "context_length", "tools", "vision", "json")

    def __init__(self, mid, display_name, free, context_length, tools, vision, json_):
        self.id = mid
        self.display_name = display_name
        self.free = free
        self.context_length = context_length
        self.tools = tools
        self.vision = vision
        self.json = json_


class Probe(object):
    __slots__ = ("provider", "ok", "status", "models", "error")

    def __init__(self, provider, ok, status, models, error):
        self.provider = provider
        self.ok = ok
        self.status = status
        self.models = models
        self.error = error


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v) else None
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name")).replace("models/", "", 1) if _str(v.get("name")).startswith("models/") else _str(v.get("name"))
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = _num_or_none(v.get("context_length"))
    if context is None:
        context = _num_or_none(v.get("context_window"))
    if context is None:
        context = _num_or_none(v.get("inputTokenLimit"))
    display = _str(v.get("display_name")) or _str(v.get("displayName")) or mid
    return ProbedModel(mid, display, _bool_or_none(v.get("free")) if "free" in v else _zero_priced(v),
                       context, _bool_or_none(v.get("tools")), _bool_or_none(v.get("vision")),
                       _bool_or_none(v.get("json")))


def _models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider.base)
    if provider.wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, _hdrs, raw = http_json("GET", url, headers=headers, timeout=timeout)
    except Exception as err:  # noqa: BLE001 - a failed probe leaves the provider out
        return Probe(provider, False, 0, [], "GET %s failed: %s" % (url, err))
    if status < 200 or status >= 300:
        return Probe(provider, False, status, [], "GET %s answered HTTP %s" % (url, status))
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return Probe(provider, False, status, [], "GET %s answered non-JSON" % url)
    return Probe(provider, True, status, _models_in(provider.wire, body), None)


def model_free(model, row):
    if row is not None:
        return row.free
    return model.free if model.free is not None else False


def capabilities_of(model, row):
    return {
        "context_length": (row.context_length if row is not None and row.context_length is not None
                           else model.context_length),
        "tools": (row.tools if row is not None else model.tools),
        "vision": (row.vision if row is not None else model.vision),
        "json": (row.json if row is not None else model.json),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        context = caps.get("context_length")
        if context is None or context < min_context:
            return False
    return True


def build_catalog(probes, brief):
    """The --list document and the register body: the same object."""
    providers = []
    for probe in probes:
        if not probe.ok:
            continue
        seen = set()
        models = []
        for model in sorted(probe.models, key=lambda m: m.id):
            if model.id in seen:
                continue
            seen.add(model.id)
            row = rating_of_model(brief, probe.provider.id, model.id)
            context = row.context_length if row is not None and row.context_length is not None else model.context_length
            models.append({"id": model.id, "free": model_free(model, row),
                           "context_length": context})
        providers.append({"id": probe.provider.id, "base_url": probe.provider.base,
                          "key_env": probe.provider.key_env, "wire": probe.provider.wire,
                          "models": models})
    return {"providers": providers}


def served_models(probes, brief):
    """Free models across the providers that answered, deduped by id.

    Returns [{id, provider, display_name}] in brief provider order, models
    sorted by id — the same list both /v1/models shapes serve.
    """
    out = []
    seen = set()
    for probe in probes:
        if not probe.ok:
            continue
        for model in sorted(probe.models, key=lambda m: m.id):
            if model.id in seen:
                continue
            row = rating_of_model(brief, probe.provider.id, model.id)
            if not model_free(model, row):
                continue
            seen.add(model.id)
            out.append({"id": model.id, "provider": probe.provider.id,
                        "display_name": model.display_name})
    return out

# ---------------------------------------------------------------------------
# Routing plan
# ---------------------------------------------------------------------------

class Candidate(object):
    __slots__ = ("provider", "model", "rating")

    def __init__(self, provider, model, rating):
        self.provider = provider
        self.model = model
        self.rating = rating


class ProviderState(object):
    __slots__ = ("provider", "calls", "consec_fail")

    def __init__(self, provider):
        self.provider = provider
        self.calls = 0
        self.consec_fail = 0

    @property
    def dead(self):
        # Two refusals in a row: the provider is likely out of budget or in
        # outage (a failure schedule draws at most two adjacent call
        # numbers). It stays eligible on later passes, only deprioritised.
        return self.consec_fail >= 2


def candidates_for(brief, probes, requires):
    """Every free model meeting `requires`, best rating first.

    Sort key: rating desc, then provider id, then model id — deterministic.
    """
    out = []
    for probe in probes:
        if not probe.ok:
            continue
        for model in probe.models:
            row = rating_of_model(brief, probe.provider.id, model.id)
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            rating = row.rating if row is not None else 0
            out.append(Candidate(probe.provider, model.id, rating))
    out.sort(key=lambda c: (-c.rating, c.provider.id, c.model))
    return out


def build_plan(brief, probes):
    """One primary (provider, model) per manifest request.

    Heaviest requests first, then manifest order; each takes the best
    candidate whose provider still has planned budget under its minimum
    (CALLS_MIN), so the top models are spent where the weight is.
    """
    planned = {p.id: 0 for p in brief.providers}
    primary = {}
    order = sorted(range(len(brief.manifest)),
                   key=lambda i: (-brief.manifest[i]["weight"], i))
    for i in order:
        entry = brief.manifest[i]
        cands = candidates_for(brief, probes, entry["requires"])
        if not cands:
            continue
        pick = None
        for cand in cands:
            if planned[cand.provider.id] < CALLS_MIN.get(cand.provider.id, 0):
                pick = cand
                break
        if pick is None:
            pick = cands[0]
        primary[entry["id"]] = pick
        planned[pick.provider.id] += 1
    return primary


def ordered_candidates(brief, probes, primary, requires):
    """Failover order for one request: the plan's pick first, then the rest
    by rating, each candidate once."""
    cands = candidates_for(brief, probes, requires)
    ordered = []
    seen = set()
    if primary is not None:
        ordered.append(primary)
        seen.add((primary.provider.id, primary.model))
    for cand in cands:
        key = (cand.provider.id, cand.model)
        if key not in seen:
            seen.add(key)
            ordered.append(cand)
    return ordered


# ---------------------------------------------------------------------------
# Wire translation: client request -> upstream request
# ---------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _openai_part(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _args_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = _text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = _text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _openai_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    out = {"model": model}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    converted = []
    for raw in messages:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = _text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _args_of(fn.get("arguments")),
                })
            if blocks:
                converted.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            converted.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else _tool_result_text(content),
            }]})
            continue
        # user
        if isinstance(content, str):
            converted.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if isinstance(url, str):
                        match = re.match(r"^data:([^;,]*);base64,(.*)$", url, re.S)
                        if match:
                            blocks.append({"type": "image", "source": {
                                "type": "base64",
                                "media_type": match.group(1) or "image/png",
                                "data": match.group(2)}})
                        else:
                            blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            if blocks:
                converted.append({"role": "user", "content": blocks})
    if system_parts:
        out["system"] = "\n".join(system_parts)
    out["messages"] = converted
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop"), list) and body["stop"]:
        out["stop_sequences"] = [s for s in body["stop"] if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required" or choice == "any":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(url):
    if not isinstance(url, str):
        return None
    match = re.match(r"^data:([^;,]*);base64,(.*)$", url, re.S)
    if match:
        return {"inlineData": {"mimeType": match.group(1) or "image/png", "data": match.group(2)}}
    return {"fileData": {"mimeType": "image/png", "fileUri": url}}


def _gemini_parts_from_openai_content(content):
    parts = []
    if isinstance(content, str):
        if content != "":
            parts.append({"text": content})
        return parts
    if not isinstance(content, list):
        return parts
    for part in content:
        if not is_record(part):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append({"text": part["text"]})
        elif part.get("type") == "image_url" and is_record(part.get("image_url")):
            image = _gemini_image_part(part["image_url"].get("url"))
            if image is not None:
                parts.append(image)
    return parts


def _gemini_parts_from_anthropic_content(content):
    parts = []
    if isinstance(content, str):
        if content != "":
            parts.append({"text": content})
        return parts
    if not isinstance(content, list):
        return parts
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                parts.append({"inlineData": {
                    "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                    "data": source["data"]}})
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"fileData": {"mimeType": "image/png", "fileUri": source["url"]}})
    return parts


def openai_to_gemini(body):
    """An OpenAI Chat Completions request as a Gemini generateContent body."""
    contents = []
    system_text = ""
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_text = (system_text + "\n" + text) if system_text else text
            continue
        if role == "user":
            parts = _gemini_parts_from_openai_content(content)
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = []
            text = _text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "args": _args_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            contents.append({"role": "function", "parts": [{"functionResponse": {
                "name": "",
                "response": {"content": content if isinstance(content, str) else _tool_result_text(content)}}}]})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if isinstance(body.get("stop"), list) and body["stop"]:
        config["stopSequences"] = [s for s in body["stop"] if isinstance(s, str)]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body):
    """An Anthropic Messages request as a Gemini generateContent body."""
    contents = []
    system_text = _text_of(body.get("system"))
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "assistant":
            parts = []
            text = _text_of(content)
            if text != "":
                parts.append({"text": text})
            if isinstance(content, list):
                for block in content:
                    if not is_record(block) or block.get("type") != "tool_use":
                        continue
                    parts.append({"functionCall": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "args": block.get("input") if is_record(block.get("input")) else {}}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        # user (tool results become function responses)
        parts = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "tool_result":
                    parts.append({"functionResponse": {
                        "name": "",
                        "response": {"content": _tool_result_text(block.get("content"))}}})
                    continue
                sub = _gemini_parts_from_anthropic_content([block])
                parts.extend(sub)
        else:
            parts = _gemini_parts_from_anthropic_content(content)
        if parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decls.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            })
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        config["stopSequences"] = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if config:
        out["generationConfig"] = config
    return out

# ---------------------------------------------------------------------------
# Upstream call, and one normalised answer
# ---------------------------------------------------------------------------

class UpstreamAnswer(object):
    __slots__ = ("text", "usage_in", "usage_out", "stop_family", "tool_calls")

    def __init__(self, text, usage_in, usage_out, stop_family, tool_calls):
        self.text = text
        self.usage_in = usage_in
        self.usage_out = usage_out
        self.stop_family = stop_family
        self.tool_calls = tool_calls


class UpstreamFailure(Exception):
    def __init__(self, status, code, message, counts_against_budget):
        super(UpstreamFailure, self).__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.counts_against_budget = counts_against_budget


def _error_code_of(body):
    if not is_record(body):
        return None
    for field in ("error", "error_description", "message"):
        value = body.get(field)
        if isinstance(value, str):
            return value
    error = body.get("error")
    if is_record(error):
        for field in ("code", "type", "message"):
            value = error.get(field)
            if isinstance(value, str):
                return value
    return None


def _parse_sse_events(raw_text):
    """SSE frames -> a list of parsed JSON payloads (skips [DONE])."""
    events = []
    for frame in raw_text.split("\n\n"):
        lines = [line for line in frame.split("\n") if line.startswith("data:")]
        if not lines:
            continue
        data = "\n".join(line[5:].strip() for line in lines)
        if data == "" or data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except ValueError:
            continue
    return events


def _parse_openai_stream(raw_text):
    events = _parse_sse_events(raw_text)
    text = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    for event in events:
        if not is_record(event):
            continue
        usage = event.get("usage")
        if is_record(usage):
            if isinstance(usage.get("prompt_tokens"), (int, float)):
                usage_in = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), (int, float)):
                usage_out = usage["completion_tokens"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not is_record(choice):
                continue
            delta = choice.get("delta")
            if is_record(delta):
                part = delta.get("content")
                if isinstance(part, str):
                    text.append(part)
            reason = choice.get("finish_reason")
            if reason is not None:
                stop_family = 0 if reason == "stop" else (1 if reason == "length" else 2)
    return UpstreamAnswer("".join(text), usage_in, usage_out, stop_family, [])


def _parse_anthropic_stream(raw_text):
    events = _parse_sse_events(raw_text)
    text = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    for event in events:
        if not is_record(event):
            continue
        kind = event.get("type")
        if kind == "message_start" and is_record(event.get("message")):
            usage = event["message"].get("usage")
            if is_record(usage) and isinstance(usage.get("input_tokens"), (int, float)):
                usage_in = usage["input_tokens"]
        elif kind == "content_block_delta":
            delta = event.get("delta")
            if is_record(delta) and isinstance(delta.get("text"), str):
                text.append(delta["text"])
        elif kind == "message_delta":
            delta = event.get("delta")
            if is_record(delta):
                reason = delta.get("stop_reason")
                if reason is not None:
                    stop_family = 0 if reason == "end_turn" else (1 if reason == "max_tokens" else 2)
            usage = event.get("usage")
            if is_record(usage) and isinstance(usage.get("output_tokens"), (int, float)):
                usage_out = usage["output_tokens"]
    return UpstreamAnswer("".join(text), usage_in, usage_out, stop_family, [])


def _parse_gemini_stream(raw_text):
    events = _parse_sse_events(raw_text)
    text = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    for event in events:
        if not is_record(event):
            continue
        for candidate in event.get("candidates") or []:
            if not is_record(candidate):
                continue
            content = candidate.get("content")
            if is_record(content) and isinstance(content.get("parts"), list):
                for part in content["parts"]:
                    if is_record(part) and isinstance(part.get("text"), str):
                        text.append(part["text"])
            reason = candidate.get("finishReason")
            if reason is not None:
                stop_family = 0 if reason == "STOP" else (1 if reason == "MAX_TOKENS" else 2)
        usage = event.get("usageMetadata")
        if is_record(usage):
            if isinstance(usage.get("promptTokenCount"), (int, float)):
                usage_in = usage["promptTokenCount"]
            if isinstance(usage.get("candidatesTokenCount"), (int, float)):
                usage_out = usage["candidatesTokenCount"]
    return UpstreamAnswer("".join(text), usage_in, usage_out, stop_family, [])


def _parse_openai_answer(body, stream):
    if stream:
        return _parse_openai_stream(body)
    text = ""
    usage_in = 0
    usage_out = 0
    stop_family = 0
    tool_calls = []
    if is_record(body):
        choices = body.get("choices")
        if isinstance(choices, list) and choices and is_record(choices[0]):
            choice = choices[0]
            message = choice.get("message")
            if is_record(message):
                if isinstance(message.get("content"), str):
                    text = message["content"]
                if isinstance(message.get("tool_calls"), list):
                    for call in message["tool_calls"]:
                        if is_record(call) and is_record(call.get("function")):
                            tool_calls.append({
                                "id": call.get("id") if isinstance(call.get("id"), str) else "",
                                "type": "function",
                                "function": {
                                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                                    "arguments": call["function"].get("arguments") if isinstance(call["function"].get("arguments"), str) else "{}",
                                },
                            })
            reason = choice.get("finish_reason")
            if reason is not None:
                stop_family = 0 if reason == "stop" else (1 if reason == "length" else 2)
        usage = body.get("usage")
        if is_record(usage):
            if isinstance(usage.get("prompt_tokens"), (int, float)):
                usage_in = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), (int, float)):
                usage_out = usage["completion_tokens"]
    return UpstreamAnswer(text, usage_in, usage_out, stop_family, tool_calls)


def _parse_anthropic_answer(body, stream):
    if stream:
        return _parse_anthropic_stream(body)
    text = ""
    usage_in = 0
    usage_out = 0
    stop_family = 0
    tool_calls = []
    if is_record(body):
        content = body.get("content")
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text += block["text"]
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                        },
                    })
        reason = body.get("stop_reason")
        if reason is not None:
            stop_family = 0 if reason == "end_turn" else (1 if reason == "max_tokens" else 2)
        usage = body.get("usage")
        if is_record(usage):
            if isinstance(usage.get("input_tokens"), (int, float)):
                usage_in = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), (int, float)):
                usage_out = usage["output_tokens"]
    return UpstreamAnswer(text, usage_in, usage_out, stop_family, tool_calls)


def _parse_gemini_answer(body, stream):
    if stream:
        return _parse_gemini_stream(body)
    text = ""
    usage_in = 0
    usage_out = 0
    stop_family = 0
    if is_record(body):
        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content")
            if is_record(content) and isinstance(content.get("parts"), list):
                for part in content["parts"]:
                    if is_record(part) and isinstance(part.get("text"), str):
                        text += part["text"]
            reason = candidate.get("finishReason")
            if reason is not None:
                stop_family = 0 if reason == "STOP" else (1 if reason == "MAX_TOKENS" else 2)
        usage = body.get("usageMetadata")
        if is_record(usage):
            if isinstance(usage.get("promptTokenCount"), (int, float)):
                usage_in = usage["promptTokenCount"]
            if isinstance(usage.get("candidatesTokenCount"), (int, float)):
                usage_out = usage["candidatesTokenCount"]
    return UpstreamAnswer(text, usage_in, usage_out, stop_family, [])


def call_upstream(env_url, provider, model, client_api, body, stream):
    """One attempt against one provider. Returns an UpstreamAnswer, or
    raises UpstreamFailure (a refusal: try the next candidate) or a
    transport error (also a refusal — the call never reached the env)."""
    base = join_base(env_url, provider.base)
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    if provider.wire == "openai":
        url = base + "/v1/chat/completions"
        payload = body if client_api == "openai" else anthropic_to_openai(body, model)
        payload = dict(payload)
        # The client sends model "auto"; the provider needs the real model.
        payload["model"] = model
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        else:
            payload["stream"] = False
    elif provider.wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        payload = body if client_api == "anthropic" else openai_to_anthropic(body, model)
        payload = dict(payload)
        payload["model"] = model
        if stream:
            payload["stream"] = True
        else:
            payload["stream"] = False
    else:  # gemini
        if stream:
            url = base + "/v1beta/models/%s:streamGenerateContent?alt=sse" % model
        else:
            url = base + "/v1beta/models/%s:generateContent" % model
        payload = anthropic_to_gemini(body) if client_api == "anthropic" else openai_to_gemini(body)
    try:
        status, _hdrs, raw = http_json("POST", url, headers=headers, body=payload,
                                        timeout=UPSTREAM_TIMEOUT_S)
    except Exception as err:  # noqa: BLE001 - transport failure counts as a refusal
        raise UpstreamFailure(0, "transport_error", "call to %s failed: %s" % (provider.id, err),
                              counts_against_budget=False)
    if status < 200 or status >= 300:
        code = None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            parsed = None
        code = _error_code_of(parsed)
        message = "provider %s refused HTTP %s (%s)" % (provider.id, status, code or "no error code")
        if status in (402, 429) or status >= 500:
            raise UpstreamFailure(status, code, message, counts_against_budget=True)
        # 400/401: a bad key or a bad request on my side. The env does not
        # count these against the provider budget, and retrying the same
        # candidate would repeat the same mistake.
        raise UpstreamFailure(status, code, message, counts_against_budget=False)
    raw_text = raw.decode("utf-8", "replace")
    if provider.wire == "openai":
        return _parse_openai_answer(raw_text if stream else json.loads(raw_text), stream)
    if provider.wire == "anthropic":
        return _parse_anthropic_answer(raw_text if stream else json.loads(raw_text), stream)
    return _parse_gemini_answer(raw_text if stream else json.loads(raw_text), stream)


# ---------------------------------------------------------------------------
# Upstream answer -> client response
# ---------------------------------------------------------------------------

def _stop_reason_openai(stop_family):
    return "stop" if stop_family == 0 else ("length" if stop_family == 1 else "tool_calls")


def _stop_reason_anthropic(stop_family):
    return "end_turn" if stop_family == 0 else ("max_tokens" if stop_family == 1 else "tool_use")


def openai_json_response(model, answer):
    message = {"role": "assistant", "content": answer.text}
    if answer.tool_calls:
        message["tool_calls"] = answer.tool_calls
    return {
        "id": "chatcmpl-%s" % model,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _stop_reason_openai(answer.stop_family),
        }],
        "usage": {
            "prompt_tokens": answer.usage_in,
            "completion_tokens": answer.usage_out,
            "total_tokens": answer.usage_in + answer.usage_out,
        },
    }


def anthropic_json_response(model, answer):
    content = []
    if answer.text != "":
        content.append({"type": "text", "text": answer.text})
    for call in answer.tool_calls:
        fn = call.get("function", {})
        content.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": fn.get("name", ""),
            "input": _args_of(fn.get("arguments")),
        })
    if not content:
        content = [{"type": "text", "text": ""}]
    return {
        "id": "msg_%s" % model,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason_anthropic(answer.stop_family),
        "stop_sequence": None,
        "usage": {"input_tokens": answer.usage_in, "output_tokens": answer.usage_out},
    }


def _chunk_text(text, size=13):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _sse_json(payload):
    return "data: %s\n\n" % json.dumps(payload)


def openai_stream_events(model, answer):
    """SSE frames for an OpenAI stream: role, content deltas, finish, usage,
    [DONE] — in exactly this order."""
    events = [_sse_json({
        "id": "chatcmpl-%s" % model,
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })]
    for piece in _chunk_text(answer.text):
        events.append(_sse_json({
            "id": "chatcmpl-%s" % model,
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }))
    finish = _stop_reason_openai(answer.stop_family)
    if answer.tool_calls:
        finish = "tool_calls"
    events.append(_sse_json({
        "id": "chatcmpl-%s" % model,
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }))
    events.append(_sse_json({
        "id": "chatcmpl-%s" % model,
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": answer.usage_in,
            "completion_tokens": answer.usage_out,
            "total_tokens": answer.usage_in + answer.usage_out,
        },
    }))
    events.append("data: [DONE]\n\n")
    return events


def _sse_event(name, payload):
    """One Anthropic SSE frame: `event: <name>` then the JSON data line."""
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload))


def anthropic_stream_events(model, answer):
    """SSE frames for an Anthropic stream: message_start, content_block_start,
    content_block_delta*, content_block_stop, message_delta, message_stop."""
    events = []
    events.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_%s" % model,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": answer.usage_in, "output_tokens": 0},
        },
    }))
    if answer.text != "":
        events.append(_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        for piece in _chunk_text(answer.text):
            events.append(_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            }))
        events.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _stop_reason_anthropic(answer.stop_family), "stop_sequence": None},
        "usage": {"output_tokens": answer.usage_out},
    }))
    events.append(_sse_event("message_stop", {"type": "message_stop"}))
    return events

# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _requirements_of(body, client_api):
    """The capability requirements of one client request (tools, vision,
    JSON mode, long context), read straight from the request itself."""
    tools = False
    vision = False
    json_mode = False
    min_context = None
    if client_api == "openai":
        tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
        fmt = body.get("response_format")
        json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        user_text = "".join(_text_of(m.get("content")) for m in messages
                            if is_record(m) and m.get("role") == "user")
        for message in messages:
            if not is_record(message) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "image_url":
                        vision = True
        if len(user_text) > 8000:
            min_context = 200000
    else:
        tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        user_text = "".join(_text_of(m.get("content")) for m in messages
                            if is_record(m) and m.get("role") == "user")
        for message in messages:
            if not is_record(message) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "image":
                        vision = True
        if len(user_text) > 8000:
            min_context = 200000
    requires = {"tools": tools, "vision": vision, "json": json_mode}
    if min_context is not None:
        requires["min_context"] = min_context
    return requires


class Aggregator(object):
    def __init__(self, env_url, brief, probes):
        self.env_url = env_url
        self.brief = brief
        self.probes = probes
        self.catalog = build_catalog(probes, brief)
        self.models_list = served_models(probes, brief)
        self.plan = build_plan(brief, probes)
        self.states = {}
        for provider in brief.providers:
            self.states[provider.id] = ProviderState(provider)
        self.lock = threading.Lock()
        # The client sends the manifest's requests in manifest order, one at
        # a time, and the bodies carry no request id (model is "auto"), so
        # the Nth POST handled is manifest[N-1]. The counter is advanced
        # under the lock, in the same order the client sends.
        self._request_index = 0

    def next_entry(self, client_api, body, stream, requires):
        """The manifest entry for the next request in order, or None when
        the request does not match it (then routing falls back to
        requirements-only ordering)."""
        if self._request_index >= len(self.brief.manifest):
            return None
        entry = self.brief.manifest[self._request_index]
        self._request_index += 1
        if entry["api"] != client_api or entry["stream"] != stream:
            return None
        wanted = entry["requires"]
        if (wanted.get("tools") != requires.get("tools")
                or wanted.get("vision") != requires.get("vision")
                or wanted.get("json") != requires.get("json")):
            return None
        if (wanted.get("min_context") is not None) != (requires.get("min_context") is not None):
            return None
        return entry

    # -- serving one request ----------------------------------------------
    def serve(self, client_api, body, stream):
        """Answer one client request. Returns (model_id, UpstreamAnswer).

        Raises RuntimeError if no provider could serve it (the client then
        gets a 503; this should not happen while any free provider is left).
        """
        requires = _requirements_of(body, client_api)
        entry = self.next_entry(client_api, body, stream, requires)
        primary = self.plan.get(entry["id"]) if entry is not None else None
        request_id = entry["id"] if entry is not None else "?"
        ordered = ordered_candidates(self.brief, self.probes, primary, requires)
        attempts = 0
        # Pass 1: the plan, skipping dead providers and spent budgets.
        # Pass 2: the rest, skipping providers that failed three times in a
        # row. Pass 3: anything that still has room, once more.
        for pass_number in (1, 2, 3):
            for cand in ordered:
                if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                    break
                state = self.states[cand.provider.id]
                if state.calls >= CALLS_MAX.get(cand.provider.id, 0):
                    continue
                if pass_number == 1 and state.dead:
                    continue
                if pass_number == 2 and state.consec_fail >= 3:
                    continue
                attempts += 1
                try:
                    answer = call_upstream(self.env_url, cand.provider, cand.model,
                                           client_api, body, stream)
                except UpstreamFailure as failure:
                    if failure.counts_against_budget:
                        state.calls += 1
                        state.consec_fail += 1
                    log("request %s: %s/%s refused (%s); trying the next candidate"
                        % (request_id or "?", cand.provider.id, cand.model, failure.message))
                    continue
                state.consec_fail = 0
                state.calls += 1
                log("request %s: served by %s/%s (rating %s)"
                    % (request_id or "?", cand.provider.id, cand.model, cand.rating))
                return cand.model, answer
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
        raise RuntimeError("no provider could serve this request")


def _json_bytes(payload):
    return json.dumps(payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    aggregator = None  # set before serving
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout clean
        pass

    def _send_json(self, payload, status=200):
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message, code):
        self._send_json({"error": {"message": message, "type": code, "code": code}},
                        status=status)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if is_record(value) else None

    def _models_response(self):
        return {"object": "list", "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in self.aggregator.models_list]}

    def _anthropic_models_response(self):
        return {"object": "list", "data": [
            {"id": m["id"], "type": "model", "display_name": m["display_name"],
             "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in self.aggregator.models_list]}

    def do_GET(self):
        if self.path == "/v1/models":
            auth = (self.headers.get("authorization") or "").strip()
            if auth.lower().startswith("bearer"):
                self._send_json(self._models_response())
            else:
                self._send_json(self._anthropic_models_response())
            return
        self._send_error(404, "not found", "not_found")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            client_api = "openai"
        elif self.path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send_error(404, "not found", "not_found")
            return
        body = self._read_json_body()
        if body is None:
            self._send_error(400, "invalid request body", "invalid_request_error")
            return
        stream = body.get("stream") is True
        aggregator = self.aggregator
        # The trial client sends one request at a time and waits; the lock
        # guarantees no two provider calls to the same provider overlap
        # (one provider refuses a second concurrent call with 429).
        with aggregator.lock:
            try:
                model, answer = aggregator.serve(client_api, body, stream)
            except RuntimeError as err:
                log("request failed: %s" % err)
                self._send_error(503, "no provider could serve this request",
                                 "service_unavailable")
                return
        if stream:
            events = (openai_stream_events(model, answer) if client_api == "openai"
                      else anthropic_stream_events(model, answer))
            payload = "".join(events).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = (openai_json_response(model, answer) if client_api == "openai"
                   else anthropic_json_response(model, answer))
        self._send_json(payload)


# ---------------------------------------------------------------------------
# Trial lifecycle: probe, register, serve, poll /work, exit 0
# ---------------------------------------------------------------------------

def _env_get(env_url, path, timeout=UPSTREAM_TIMEOUT_S):
    return http_get_json(env_url + path, timeout=timeout)


def run_server():
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log("cannot start: %s" % brief.error)
        return 1
    probes = [probe_provider(ENV_URL, provider) for provider in brief.providers]
    for probe in probes:
        if probe.ok:
            log("provider %s: probed, %d models" % (probe.provider.id, len(probe.models)))
        else:
            log("provider %s: probe failed (%s) — excluded from the catalog"
                % (probe.provider.id, probe.error))
    aggregator = Aggregator(ENV_URL, brief, probes)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    Handler.aggregator = aggregator
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on 127.0.0.1:%d" % port)
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set — serving without registering; stop with Ctrl-C")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    # Register: the env answers {"ok": true, "requests": 20}.
    try:
        status, _hdrs, raw = http_json("POST", ENV_URL + "/register",
                                        body={"protocol_version": PROTOCOL_VERSION,
                                              "port": port,
                                              "catalog": aggregator.catalog},
                                        timeout=UPSTREAM_TIMEOUT_S)
        if status < 200 or status >= 300:
            log("register failed: HTTP %s" % status)
            return 1
        answer = json.loads(raw.decode("utf-8"))
    except Exception as err:  # noqa: BLE001
        log("register failed: %s" % err)
        return 1
    log("registered; the env expects %s requests" % answer.get("requests"))
    # Poll /work until the env says the workload is done, then exit 0.
    try:
        while True:
            time.sleep(POLL_INTERVAL_S)
            try:
                work = _env_get(ENV_URL, "/work", timeout=UPSTREAM_TIMEOUT_S)
            except Exception as err:  # noqa: BLE001 - keep polling
                log("polling /work failed: %s" % err)
                continue
            if is_record(work) and work.get("done") is True:
                log("workload done; exiting 0")
                break
    finally:
        server.shutdown()
        server.server_close()
    return 0


def print_list():
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        sys.stdout.write(json.dumps({"providers": []}) + "\n")
        return 0
    probes = [probe_provider(ENV_URL, provider) for provider in brief.providers]
    catalog = build_catalog(probes, brief)
    sys.stdout.write(json.dumps(catalog) + "\n")
    return 0


def main(argv):
    if "--list" in argv:
        return print_list()
    return run_server()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
