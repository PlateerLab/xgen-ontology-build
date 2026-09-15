"""LLM adapters + a lenient JSON helper.

* ``EchoLLM`` — returns the fused evidence so the search pipeline runs end-to-end
  with no API key; build stages that expect JSON simply get nothing back and
  no-op (so the deterministic CSV path needs no LLM at all).
* ``CallableLLM`` — wrap any ``f(prompt, system) -> str`` (your OpenAI / Anthropic /
  vLLM call) and you have a drop-in LLM.

``invoke_json`` accepts whatever shape a model wraps its JSON in — leading prose,
code fences, comments, trailing commas, smart quotes, a top-level array, or a
reply cut off mid-generation by the output cap. Strict parsing would turn each of
those model quirks into "0 items extracted", making build quality depend on which
model you happen to run.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


class EchoLLM:
    """No-op LLM: echoes the evidence. Lets the pipeline run with zero credentials."""

    def generate(self, prompt: str, *, system: str = "", timeout: float | None = None) -> str:
        return prompt


class CallableLLM:
    """Adapt ``f(prompt, system=...)`` (or ``f(prompt)``) into the LLM protocol."""

    def __init__(self, fn: Callable[..., str]):
        self._fn = fn

    def generate(self, prompt: str, *, system: str = "", timeout: float | None = None) -> str:
        try:
            return self._fn(prompt, system=system)
        except TypeError:
            return self._fn(prompt)


_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def invoke_json(llm: Any, system: str, user: str, *, schema: dict | None = None) -> dict:
    """Call the LLM and parse a JSON object out of the reply, leniently.

    When ``schema`` is given and the LLM exposes ``generate_json(prompt, system=,
    schema=)`` (structured output), that is used and a dict reply is taken as is;
    otherwise ``llm.generate`` is called and the text is parsed. Returns ``{}`` on
    any failure (no LLM, non-JSON echo, parse error) so build stages degrade
    gracefully to their rule-based behavior."""
    if llm is None:
        return {}
    try:
        if schema is not None and hasattr(llm, "generate_json"):
            raw = llm.generate_json(user, system=system, schema=schema)
            if isinstance(raw, dict):
                return raw
        else:
            raw = llm.generate(user, system=system)
    except Exception:
        return {}
    if not raw or not isinstance(raw, str):
        return {}
    obj = parse_json_lenient(raw)
    return obj if isinstance(obj, dict) else {}


def parse_json_lenient(text: str) -> dict | None:
    """Accept JSON however the model wraps it.

    Order: fenced block anywhere -> direct parse -> comment/trailing-comma
    relaxation (outside string literals only) -> first balanced {...}/[...] ->
    truncation salvage. A top-level array is wrapped as ``{"entities": [...]}``
    so callers can keep using ``.get``.
    """
    if not isinstance(text, str):
        return None
    content = text.strip()
    m = _FENCE.search(content)
    if m:
        content = m.group(1).strip()

    def _try(txt: str):
        try:
            return json.loads(txt)
        except Exception:
            return None

    obj = _try(content)
    if obj is None:
        obj = _try(_relax(content))
    if obj is None:
        cut = _balanced(content)
        if cut:
            obj = _try(cut) or _try(_relax(cut))
    if obj is None:
        obj = salvage_truncated(content)
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"entities": obj}
    return None


def _relax(txt: str) -> str:
    """Strip comments and trailing commas outside string literals; fix smart quotes."""
    out, i, n, instr, esc = [], 0, len(txt), False, False
    while i < n:
        ch = txt[i]
        if instr:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            i += 1
            continue
        if ch == '"':
            instr = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and txt[i + 1] == "/":
            while i < n and txt[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and txt[i + 1] == "*":
            j = txt.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        out.append(ch)
        i += 1
    s2 = "".join(out)
    s2 = s2.replace("“", '"').replace("”", '"')
    return re.sub(r",\s*([}\]])", r"\1", s2)


def _balanced(txt: str) -> str | None:
    """First { or [ through its matching close. A greedy regex would swallow trailing prose."""
    start = min([p for p in (txt.find("{"), txt.find("[")) if p >= 0], default=-1)
    if start < 0:
        return None
    open_ch = txt[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, instr, esc = 0, False, False
    for k in range(start, len(txt)):
        c = txt[k]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return txt[start:k + 1]
    return None


def salvage_truncated(txt: str) -> Any | None:
    """Close a reply cut off mid-generation and keep the complete elements.

    On a slow model one truncated call costs minutes of decode time; discarding
    it and re-splitting spends that time again. Everything before the cut is
    valid data, so drop the unfinished trailing element, close open strings /
    arrays / objects, and parse what remains.
    """
    if not txt:
        return None
    stack, instr, esc, last_safe = [], False, False, -1
    for i, c in enumerate(txt):
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack:
                stack.pop()
        elif c == "," and len(stack) <= 2:
            last_safe = i  # shallow-depth comma = element boundary
    if not stack:
        return None
    body = txt[:last_safe] if last_safe > 0 else txt
    q = body.count('"') - body.count('\\"')
    if q % 2:
        body = body[:body.rindex('"')]
    body = body.rstrip().rstrip(",")
    # Recompute open scopes against the trimmed body (trimming may have changed depth).
    st2, instr2, esc2 = [], False, False
    for c in body:
        if instr2:
            if esc2:
                esc2 = False
            elif c == "\\":
                esc2 = True
            elif c == '"':
                instr2 = False
            continue
        if c == '"':
            instr2 = True
        elif c in "{[":
            st2.append("}" if c == "{" else "]")
        elif c in "}]":
            if st2:
                st2.pop()
    try:
        return json.loads(body + "".join(reversed(st2)))
    except Exception:
        return None
