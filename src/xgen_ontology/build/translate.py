"""Name -> ASCII IRI local names, by LLM translation (the production OWL generator's rule).

A Korean (or any non-ASCII) ontology gets readable IRIs by translating each class
and property name once: classes in UpperCamelCase, properties in lowerCamelCase.
Names the model does not translate keep their own characters (IRIs allow them).
The translation map is cached by the caller (``Ontology.translations``) so a
rebuild translates only new names.
"""
from __future__ import annotations

import re

from ..llm import invoke_json
from ..models import Concepts

_BATCH = 50
_TRANSLATE_SCHEMA = {
    "type": "object", "required": ["translations"],
    "properties": {"translations": {"type": "object", "additionalProperties": {"type": "string"}}},
}
_SYSTEM = "You translate ontology terms into English identifiers."
_USER = """Translate the ontology terms below into English identifiers.
Rules:
1. A class is UpperCamelCase (e.g. 신용등급 -> CreditRating)
2. A property is lowerCamelCase (e.g. 대출한도 -> loanLimit)
3. Keep the domain meaning of a technical term exact
4. CamelCase only: no spaces, no special characters

Terms:
{terms}

Output (JSON only):
{{"translations": {{"term": "EnglishTerm", ...}}}}"""


def clean_korean_name(raw: str) -> str:
    """Strip markup and stray punctuation an extractor leaves in a name."""
    if not raw:
        return ""
    name = re.sub(r"(range|domain|type|class):", "", raw) if ":" in raw else raw
    name = re.sub(r"<[^>]*>", "", name)
    name = re.sub(r"[():/,\[\]{}<>\"'\\]", "", name)
    return name.strip()


def collect_terms(concepts: Concepts) -> list[str]:
    """Every class and property name that needs an identifier."""
    terms: dict[str, None] = {}
    for c in concepts.classes:
        n = clean_korean_name(c.name)
        if n:
            terms[n] = None
    for p in [*concepts.object_properties, *concepts.datatype_properties]:
        n = clean_korean_name(p.name)
        if n:
            terms[n] = None
    return list(terms)


def translate_names(names: list[str], llm, *, cache: dict[str, str] | None = None,
                    batch_size: int = _BATCH) -> dict[str, str]:
    """Translate ``names`` in batches; names already in ``cache`` are not sent again.

    Returns the full map (cache included). A batch the model fails to answer is
    simply left untranslated; nothing raises.
    """
    out: dict[str, str] = dict(cache or {})
    todo = [n for n in dict.fromkeys(names) if n and n not in out]
    if llm is None or not todo:
        return out
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        listing = "\n".join(f"- {t}" for t in batch)
        result = invoke_json(llm, _SYSTEM, _USER.format(terms=listing), schema=_TRANSLATE_SCHEMA)
        got = result.get("translations") if isinstance(result, dict) else None
        if isinstance(got, dict):
            for k, v in got.items():
                if isinstance(k, str) and isinstance(v, str) and k and v:
                    out[k] = v
    return out


def english_local_name(name: str, translations: dict[str, str], *, is_class: bool = True) -> str:
    """The IRI local name for ``name``: the translation in the right case, else the name's own safe characters."""
    eng = translations.get(name) or translations.get(clean_korean_name(name))
    if eng:
        eng = re.sub(r"[^a-zA-Z0-9]", "", eng)
        if eng:
            return (eng[0].upper() if is_class else eng[0].lower()) + eng[1:]
    safe = re.sub(r"[^가-힣a-zA-Z0-9]", "", name or "")
    return safe or "Unknown"
