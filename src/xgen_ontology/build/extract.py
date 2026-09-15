"""Document -> ontology extraction with an LLM.

Two modes, mirroring the production build:

* **full** (:meth:`DocumentExtractor.extract`) -- for each batch of chunks the
  model returns schema (classes / object & datatype properties / hierarchy)
  *and* instances (entities / relations / data values) in one call, tagged back
  to their source chunk ids.
* **relations only** (:meth:`DocumentExtractor.extract_relations`) -- the enrich
  pass. Entities and classes already exist (from the zero-LLM
  :mod:`.deterministic` build); the model is asked only to connect the known
  entities, choosing from the predicates already in use where possible.

Shared post-processing, all rule-based: a data value whose value is a known
entity is really a relation (and the reverse), sentence fragments returned as
entity names are shortened to the name, relation direction is voted on, and
predicates are governed to one vocabulary. A junk filter skips machine dumps
before spending a call; merging is conservative (reuse an existing class only
for the *same* concept -- synonym folding happens later in :mod:`.dedup`).

The prompt is overridable for any language / domain via ``system_prompt`` /
``user_template``; the default is English-neutral.
"""
from __future__ import annotations

import base64
import re
import unicodedata

from ..korean import is_sentence_like
from ..llm import invoke_json
from ..models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation
from .dedup import shorten_entity_name
from .deterministic import extract_from_chunk, is_value, strip_headers
from .govern import apply_canonical_predicates, govern_predicates, vote_relation_direction

_CJK = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏぀-ヿ一-鿿㐀-䶿]")

# Top-level key aliases — absorb singular/plural and non-English key spellings a
# model may use. Without this a perfectly good reply silently becomes 0 items.
KEY_ALIASES = {
    "classes": ("class", "클래스", "개념", "concepts", "concept"),
    "entities": ("entity", "개체", "인스턴스", "instances", "instance"),
    "relations": ("relation", "관계", "triples", "triple", "relationships", "relationship"),
    "data_values": ("datavalues", "datavalue", "data_value", "데이터값", "속성값", "values"),
    "object_properties": ("objectproperties", "objectproperty", "object_property", "관계속성"),
    "datatype_properties": ("datatypeproperties", "datatypeproperty", "datatype_property",
                            "data_properties", "dataproperties", "데이터속성"),
}

_REL_KNOWN_MAX = 200     # known entities listed to the relations-only prompt
_REL_PRED_MAX = 60       # known predicates listed to the relations-only prompt

# JSON-schema caps for structured output (a guard against runaway arrays, not a quota).
_SCHEMA_MAX_ITEMS = 60
_SCHEMA_MIN_ITEMS = 8
_SCHEMA_NAME_LEN = 80
_SCHEMA_DESC_LEN = 200
_SCHEMA_VALUE_LEN = 200
_SCHEMA_ID_LEN = 64
_SCHEMA_ALIAS_ID_LEN = 8
_SCHEMA_MAX_CHUNK_REFS = 8
_SCHEMA_TOK_PER_ITEM = 45
_SCHEMA_ITEM_SHARE = 0.8


def _normalize_keys(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    def norm(k: str) -> str:
        return str(k).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    lookup: dict[str, str] = {}
    for canon, aliases in KEY_ALIASES.items():
        lookup[norm(canon)] = canon
        for a in aliases:
            lookup[norm(a)] = canon
    out = dict(parsed)
    for k in list(parsed.keys()):
        canon = lookup.get(norm(k))
        if canon and canon != k and isinstance(parsed[k], list) and not out.get(canon):
            out[canon] = parsed[k]
    return out


# Korean numeric unit scales, compound units first so "천만" wins over "만". Pass as
# ``unit_scales`` to enable source-grounded magnitude verification; the default is
# neutral (off).
KO_UNIT_SCALES = {
    "조": 10 ** 12,
    "천억": 10 ** 11, "백억": 10 ** 10, "십억": 10 ** 9, "억": 10 ** 8,
    "천만": 10 ** 7, "백만": 10 ** 6, "십만": 10 ** 5, "만": 10 ** 4,
    "천": 10 ** 3, "백": 10 ** 2,
}


def _is_pow10(x: int) -> bool:
    while x % 10 == 0 and x > 1:
        x //= 10
    return x == 1


def verify_numeric_units(data_values: list, source_text: str, unit_scales: dict) -> int:
    """Re-derive magnitudes from unit notation in the source and fix off-by-scale values.

    Models frequently write "15억원" as 150000000 (one order of magnitude off);
    prompting does not fix it. So the truth is read from the source: if the
    source says "15억" and an extracted value has the same significant digits but
    a magnitude that differs by a power of ten, the value is replaced with the
    source-derived one. Different significant digits, a non-power-of-ten ratio,
    or a value that also occurs in the source as a stand-alone number ("15명")
    are left alone. Returns the number of corrected values.
    """
    if not data_values or not source_text or not unit_scales:
        return 0
    units = sorted(unit_scales, key=len, reverse=True)
    alt = "|".join(map(re.escape, units))
    unit_re = re.compile(r"(\d+(?:[.,]\d+)?)\s*(" + alt + r")")
    truths: dict[str, int] = {}
    for num, unit in unit_re.findall(source_text):
        try:
            base = float(num.replace(",", ""))
        except ValueError:
            continue
        val = int(base * unit_scales[unit])
        mant = str(int(base)) if base == int(base) else str(base).replace(".", "")
        truths.setdefault(mant.lstrip("0") or "0", val)

    fixed = 0
    for d in data_values:
        raw = re.sub(r"[,\s]", "", str(getattr(d, "value", "") or ""))
        if not raw.isdigit():
            continue
        mant = raw.rstrip("0").lstrip("0") or raw
        truth = truths.get(mant)
        if truth is None or int(raw) == truth:
            continue
        n = int(raw)
        hi, lo = max(truth, n), min(truth, n)
        if lo <= 0 or hi % lo != 0 or not _is_pow10(hi // lo):
            continue
        if re.search(rf"(?<![\d.,]){re.escape(raw)}(?!\d)\s*(?!{alt})", source_text):
            continue
        d.value = str(truth)
        fixed += 1
    return fixed


def schema_max_items(max_out_tokens: int, tok_per_item: float = 0.0) -> int:
    """How many elements one array may hold within an output-token cap."""
    if max_out_tokens <= 0:
        return _SCHEMA_MAX_ITEMS
    tpi = tok_per_item if tok_per_item and tok_per_item > 0 else _SCHEMA_TOK_PER_ITEM
    return max(_SCHEMA_MIN_ITEMS,
               min(_SCHEMA_MAX_ITEMS, int(max_out_tokens * _SCHEMA_ITEM_SHARE / tpi)))


def extraction_schema(has_chunk_ids: bool = False, max_out_tokens: int = 0,
                      tok_per_item: float = 0.0, include_relations: bool = True,
                      relations_only: bool = False) -> dict:
    """The JSON schema of an extraction reply, for models that support structured output.

    Closed objects with length caps; ``parent`` is nullable; relations cannot
    carry a literal (``predicate_type`` is fixed to ``ObjectProperty``);
    ``relations_only`` keeps just the ``relations`` array. Pass it to an LLM that
    exposes ``generate_json(prompt, system=, schema=)`` -- see
    :func:`~xgen_ontology.llm.invoke_json`.
    """
    def s(maxlen: int) -> dict:
        return {"type": "string", "maxLength": maxlen}

    chunk_prop: dict = {}
    chunk_req: list[str] = []
    if has_chunk_ids:
        id_len = _SCHEMA_ALIAS_ID_LEN if relations_only else _SCHEMA_ID_LEN
        chunk_prop = {"source_chunks": {"type": "array", "items": s(id_len),
                                        "maxItems": _SCHEMA_MAX_CHUNK_REFS}}
        chunk_req = ["source_chunks"]

    def obj(props: dict, with_chunks: bool = False) -> dict:
        p = dict(props)
        r = list(props.keys())
        if with_chunks:
            p.update(chunk_prop)
            r += chunk_req
        return {"type": "object", "properties": p, "required": r, "additionalProperties": False}

    per_arr = schema_max_items(max_out_tokens, tok_per_item)

    def arr(item: dict) -> dict:
        return {"type": "array", "items": item, "maxItems": per_arr}

    required = ["classes", "object_properties", "datatype_properties", "entities", "data_values"]
    if include_relations:
        required.insert(4, "relations")
    schema = {
        "type": "object", "additionalProperties": False, "required": required,
        "properties": {
            "classes": arr(obj({"name": s(_SCHEMA_NAME_LEN), "description": s(_SCHEMA_DESC_LEN),
                                "parent": {"type": ["string", "null"], "maxLength": _SCHEMA_NAME_LEN}})),
            "object_properties": arr(obj({"name": s(_SCHEMA_NAME_LEN), "domain": s(_SCHEMA_NAME_LEN),
                                          "range": s(_SCHEMA_NAME_LEN)})),
            "datatype_properties": arr(obj({"name": s(_SCHEMA_NAME_LEN), "display_name": s(_SCHEMA_NAME_LEN),
                                            "domain": s(_SCHEMA_NAME_LEN), "range": s(_SCHEMA_NAME_LEN)})),
            "entities": arr(obj({"entity": s(_SCHEMA_NAME_LEN), "class": s(_SCHEMA_NAME_LEN),
                                 "type": {"type": "string", "enum": ["INSTANCE"]}}, with_chunks=True)),
            "relations": arr(obj({"subject": s(_SCHEMA_NAME_LEN), "predicate": s(_SCHEMA_NAME_LEN),
                                  "object": s(_SCHEMA_NAME_LEN),
                                  "predicate_type": {"type": "string", "enum": ["ObjectProperty"]}},
                                 with_chunks=True)),
            "data_values": arr(obj({"entity": s(_SCHEMA_NAME_LEN), "property": s(_SCHEMA_NAME_LEN),
                                    "value": s(_SCHEMA_VALUE_LEN), "value_type": s(_SCHEMA_NAME_LEN)},
                                   with_chunks=True)),
        },
    }
    if not include_relations:
        schema["properties"].pop("relations", None)
    if relations_only:
        schema["properties"] = {"relations": schema["properties"]["relations"]}
        schema["required"] = ["relations"]
    return schema


_SYSTEM = (
    "You are a knowledge-graph engineer. Extract schema and instances from a document.\n"
    "Principles:\n"
    "1. A class is a recurring type/category; an instance is a concrete individual.\n"
    "2. Extract every proper noun (organization, person, law, product, ...) as an instance. "
    "An instance must single out one thing in the document; a name that does not distinguish "
    "among several of the same kind is a class, not an instance.\n"
    "3. Extract every figure/date/ratio as a data value — losing numbers loses knowledge. "
    "Write magnitudes out in full (no unit shorthand, no percent/currency suffixes).\n"
    "4. Do not make generic words ('document', 'information', 'data') into classes.\n"
    "{p5}"
    "6. Never lump distinct concepts into one class; concrete individuals are instances, not classes.\n"
    "7. A class name can never equal an instance name: a proper noun is the instance, the category "
    "it belongs to is the class.\n"
    "{p9}"
)
_P5_RELATIONS = (
    "5. Extract every relation (subject-predicate-object) stated in the text, including links "
    "between a table's row entity and its column items. Never invent a relation the document does "
    "not support. The subject is the side that does or has, the object the side that receives.\n"
    "5-2. 'A is a B' (type/role) is not a relation but the instance's type: express it as the "
    "entity's class, not in relations.\n"
)
_P9_RELATIONS = (
    "9. Predicate names (relations.predicate, object_properties.name) are short noun phrases of "
    "1-3 words; no sentence-like or inflected forms; one name per meaning.\n"
)
_P9_NO_RELATIONS = (
    "9. object_properties.name values are short noun phrases of 1-3 words; no sentence-like or "
    "inflected forms.\n"
)

_USER_TEMPLATE = """Extract an ontology schema and instances from the document.

## Domain: {domain}
## Document: {doc}
{schema_context}

## Document content
{text}
{source_instruction}
## Output (JSON only)
{{
  "classes": [{{"name": "...", "description": "...", "parent": "parent class or null"}}],
  "object_properties": [{{"name": "...", "domain": "...", "range": "..."}}],
  "datatype_properties": [{{"name": "...", "display_name": "...", "domain": "...", "range": "xsd:string|xsd:integer|xsd:decimal|xsd:date|xsd:boolean"}}],
  "entities": [{{"entity": "...", "class": "...", "type": "INSTANCE"{src_field}}}],{rel_example}
  "data_values": [{{"entity": "...", "property": "...", "value": "...", "value_type": "xsd:string"{src_field}}}]
}}"""
_REL_EXAMPLE = '\n  "relations": [{{"subject": "...", "predicate": "...", "object": "...", "predicate_type": "ObjectProperty"{src_field}}}],'

_RELATIONS_SYSTEM = (
    "You are a knowledge-graph engineer. Extract only the relations between entities that "
    "were already extracted.\n"
    "Principles:\n"
    "1. Extract every relation (subject-predicate-object), including links between a table's "
    "row entity and its column items.\n"
    "2. Never invent a relation the document does not support. No guessing.\n"
    "3. Spell subject and object exactly as in the known-entity list where possible; an entity "
    "named in the document but missing from the list may still be used.\n"
    "4. Predicates are short noun phrases of 1-3 words; no sentence-like or inflected forms; one "
    "name per meaning.\n"
    "5. When a list of known predicates is given, choose from it; coin a new predicate only when "
    "none of them can express the relation.\n"
    "6. Do not extract classes, properties or numeric values. Relations only."
)
_RELATIONS_USER_TEMPLATE = """Extract the relations between entities in the document.
{known_block}{pred_block}
{text}
{source_instruction}
One relation is one element of the "relations" array; the predicate is the actual relation
between the two entities as the document states it.
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...", "predicate_type": "ObjectProperty"{src_field}}}]}}"""


class DocumentExtractor:
    """Extract a typed ontology from chunked documents using an LLM."""

    def __init__(self, llm, *, domain: str = "", max_text_len: int = 10000,
                 system_prompt: str = _SYSTEM, user_template: str = _USER_TEMPLATE,
                 unit_scales: dict | None = None, self_typed_ratio: float = 0.5,
                 split_depth: int = 2, header_patterns=(), include_relations: bool = True,
                 use_schema: bool = True):
        self.llm = llm
        self.domain = domain
        self.max_text_len = max_text_len
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.unit_scales = unit_scales
        self.self_typed_ratio = self_typed_ratio
        self.split_depth = split_depth
        self.header_patterns = header_patterns
        self.include_relations = include_relations
        self.use_schema = use_schema
        self.llm_calls = 0

    # ── full extraction ──

    def extract(
        self, documents: dict[str, list[dict]], *,
        existing: Concepts | None = None, schema_context: str = "",
    ) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
        """Schema + instances from every document. ``existing`` classes are offered for reuse."""
        concepts = Concepts()
        instances: list[Instance] = []
        relations: list[Relation] = []
        data_values: list[DataValue] = []
        for doc, chunks in documents.items():
            for dc, di, dr, dv in self._run_batches(doc, chunks, existing, schema_context):
                _merge_concepts(concepts, dc)
                instances.extend(di)
                relations.extend(dr)
                data_values.extend(dv)
        self._postprocess(concepts, instances, relations, data_values)
        return concepts, instances, relations, data_values

    def extract_relations(
        self, documents: dict[str, list[dict]], *,
        known_entities: list[str] | None = None,
        known_predicates: list[str] | None = None,
    ) -> list[Relation]:
        """The enrich pass: relations only, between entities that already exist.

        Each batch is shown the entities the zero-LLM extractor finds in its own
        chunks first, topped up from ``known_entities``, and the predicates
        already in use (``known_predicates``). Direction vote and predicate
        governance run over the result.
        """
        relations: list[Relation] = []
        for doc, chunks in documents.items():
            for _c, _i, dr, _v in self._run_batches(doc, chunks, None, "", relations_only=True,
                                                     known_entities=known_entities,
                                                     known_predicates=known_predicates):
                relations.extend(dr)
        vote_relation_direction(relations, {})
        govern_predicates(relations, None, known_predicates)
        return relations

    # ── batching ──

    def _run_batches(self, doc, chunks, existing, schema_context, *, relations_only=False,
                     known_entities=None, known_predicates=None):
        chunks = _split_oversized_chunks(chunks, self.max_text_len)
        pending = [(b, 0) for b in reversed(self._batches(chunks))]
        while pending:
            batch, depth = pending.pop()
            r = self._extract_batch(doc, batch, existing, schema_context, relations_only=relations_only,
                                    known_entities=known_entities, known_predicates=known_predicates)
            if r == "junk":
                continue
            if r is None:
                # LLM/parse failure. Dropping the batch silently loses data: halve it and
                # retry; a smaller reply survives output caps and model quirks.
                if len(batch) > 1 and depth < self.split_depth:
                    mid = len(batch) // 2
                    pending.append((batch[mid:], depth + 1))
                    pending.append((batch[:mid], depth + 1))
                continue
            yield r

    def _batches(self, chunks: list[dict]) -> list[list[dict]]:
        batches: list[list[dict]] = []
        cur: list[dict] = []
        cur_len = 0
        for ch in sorted(chunks, key=lambda c: c.get("chunk_index", 0)):
            tl = len(ch.get("chunk_text", ""))
            if cur and cur_len + tl > self.max_text_len:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(ch)
            cur_len += tl
        if cur:
            batches.append(cur)
        return batches

    def _extract_batch(self, doc, batch, existing, schema_context, *, relations_only=False,
                       known_entities=None, known_predicates=None):
        chunk_ids = [c.get("chunk_id", "") for c in batch]
        alias_of, real_of = {}, {}
        if relations_only:
            # Short aliases (c0, c1, ...) keep the reply small; unknown ids are never kept.
            for i, cid in enumerate(chunk_ids):
                if cid:
                    alias_of[cid] = "c%d" % i
                    real_of["c%d" % i] = cid
        parts = []
        for c in batch:
            cid, text = c.get("chunk_id", ""), strip_headers(c.get("chunk_text", ""), self.header_patterns)
            parts.append(f"[CHUNK:{alias_of.get(cid, cid)}]\n{text}" if cid else text)
        combined = "\n\n".join(parts)
        if not combined.strip() or not _is_extractable(combined):
            return "junk"
        text = combined[:16000]
        valid = [alias_of.get(cid, cid) for cid in chunk_ids if cid]
        fallback = [cid for cid in chunk_ids if cid]
        src_instruction, src_field = "", ""
        if valid:
            src_instruction = (f"\n- source_chunks: record the chunk id each item came from "
                               f"(see the [CHUNK:id] markers). Available: {valid}\n")
            src_field = ', "source_chunks": ["chunk_id"]'

        if relations_only:
            return self._relations_batch(doc, batch, text, valid, fallback, real_of, src_instruction,
                                         src_field, known_entities, known_predicates)

        context = ""
        if existing is not None and existing.classes:
            names = [c.name for c in existing.classes if c.name][:40]
            if names:
                context = ("\n\n## Classes already extracted (for reference)\n" + ", ".join(names) +
                           "\n-> Reuse a name above only for the *same* concept; a different concept "
                           "gets its own new class.")
        if schema_context:
            context += f"\n\n{schema_context}"
        system = self.system_prompt.format(
            p5=_P5_RELATIONS if self.include_relations else "",
            p9=_P9_RELATIONS if self.include_relations else _P9_NO_RELATIONS)
        user = self.user_template.format(
            domain=self.domain or "auto-detect", doc=doc, schema_context=context, text=text,
            source_instruction=src_instruction, src_field=src_field,
            rel_example=_REL_EXAMPLE.format(src_field=src_field) if self.include_relations else "")
        schema = (extraction_schema(bool(valid), include_relations=self.include_relations)
                  if self.use_schema else None)
        result = invoke_json(self.llm, system, user, schema=schema)
        self.llm_calls += 1
        if not result:
            return None
        result = _normalize_keys(result)
        result = self._repair_self_typed(result)

        classes = [Class(name=c.get("name", ""), description=c.get("description", ""),
                         parent=c.get("parent") or None, source_chunks=c.get("source_chunks", fallback))
                   for c in result.get("classes", []) if c.get("name")]
        class_names = {c.name for c in classes}
        prop_names = ({p.get("name") for p in result.get("object_properties", []) if p.get("name")}
                      | {p.get("name") for p in result.get("datatype_properties", []) if p.get("name")})
        hierarchy: list[tuple[str, str]] = []
        for c in result.get("classes", []):
            name, parent = c.get("name"), c.get("parent")
            if not name or not parent or parent == name:
                continue
            if parent in prop_names and parent not in class_names:
                continue  # a property mislabeled as a parent -> not is-a
            hierarchy.append((parent, name))
        concepts = Concepts(
            classes=classes,
            object_properties=[ObjectProperty(name=p.get("name", ""), domain=p.get("domain", ""),
                                              range=p.get("range", ""))
                               for p in result.get("object_properties", []) if p.get("name")],
            datatype_properties=[DataProperty(name=p.get("name", ""), display_name=p.get("display_name", ""),
                                              domain=p.get("domain", ""), range=p.get("range", "xsd:string"))
                                 for p in result.get("datatype_properties", []) if p.get("name")],
            class_hierarchy=hierarchy,
        )
        # A name with no letter in it (a bare number, a date) is a value, not an entity.
        instances = [Instance(name=e.get("entity", ""), class_name=e.get("class", ""),
                              source_chunks=e.get("source_chunks") or fallback)
                     for e in result.get("entities", [])
                     if e.get("entity") and any(ch.isalpha() for ch in str(e["entity"]))]
        relations = [Relation(subject=r.get("subject", ""), predicate=r.get("predicate", ""),
                              object=r.get("object", ""), predicate_type=r.get("predicate_type", "ObjectProperty"),
                              source_chunks=r.get("source_chunks") or fallback)
                     for r in result.get("relations", []) if r.get("subject") and r.get("predicate")]
        data_values = [DataValue(entity=d.get("entity", ""), property=d.get("property", ""),
                                 value=d.get("value", ""), value_type=d.get("value_type", "xsd:string"),
                                 source_chunks=d.get("source_chunks") or fallback)
                       for d in result.get("data_values", []) if d.get("entity") and d.get("property")]
        if self.unit_scales:
            verify_numeric_units(data_values, combined, self.unit_scales)
        return concepts, instances, relations, data_values

    def _relations_batch(self, doc, batch, text, valid, fallback, real_of, src_instruction,
                         src_field, known_entities, known_predicates):
        local: list[str] = []
        seen_local: set[str] = set()
        try:
            for c in batch:
                for n, _cls in extract_from_chunk(c.get("chunk_text") or "", self.header_patterns)[0]:
                    if n not in seen_local:
                        seen_local.add(n)
                        local.append(n)
        except Exception:
            local = []
        known = local[:_REL_KNOWN_MAX]
        if len(known) < _REL_KNOWN_MAX:
            have = set(known)
            known += [e for e in (known_entities or []) if e and e not in have][:_REL_KNOWN_MAX - len(known)]
        known_block = ("\n## Known entities (connect these)\n" + ", ".join(known)) if known else ""
        preds = [q for q in (known_predicates or []) if q][:_REL_PRED_MAX]
        pred_block = ("\n## Known predicates (choose from these where possible)\n" + ", ".join(preds)) if preds else ""
        user = _RELATIONS_USER_TEMPLATE.format(known_block=known_block, pred_block=pred_block, text=text,
                                               source_instruction=src_instruction, src_field=src_field)
        schema = extraction_schema(bool(valid), relations_only=True) if self.use_schema else None
        result = invoke_json(self.llm, _RELATIONS_SYSTEM, user, schema=schema)
        self.llm_calls += 1
        if not result:
            return None
        result = _normalize_keys(result)

        def unalias(vals):
            out = [real_of[v] for v in (vals or []) if v in real_of]
            return out or fallback

        relations = [Relation(subject=r.get("subject", ""), predicate=r.get("predicate", ""),
                              object=r.get("object", ""), source_chunks=unalias(r.get("source_chunks")))
                     for r in result.get("relations", []) if r.get("subject") and r.get("predicate")]
        return Concepts(), [], relations, []

    # ── post-processing (rule-based) ──

    def _postprocess(self, concepts, instances, relations, data_values) -> None:
        def nk(s) -> str:
            return " ".join(unicodedata.normalize("NFKC", str(s or "")).split())

        ent_names = {nk(i.name) for i in instances if i.name}
        ent_names.discard("")
        # A data value whose value is a known entity is a relation.
        if ent_names and data_values:
            kept = []
            for d in data_values:
                v = nk(d.value)
                if v and v in ent_names and nk(d.entity) != v:
                    relations.append(Relation(subject=d.entity, predicate=d.property, object=str(d.value),
                                              source_chunks=list(d.source_chunks)))
                else:
                    kept.append(d)
            data_values[:] = kept
        # ... and a relation whose object is a value, not an entity, is a data value.
        if relations:
            kept_r = []
            for r in relations:
                o = nk(r.object)
                if o and o not in ent_names and is_value(o):
                    data_values.append(DataValue(entity=r.subject, property=r.predicate, value=r.object,
                                                 source_chunks=list(r.source_chunks)))
                else:
                    kept_r.append(r)
            relations[:] = kept_r
        # Sentence fragments returned as entity names are shortened to the name.
        renamed: dict[str, str] = {}
        for i in instances:
            if i.name and is_sentence_like(i.name):
                short = shorten_entity_name(i.name)
                if short and short != i.name:
                    renamed[nk(i.name)] = short
                    i.name = short
        if renamed:
            for r in relations:
                r.subject = renamed.get(nk(r.subject), r.subject)
                r.object = renamed.get(nk(r.object), r.object)
            for d in data_values:
                d.entity = renamed.get(nk(d.entity), d.entity)
        ent_class: dict[str, str] = {}
        for i in instances:
            if i.name and i.class_name:
                ent_class.setdefault(nk(i.name), i.class_name)
        vote_relation_direction(relations, ent_class)
        stats = govern_predicates(relations, concepts.object_properties)
        canon = stats.get("canonical_map") or {}
        if canon:
            # The same vocabulary on declarations and data values, so a predicate does not
            # split by batch or model spelling.
            names = apply_canonical_predicates(canon, [op.name for op in concepts.object_properties])
            for op, n in zip(concepts.object_properties, names):
                op.name = n
            names = apply_canonical_predicates(canon, [d.property for d in data_values])
            for d, n in zip(data_values, names):
                d.property = n

    def _repair_self_typed(self, result: dict) -> dict:
        """Fix entities typed as themselves (entity == class), one extra call at most.

        Promoting proper nouns to classes happens on every model family and
        flattens the schema (one private class per instance). Inventing hypernyms
        locally would create unsupported classes, so the wrong list is handed
        back to the model for re-typing. Fires only when the self-typed share of
        a batch crosses ``self_typed_ratio``; well-behaved replies cost nothing.
        """
        ents = result.get("entities") or []
        selfref = [e for e in ents
                   if isinstance(e, dict) and e.get("entity") and e.get("entity") == e.get("class")]
        if not selfref or len(selfref) < max(1, int(len(ents) * self.self_typed_ratio)):
            return result
        listing = "\n".join("- " + e["entity"] for e in selfref)
        fixed = invoke_json(
            self.llm,
            "You are an ontology engineer. Assign each entity its proper abstract type (class).",
            "The entities below were wrongly typed as themselves. Assign each a type.\n\n"
            "Rules: the type must differ from the entity name and be a category in the document's\n"
            "domain. Use the category a proper noun belongs to, not the noun itself. Several\n"
            "entities may share one type.\n\n" + listing + "\n\n"
            '{"types": [{"entity": "name", "class": "type"}]}',
        )
        self.llm_calls += 1
        mapping = {}
        for row in (fixed.get("types") or []):
            if isinstance(row, dict) and row.get("entity") and row.get("class"):
                if row["class"] != row["entity"]:
                    mapping[row["entity"]] = row["class"]
        if not mapping:
            return result
        for e in ents:
            if isinstance(e, dict) and e.get("entity") in mapping:
                e["class"] = mapping[e["entity"]]
        dropped = set(mapping)
        kept = [c for c in (result.get("classes") or [])
                if not (isinstance(c, dict) and c.get("name") in dropped)]
        have = {c.get("name") for c in kept if isinstance(c, dict)}
        for cls in dict.fromkeys(mapping.values()):
            if cls not in have:
                kept.append({"name": cls, "description": "", "parent": None})
        result["classes"] = kept
        return result


def _merge_concepts(into: Concepts, new: Concepts) -> None:
    """Merge a batch's schema into the running one; a repeated class only gains source chunks."""
    by_name = {c.name: c for c in into.classes if c.name}
    for c in new.classes:
        if not c.name:
            continue
        if c.name in by_name:
            have = by_name[c.name]
            for cid in c.source_chunks or []:
                if cid not in have.source_chunks:
                    have.source_chunks.append(cid)
        else:
            into.classes.append(c)
            by_name[c.name] = c
    seen_op = {p.name for p in into.object_properties}
    for op in new.object_properties:
        if op.name and op.name not in seen_op:
            into.object_properties.append(op)
            seen_op.add(op.name)
    seen_dp = {p.name for p in into.datatype_properties}
    for dp in new.datatype_properties:
        if dp.name and dp.name not in seen_dp:
            into.datatype_properties.append(dp)
            seen_dp.add(dp.name)
    seen_h = set(into.class_hierarchy)
    for edge in new.class_hierarchy:
        if edge not in seen_h:
            into.class_hierarchy.append(edge)
            seen_h.add(edge)


def _split_oversized_chunks(chunks: list[dict], max_chars: int) -> list[dict]:
    """Split a single chunk larger than the budget (a huge table): by ``</tr>`` rows when it is one, else by character window."""
    out: list[dict] = []
    for c in chunks:
        text = c.get("chunk_text", "") or ""
        if len(text) <= max_chars:
            out.append(c)
            continue
        for part in _split_text(text, max_chars):
            nc = dict(c)
            nc["chunk_text"] = part
            out.append(nc)
    return out


def _split_text(text: str, max_chars: int) -> list[str]:
    def window(s: str) -> list[str]:
        return [s[i:i + max_chars] for i in range(0, len(s), max_chars)] or [s]

    if "</tr>" not in text:
        return window(text)
    parts: list[str] = []
    cur = ""
    for row in re.split(r"(?<=</tr>)", text):
        if cur and len(cur) + len(row) > max_chars:
            parts.append(cur)
            cur = ""
        cur += row
    if cur:
        parts.append(cur)
    final: list[str] = []
    for p in parts:
        final.extend([p] if len(p) <= max_chars else window(p))
    return final or window(text)


def _looks_base64(s: str) -> bool:
    if len(s) < 100 or re.search(r"\s", s):
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s) or len(s) % 4 != 0:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def _is_extractable(text: str) -> bool:
    """False only for obvious machine junk; numbers/tables/short text/CJK pass."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _CJK.search(stripped):
        return True
    if len(stripped) >= 100 and not re.search(r"\s", stripped):
        uniq = len(set(stripped))
        if uniq >= 16 and _looks_base64(stripped):
            return False
        if uniq <= 8:
            return False
    if not any(c.isalnum() for c in stripped):
        return False
    return True
