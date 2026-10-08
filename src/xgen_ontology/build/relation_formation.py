"""Relation formation: name the relations between the nodes a build already has.

Ported from the production relation pass (xgen-documents, the relation classifier of the
"full" and "enrich" builds). The basic build gives nodes; this pass asks a model what the
document says between two of them, and only that:

1. documents are cut into units of evidence (a sentence or clause, a table row) and the
   nodes each unit mentions are numbered (:mod:`.relation_units`, deterministic); units
   with fewer than two mentions are never sent
2. units are batched by an input budget and by the expected size of the answer (units x
   relations per unit, learnt as calls come back, against the answer's array cap)
3. relation names come from a **vocabulary** (name + definition). With none, one is
   defined first from units sampled evenly across the corpus
4. each batch is answered with numbers: ``{"u": unit, "s": subject, "p": predicate,
   "o": object}``. A cut-off answer keeps its complete part and the batch is halved; an
   empty answer is retried once after a pause; an answer with no relation is asked once
   more (models drop whole batches now and then). A row that does not hold together (a
   number out of range, a subject that is its object, a predicate that is not an English
   identifier or is the graph's own vocabulary) is dropped
5. a relation keeps the chunk that states it (its source) and, for a table row, the
   object cell's column name as its label
6. names used often enough outside the vocabulary (a natural break in their frequencies,
   no threshold) are defined and added to it
7. at the end of a build, :func:`canonicalize_predicates` holds every name to the
   vocabulary: a name the model maps to a vocabulary name takes it, frequent unmapped
   names join the vocabulary, the rest become ``relatedTo`` with the old name as label

Model specifics stay outside: pass the output cap, the input budget and the timeout the
model supports; the lenient JSON reader flags a cut-off answer for any model, and an LLM
may report it exactly (:func:`~xgen_ontology.llm.invoke_json_meta`). The prompts are a
:class:`RelationPrompts`; :data:`KO_RELATION_PROMPTS` are the production prompts verbatim.
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..llm import invoke_json_meta
from ..models import Concepts, ObjectProperty, Relation
from .deterministic import RELATED_PREDICATE
from .extract import _SCHEMA_DESC_LEN, _SCHEMA_NAME_LEN, _SCHEMA_TOK_PER_ITEM, schema_max_items
from .finalize import GRAPH_VOCABULARY, RESERVED_PREDICATE_NAMES
from .govern import govern_predicates
from .relation_units import Unit, relation_label, units_for_documents

# A relation name is one English identifier: no spaces, no Hangul, no punctuation.
PREDICATE_PATTERN = r"^[A-Za-z][A-Za-z0-9]*$"
_PRED_RE = re.compile(PREDICATE_PATTERN)
# "unit subject predicate object" written as one line instead of an object.
_REL_LINE = re.compile(r"^\s*(\d+)[\s,]+(\d+)[\s,]+([A-Za-z][A-Za-z0-9_\-]*)[\s,]+(\d+)\s*$")


@dataclass(frozen=True)
class RelationPrompts:
    """The words relation formation shows a model. Layout and rules are the production ones."""

    classify_system: str
    vocabulary_system: str
    canonicalize_system: str
    extend_system: str
    graph_relations: tuple[tuple[str, str], ...]   # the graph's own relations, with definitions
    vocabulary_heading: str
    list_heading: str
    graph_heading: str
    domain_line: str                               # "{domain}" is filled in
    auto_domain: str
    units_heading: str
    entities_prefix: str
    missing_heading: str


KO_RELATION_PROMPTS = RelationPrompts(
    classify_system="""당신은 문서의 개체 사이 관계 이름을 정하는 분류기입니다.
단위마다 글 한 조각과 그 안에 나온 개체의 번호 목록이 있습니다.

규칙:
1. 그 단위의 글이 명시한 관계만 적는다. 같은 단위 안의 두 개체 사이에서만. 추측 금지.
2. 한 쌍에는 가장 구체적인 관계 하나만.
3. 술어 p 는 그 쌍에 대해 글이 말하는 관계를 가장 구체적으로 나타내는 영어 camelCase 동사 하나다.
   같은 뜻의 이름이 '관계 이름 목록'에 있으면 그 이름을 그대로 쓴다. 두 개체가 함께 나왔다는 것 외에 아무 뜻이 없는 이름은 쓰지 않는다.
4. 주어 s 는 관계를 행하거나 가지는 쪽, 목적어 o 는 받는 쪽이다. u 는 단위 번호다.
5. 관계가 없는 단위는 아무것도 적지 않는다.

출력: {"units_with_relations": [관계가 명시된 단위 번호, ...], "relations": [{"u": 단위번호, "s": 주어번호, "p": 술어, "o": 목적어번호}, ...]}
먼저 단위를 하나씩 훑어 관계가 명시된 단위 번호를 모두 적고, 그다음 그 단위들의 관계를 적는다.""",
    vocabulary_system="""당신은 문서의 개체 사이 관계에 이름을 붙이는 온톨로지 설계자입니다.
단위마다 글 한 조각과 그 안에 나온 개체 목록이 있습니다. 이 글들이 말하는 개체 사이 관계의 이름 목록을 만드세요.
'그래프가 이미 가진 관계' 는 그래프가 스스로 쓰는 관계라 목록에 넣지 않습니다.

규칙:
1. 문서가 명시하는 관계만. 추측 금지.
2. 하나의 뜻에 이름 하나. 뜻이 같은 관계를 둘로 나누지 않는다.
3. 이름은 영문자로 시작하는 영어 camelCase 한 단어다. 한글·공백 금지. 주어나 목적어의 이름을 술어에 넣지 않는다.
4. 어떤 쌍에나 붙일 수 있어 뜻이 없는 말, '그래프가 이미 가진 관계' 와 뜻이 같은 말은 관계 이름이 아니다.
5. 한쪽이 다른 쪽에 하는 행위 관계뿐 아니라 부분 · 적용 · 근거 · 정의 · 순서 같은 구조 관계도 넣는다.
6. definition 은 그 관계가 무엇을 뜻하는지, 주어와 목적어가 보통 무엇인지 한 줄로 적는다.

출력: {"predicates": [{"name": 관계이름, "definition": 한 줄 정의}, ...]}""",
    canonicalize_system="""당신은 온톨로지 설계자입니다. 관계 이름 목록(정의 포함)이 있고, 그 목록에 없는 이름들이 예시와 함께 주어집니다.
없는 이름마다 뜻이 같은 목록의 이름을 고르세요. 뜻이 같은 이름이 없으면 target 을 빈 문자열로 둡니다.

규칙:
1. 예시의 주어 · 목적어를 보고 실제로 쓰인 뜻으로 판단한다.
2. 비슷해 보여도 뜻이 다르면 묶지 않는다.

출력: {"mapping": [{"name": 없는이름, "target": 목록이름 또는 ""}, ...]}""",
    extend_system="""당신은 온톨로지 설계자입니다. 관계 이름 목록(정의 포함)이 있고, 목록에 없는데 문서에서 자주 쓰인 이름들이 예시와 함께 주어집니다.
그 이름들을 목록에 더할 항목으로 정리하세요.

규칙:
1. 예시의 주어 · 목적어로 실제 뜻을 판단해 definition 을 한 줄로 적는다.
2. 뜻이 같은 이름들은 하나로 합쳐 대표 이름 하나만 낸다. 기존 목록이나 '그래프가 이미 가진 관계' 와 뜻이 같으면 내지 않는다.
3. 이름은 영문자로 시작하는 영어 camelCase 한 단어다. 어떤 쌍에나 붙일 수 있는 이름은 내지 않는다.

출력: {"predicates": [{"name": 관계이름, "definition": 한 줄 정의}, ...]}""",
    graph_relations=(
        ("relatedTo", "두 개체가 함께 나왔다는 것 외에 뜻을 정하지 않은 관계"),
        ("sameAs", "두 이름이 같은 대상을 가리킨다"),
        ("instanceOf", "개체가 그 클래스에 속한다"),
        ("subClassOf", "클래스가 더 넓은 클래스에 속한다"),
    ),
    vocabulary_heading="## 관계 이름 목록 (같은 뜻이면 이 이름을 쓴다)",
    list_heading="## 관계 이름 목록",
    graph_heading="## 그래프가 이미 가진 관계",
    domain_line="## 도메인: {domain}",
    auto_domain="자동 판별",
    units_heading="## 단위",
    entities_prefix="  개체: ",
    missing_heading="## 목록에 없는 이름과 예시",
)

EN_RELATION_PROMPTS = RelationPrompts(
    classify_system="""You name the relations between the entities of a document.
Each unit holds one piece of text and the numbered entities that appear in it.

Rules:
1. Write only relations the unit's text states, and only between two entities of the same unit. No guessing.
2. One relation per pair: the most specific one.
3. The predicate p is one English camelCase verb that most specifically says what the text states about the pair.
   When the 'relation names' list has a name with the same meaning, use that name as it is. Never use a name that says nothing beyond the two entities appearing together.
4. The subject s is the side that does or has the relation, the object o the side that receives it. u is the unit number.
5. Write nothing for a unit that states no relation.

Output: {"units_with_relations": [numbers of the units that state a relation, ...], "relations": [{"u": unit number, "s": subject number, "p": predicate, "o": object number}, ...]}
First go over the units one by one and list every unit that states a relation, then write those units' relations.""",
    vocabulary_system="""You design an ontology by naming the relations between the entities of documents.
Each unit holds one piece of text and the entities that appear in it. Make the list of names for the relations these texts state between entities.
'Relations the graph already has' are the graph's own relations; do not put them in the list.

Rules:
1. Only relations the documents state. No guessing.
2. One name per meaning. Do not split one meaning into two relations.
3. A name is one English camelCase word starting with a letter. No spaces, no other scripts. Do not put the subject's or the object's name into the predicate.
4. A word that fits any pair and so means nothing, or one that means the same as a 'relation the graph already has', is not a relation name.
5. Include structural relations (part, application, basis, definition, order) as well as actions one side takes on the other.
6. The definition says in one line what the relation means and what its subject and object usually are.

Output: {"predicates": [{"name": relation name, "definition": one-line definition}, ...]}""",
    canonicalize_system="""You design an ontology. There is a list of relation names (with definitions), and names that are not in it are given with examples.
For each missing name choose the listed name with the same meaning. When none has the same meaning, leave target empty.

Rules:
1. Judge by the meaning actually used, from the examples' subjects and objects.
2. Do not group names that look alike but mean different things.

Output: {"mapping": [{"name": missing name, "target": listed name or ""}, ...]}""",
    extend_system="""You design an ontology. There is a list of relation names (with definitions), and names used often in the documents but missing from it are given with examples.
Turn those names into entries to add to the list.

Rules:
1. Judge the actual meaning from the examples' subjects and objects and write the definition in one line.
2. Merge names with the same meaning into one representative name. Leave out a name that means the same as one in the list or a 'relation the graph already has'.
3. A name is one English camelCase word starting with a letter. Leave out a name that fits any pair.

Output: {"predicates": [{"name": relation name, "definition": one-line definition}, ...]}""",
    graph_relations=(
        ("relatedTo", "the two entities appear together; no further meaning is given"),
        ("sameAs", "the two names refer to the same thing"),
        ("instanceOf", "the entity belongs to the class"),
        ("subClassOf", "the class belongs to a broader class"),
    ),
    vocabulary_heading="## Relation names (use the name when the meaning is the same)",
    list_heading="## Relation names",
    graph_heading="## Relations the graph already has",
    domain_line="## Domain: {domain}",
    auto_domain="detect automatically",
    units_heading="## Units",
    entities_prefix="  entities: ",
    missing_heading="## Names missing from the list, with examples",
)


def natural_cut(scored: list[tuple], *, min_k: int = 0, max_k: int = 0) -> list:
    """Self-sizing top-k: cut where the scores split into a high and a low group.

    Picks the k that maximizes the between-group variance of head ``scored[:k]`` vs tail
    (Jenks natural breaks / Otsu with two classes), so the data decides how many items
    survive; there is no ratio or threshold to tune. ``min_k`` / ``max_k`` only bound
    the answer (0 = unbounded). Flat scores have no break and are kept whole.
    ``scored`` is [(item, score), ...] sorted desc; returns the items. Same function as
    ``omnifuse.fusion.natural_cut`` (the build does not import the search package).
    """
    if not scored:
        return []
    n = len(scored) if max_k <= 0 else min(len(scored), max_k)
    lo = max(1, min_k)
    if n <= lo:
        return [it for it, _ in scored[:n]]
    vals = [float(sc) for _, sc in scored[:n]]
    prefix = [0.0]
    for v in vals:
        prefix.append(prefix[-1] + v)
    total = prefix[-1]
    best, best_k = 0.0, n
    for k in range(lo, n):
        head, tail = prefix[k] / k, (total - prefix[k]) / (n - k)
        var = k * (n - k) * (head - tail) ** 2
        if var > best:
            best, best_k = var, k
    return [it for it, _ in scored[:best_k]]


def vocab_name(name: Any) -> str:
    """The name if it can be a relation-vocabulary name: an English identifier that is not the
    graph's own vocabulary or one of its aliases (relatedTo, instanceOf, isA ...); else ``""``."""
    n = str(name or "").strip()
    return n if _PRED_RE.match(n) and n.lower() not in RESERVED_PREDICATE_NAMES else ""


def vocabulary_of(concepts: Concepts) -> list[dict[str, str]]:
    """The relation vocabulary an ontology holds: its object properties named as vocabulary
    names, with their descriptions as definitions."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for op in concepts.object_properties:
        name = vocab_name(op.name)
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name, "definition": op.description or ""})
    return out


def relation_schema(max_out_tokens: int, tok_per_item: float, n_units: int = 0) -> dict[str, Any]:
    """The answer of one classification call: numbers for unit / subject / object and an English
    identifier for the predicate. The list of units that state a relation comes first: a model
    asked for the array alone empties whole batches now and then, and scanning the units first
    stops that; a count instead of numbers would make it write as many relations as it counted."""
    pred: dict[str, Any] = {"type": "string", "pattern": PREDICATE_PATTERN, "maxLength": _SCHEMA_NAME_LEN}
    item = {
        "type": "object", "additionalProperties": False, "required": ["u", "s", "p", "o"],
        "properties": {"u": {"type": "integer"}, "s": {"type": "integer"}, "p": pred, "o": {"type": "integer"}},
    }
    scan: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
    if n_units > 0:
        scan["maxItems"] = n_units
    return {
        "type": "object", "additionalProperties": False, "required": ["units_with_relations", "relations"],
        "properties": {"units_with_relations": scan,
                       "relations": {"type": "array", "maxItems": schema_max_items(max_out_tokens, tok_per_item),
                                     "items": item}},
    }


def vocabulary_schema(max_out_tokens: int, tok_per_item: float) -> dict[str, Any]:
    """The answer that defines relation names."""
    return {
        "type": "object", "additionalProperties": False, "required": ["predicates"],
        "properties": {"predicates": {
            "type": "array", "maxItems": schema_max_items(max_out_tokens, tok_per_item),
            "items": {"type": "object", "additionalProperties": False, "required": ["name", "definition"],
                      "properties": {"name": {"type": "string", "pattern": PREDICATE_PATTERN,
                                              "maxLength": _SCHEMA_NAME_LEN},
                                     "definition": {"type": "string", "maxLength": _SCHEMA_DESC_LEN}}}}},
    }


def canon_schema(names: Sequence[str], count: int) -> dict[str, Any]:
    """The answer that maps names outside the vocabulary onto it."""
    return {
        "type": "object", "additionalProperties": False, "required": ["mapping"],
        "properties": {"mapping": {
            "type": "array", "maxItems": max(1, count),
            "items": {"type": "object", "additionalProperties": False, "required": ["name", "target"],
                      "properties": {"name": {"type": "string", "maxLength": _SCHEMA_NAME_LEN},
                                     "target": {"type": "string", "enum": list(names) + [""]}}}}},
    }


class RelationFormer:
    """Names the relations between known nodes, unit by unit, with a model.

    ``llm`` follows the library's LLM protocol. ``char_budget`` is the input size of one
    call (the production default when the model's window is unknown), ``max_output_tokens``
    the answer cap (0: unknown, the array cap is then the schema maximum), ``timeout`` one
    call's limit. ``max_workers`` > 1 sends batches in parallel threads. ``should_stop`` is
    asked before every call: once it says so, nothing more is sent and the chunks not
    answered go to :attr:`lost_chunk_ids`. ``progress(done, total)`` is told each batch.

    After :meth:`extract`, :attr:`lost_chunk_ids` holds the chunks no answer came back for
    (a later run should ask again) and :attr:`llm_calls` the calls made.
    """

    def __init__(self, llm, *, prompts: RelationPrompts = EN_RELATION_PROMPTS, char_budget: int = 10000,
                 max_output_tokens: int = 0, timeout: float | None = None, max_workers: int = 1,
                 max_vocabulary: int = 120, canon_batch: int = 80, max_split_depth: int = 3,
                 retry_backoff: float = 5.0, should_stop: Callable[[], bool] | None = None,
                 progress: Callable[[int, int], None] | None = None, header_patterns=()):
        self.llm = llm
        self.prompts = prompts
        self.char_budget = max(1, int(char_budget))
        self.max_output_tokens = max(0, int(max_output_tokens or 0))
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self.max_vocabulary = max_vocabulary
        self.canon_batch = max(1, canon_batch)
        self.max_split_depth = max_split_depth
        self.retry_backoff = retry_backoff
        self.should_stop = should_stop
        self.progress = progress
        self.header_patterns = tuple(header_patterns or ())
        self.llm_calls = 0
        self.lost_chunk_ids: set[str] = set()
        self.empty_answers = 0
        self._lock = threading.Lock()
        self._obs_rel_per_unit: float | None = None
        self._obs_tpi: float | None = None

    # ── text blocks ──

    def units_block(self, units: Sequence[Unit]) -> str:
        lines: list[str] = []
        for k, u in enumerate(units, 1):
            lines.append(f"[{k}] {u.prompt_text()}")
            lines.append(self.prompts.entities_prefix + " ".join(f"{n}={m}" for n, m in enumerate(u.mentions, 1)))
        return "\n".join(lines)

    def unit_chars(self, u: Unit) -> int:
        """What a unit takes of a prompt: its block, plus the line break before it."""
        return len(self.units_block([u])) + 1

    @staticmethod
    def vocab_block(vocab: Sequence[dict[str, str]]) -> str:
        return "\n".join(f"- {v['name']}: {v.get('definition') or v.get('description') or ''}".rstrip(": ")
                         for v in vocab)

    def _graph_block(self) -> str:
        return self.vocab_block([{"name": n, "definition": d} for n, d in self.prompts.graph_relations])

    def _domain(self, domain: str) -> str:
        return self.prompts.domain_line.format(domain=domain or self.prompts.auto_domain)

    def relation_prompt(self, units: Sequence[Unit], domain: str, vocab: Sequence[dict[str, str]]) -> str:
        p = self.prompts
        return (f"{p.vocabulary_heading}\n{self.vocab_block(vocab)}\n\n"
                f"{self._domain(domain)}\n\n{p.units_heading}\n" + self.units_block(units))

    @staticmethod
    def _example_lines(names: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> list[str]:
        return [f"- {n}: " + " / ".join(f"{a} → {n} → {b}" for a, b in list(ex)[:2]) for n, ex in names]

    # ── planning ──

    def plan(self, documents: dict[str, list[dict]],
             labels_by_chunk: dict[str, Sequence[str]]) -> tuple[list[Unit], list[list[Unit]]]:
        """Units, and the batches they are sent in: within the input budget, and with units x
        expected relations per unit within the answer's array cap."""
        units = units_for_documents(documents, labels_by_chunk, header_patterns=self.header_patterns)
        out_items = schema_max_items(self.max_output_tokens, self._tok_per_item())
        per_unit = self._rel_per_unit()
        batches: list[list[Unit]] = []
        cur: list[Unit] = []
        size = 0
        for u in units:
            n = self.unit_chars(u)
            full = size + n > self.char_budget or (len(cur) + 1) * per_unit > out_items
            if cur and full:
                batches.append(cur)
                cur, size = [], 0
            cur.append(u)
            size += n
        if cur:
            batches.append(cur)
        return units, batches

    def sample_units(self, units: Sequence[Unit]) -> list[Unit]:
        """Units drawn evenly over the whole input, within one call's budget (the first batch alone
        would give the vocabulary of one document)."""
        if not units:
            return []
        step = max(1, len(units) * 100 // max(self.char_budget, 1))
        out: list[Unit] = []
        size = 0
        for u in units[::step]:
            n = self.unit_chars(u)
            if out and size + n > self.char_budget:
                break
            out.append(u)
            size += n
        return out

    # ── the pass ──

    def extract(self, documents: dict[str, list[dict]], labels_by_chunk: dict[str, Sequence[str]], *,
                vocabulary: Sequence[dict[str, str]] | None = None,
                domain: str = "") -> tuple[list[Relation], dict[str, Any]]:
        """The relations ``documents`` state between the nodes each chunk mentions.

        ``labels_by_chunk`` maps a chunk id to the node labels linked to it. ``vocabulary`` is
        ``[{"name", "definition"}]``; without one a vocabulary is defined from a sample of the
        units first. Returns the relations and stats; names defined on the way (a new
        vocabulary, frequent new names) are ``stats["new_vocabulary"]``.
        """
        units, batches = self.plan(documents, labels_by_chunk)
        stats: dict[str, Any] = {"units": len(units), "calls": len(batches),
                                 "prompt_chars": sum(self.unit_chars(u) for u in units), "relations": 0}
        if not batches:
            return [], stats
        vocab = [{"name": vocab_name(v.get("name")), "definition": str(v.get("definition") or "")}
                 for v in (vocabulary or []) if vocab_name(v.get("name"))][:self.max_vocabulary]
        if not vocab:
            if self._stopped():
                self.lost_chunk_ids.update(u.chunk_id for u in units)
                return [], stats
            vocab = self.define_vocabulary(self.sample_units(units), domain)
            stats["new_vocabulary"] = vocab
        stats["vocabulary"] = len(vocab)
        if not vocab:
            # nothing to classify against: ask again next time rather than mark these chunks done
            self.lost_chunk_ids.update(u.chunk_id for u in units)
            return [], stats

        done = [0]

        def run(batch: list[Unit]) -> list[Relation]:
            try:
                return self._classify(batch, domain, vocab)
            except Exception:
                with self._lock:
                    self.lost_chunk_ids.update(u.chunk_id for u in batch)
                return []
            finally:
                with self._lock:
                    done[0] += 1
                    n = done[0]
                if self.progress is not None:
                    try:
                        self.progress(n, len(batches))
                    except Exception:
                        pass

        if self.max_workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(run, batches))
        else:
            results = [run(b) for b in batches]
        relations = [r for got in results for r in got]
        govern_predicates(relations, None, seed_predicates=[v["name"] for v in vocab])
        stats["empty_answers"] = self.empty_answers
        if not self._stopped():
            grown = self.grow_vocabulary(relations, vocab)
            if grown:
                stats["new_vocabulary"] = list(stats.get("new_vocabulary") or []) + grown
        stats["relations"] = len(relations)
        return relations, stats

    def define_vocabulary(self, units: Sequence[Unit], domain: str = "") -> list[dict[str, str]]:
        """Relation names and definitions from a sample of units."""
        if not units:
            return []
        p = self.prompts
        parsed, _meta = self._call(
            p.vocabulary_system,
            f"{p.graph_heading}\n{self._graph_block()}\n\n{self._domain(domain)}\n\n{p.units_heading}\n"
            + self.units_block(units),
            vocabulary_schema(self.max_output_tokens, self._tok_per_item()))
        return self._names_from(parsed, set())[:self.max_vocabulary]

    def define_from_names(self, names: dict[str, Sequence[tuple[str, str]]],
                          vocab: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        """Vocabulary entries for names used often but missing from ``vocab`` (``names``: name ->
        [(subject, object)] examples). Names may come back merged under one representative."""
        if not names:
            return []
        p = self.prompts
        parsed, _meta = self._call(
            p.extend_system,
            f"{p.graph_heading}\n{self._graph_block()}\n\n{p.list_heading}\n{self.vocab_block(vocab)}\n\n"
            f"{p.missing_heading}\n" + "\n".join(self._example_lines(list(names.items()))),
            vocabulary_schema(self.max_output_tokens, self._tok_per_item()))
        return self._names_from(parsed, {v["name"].lower() for v in vocab})

    def canonicalize_names(self, names: dict[str, Sequence[tuple[str, str]]],
                           vocab: Sequence[dict[str, str]]) -> dict[str, str]:
        """Names outside the vocabulary -> the vocabulary name with the same meaning, or ``""``."""
        if not names or not vocab:
            return {}
        p = self.prompts
        allowed = {v["name"] for v in vocab}
        out: dict[str, str] = {}
        items = list(names.items())
        for i in range(0, len(items), self.canon_batch):
            part = items[i:i + self.canon_batch]
            if self._stopped():
                break
            parsed, _meta = self._call(
                p.canonicalize_system,
                f"{p.list_heading}\n{self.vocab_block(vocab)}\n\n{p.missing_heading}\n"
                + "\n".join(self._example_lines(part)),
                canon_schema([v["name"] for v in vocab], len(part)))
            for row in (parsed or {}).get("mapping", []) or []:
                if not isinstance(row, dict):
                    continue
                n, t = str(row.get("name") or "").strip(), str(row.get("target") or "").strip()
                if n in names:
                    out[n] = t if t in allowed else ""
        return out

    def grow_vocabulary(self, relations: Sequence[Relation],
                        vocab: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        """Define the names used often outside the vocabulary; "often" is the natural break."""
        have = {v["name"].lower() for v in vocab}
        counts: dict[str, int] = {}
        examples: dict[str, list[tuple[str, str]]] = {}
        for r in relations:
            pred = str(r.predicate or "")
            if pred.lower() in have or pred.lower() in RESERVED_PREDICATE_NAMES:
                continue
            counts[pred] = counts.get(pred, 0) + 1
            if len(examples.setdefault(pred, [])) < 2:
                examples[pred].append((r.subject, r.object))
        if not counts:
            return []
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        frequent = natural_cut(ranked, min_k=1) if len(ranked) > 1 else [ranked[0][0]]
        return self.define_from_names({n: examples[n] for n in frequent}, vocab)

    # ── one batch ──

    def _classify(self, batch: list[Unit], domain: str, vocab: Sequence[dict[str, str]]) -> list[Relation]:
        """One batch. A cut-off answer keeps what came and the batch is halved (up to
        ``max_split_depth``); an empty answer is retried once after ``retry_backoff``; an answer
        with no relation is asked once more, after which it is taken as it is."""
        pending: list[tuple[list[Unit], int]] = [(batch, 0)]
        out: list[Relation] = []
        retried = False
        asked_again: set = set()
        allowed = {v["name"].lower(): v["name"] for v in vocab}
        while pending:
            cur, depth = pending.pop()
            if self._stopped():
                with self._lock:
                    self.lost_chunk_ids.update(u.chunk_id for c, _d in [(cur, depth), *pending] for u in c)
                return out
            parsed, meta = self._call(self.prompts.classify_system, self.relation_prompt(cur, domain, vocab),
                                      relation_schema(self.max_output_tokens, self._tok_per_item(), len(cur)))
            if meta.get("truncated") or not parsed:
                if parsed:
                    out.extend(self._parse(cur, parsed, allowed))
                if meta.get("truncated"):
                    self._note_truncation(len(cur))
                if len(cur) > 1 and depth < self.max_split_depth:
                    mid = len(cur) // 2
                    pending.extend([(cur[:mid], depth + 1), (cur[mid:], depth + 1)])
                    continue
                if not parsed and not retried:
                    retried = True
                    if self.retry_backoff > 0:
                        time.sleep(self.retry_backoff)
                    pending.append((cur, depth))
                    continue
                if not parsed:
                    with self._lock:
                        self.lost_chunk_ids.update(u.chunk_id for u in cur)
                continue
            self._note_item_cost(parsed, int(meta.get("out_tokens") or 0))
            got = self._parse(cur, parsed, allowed)
            key = (cur[0].chunk_id, len(cur))
            if not got and key not in asked_again:
                asked_again.add(key)
                with self._lock:
                    self.empty_answers += 1
                pending.append((cur, depth))
                continue
            if len(got) >= schema_max_items(self.max_output_tokens, self._tok_per_item()):
                self._note_truncation(len(cur))    # the array is full: these units say more than expected
            else:
                self._note_relations_per_unit(len(cur), len(got))
            out.extend(got)
        return out

    @staticmethod
    def parse_relations(units: Sequence[Unit], parsed: dict[str, Any],
                        allowed: dict[str, str] | None = None) -> list[Relation]:
        """``{"u","s","p","o"}`` rows -> relations. A row with a number out of range, a subject that
        is its object, or a predicate that is not an English identifier or is the graph's own
        vocabulary is dropped. ``allowed`` (lower case -> vocabulary name) fixes the spelling of
        vocabulary names; other names stay for the end-of-build canonicalization."""
        rows = parsed.get("relations") if isinstance(parsed, dict) else None
        seen = set()
        out: list[Relation] = []
        for row in rows or []:
            if isinstance(row, dict):
                try:
                    u, i, j = int(row.get("u")), int(row.get("s")), int(row.get("o"))
                except (TypeError, ValueError):
                    continue
                pred = str(row.get("p") or "").strip()
            else:
                m = _REL_LINE.match(str(row))
                if not m:
                    continue
                u, i, pred, j = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            if not 1 <= u <= len(units):
                continue
            unit = units[u - 1]
            n = len(unit.mentions)
            if i == j or not (1 <= i <= n and 1 <= j <= n):
                continue
            if not _PRED_RE.match(pred) or pred.lower() in RESERVED_PREDICATE_NAMES:
                continue
            if allowed:
                pred = allowed.get(pred.lower(), pred)
            key = (u, i, pred.lower(), j)
            if key in seen:
                continue
            seen.add(key)
            out.append(Relation(subject=unit.mentions[i - 1], predicate=pred, object=unit.mentions[j - 1],
                                predicate_type="ObjectProperty", source_chunks=[unit.chunk_id],
                                label=relation_label(unit, i - 1, j - 1)))
        return out

    _parse = parse_relations

    # ── plumbing and what the calls teach ──

    def _stopped(self) -> bool:
        if self.should_stop is None:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        with self._lock:
            self.llm_calls += 1
        return invoke_json_meta(self.llm, system, user, schema=schema,
                                max_tokens=self.max_output_tokens or None, timeout=self.timeout)

    def _names_from(self, parsed: dict, have: set[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in (parsed or {}).get("predicates", []) or []:
            if not isinstance(row, dict):
                continue
            name = vocab_name(row.get("name"))
            if not name or name.lower() in have:
                continue
            have.add(name.lower())
            out.append({"name": name, "definition": str(row.get("definition") or "").strip()[:_SCHEMA_DESC_LEN]})
        return out

    def _tok_per_item(self) -> float:
        obs = self._obs_tpi
        return float(obs) if obs and obs > 0 else float(_SCHEMA_TOK_PER_ITEM)

    def _note_item_cost(self, parsed: dict[str, Any], out_tokens: int) -> None:
        """Output tokens per answer element, as a moving average (known only when the LLM reports tokens)."""
        if not isinstance(parsed, dict) or not out_tokens or out_tokens <= 0:
            return
        n = sum(len(v) for v in parsed.values() if isinstance(v, list))
        if n <= 0:
            return
        cost = out_tokens / n
        if not (5.0 <= cost <= 500.0):
            return
        with self._lock:
            prev = self._obs_tpi
            self._obs_tpi = cost if prev is None else prev * 0.7 + cost * 0.3

    def _rel_per_unit(self) -> float:
        obs = self._obs_rel_per_unit
        return float(obs) if obs and obs > 0 else 1.0

    def _note_relations_per_unit(self, n_units: int, n_relations: int) -> None:
        if n_units <= 0:
            return
        ratio = max(n_relations / n_units, 0.1)
        with self._lock:
            prev = self._obs_rel_per_unit
            self._obs_rel_per_unit = ratio if prev is None else prev * 0.7 + ratio * 0.3

    def _note_truncation(self, n_units: int) -> None:
        """A cut-off (or full) answer says these units hold at least the array cap of relations.
        That, with a margin, goes into the same moving average as complete answers, so one cut
        does not decide the next batch size alone."""
        if n_units <= 0:
            return
        seen = schema_max_items(self.max_output_tokens, self._tok_per_item()) / n_units * 1.5
        with self._lock:
            prev = self._obs_rel_per_unit
            self._obs_rel_per_unit = max(seen, 1.0) if prev is None else prev * 0.7 + seen * 0.3


def labels_by_chunk(concepts: Concepts, instances) -> dict[str, list[str]]:
    """The node labels linked to each chunk: individuals and classes by their source chunks."""
    out: dict[str, set[str]] = {}
    for i in instances:
        if i.name:
            for cid in i.source_chunks or []:
                out.setdefault(str(cid), set()).add(i.name)
    for c in concepts.classes:
        if c.name:
            for cid in c.source_chunks or []:
                out.setdefault(str(cid), set()).add(c.name)
    return {k: sorted(v) for k, v in out.items()}


def canonicalize_predicates(concepts: Concepts, relations: list[Relation], former: RelationFormer) -> dict[str, int]:
    """Hold every relation name to the vocabulary, in place (the production end-of-build step).

    Names outside the vocabulary (and not the graph's own) are mapped by the model onto a
    vocabulary name with the same meaning. Of the unmapped ones, the frequent ones (natural
    break) are defined and join the vocabulary as object properties; the rest become
    ``relatedTo`` with their old name kept as the relation's label. Merge duplicates afterwards
    (:func:`~xgen_ontology.build.finalize.normalize_graph`)."""
    vocab = vocabulary_of(concepts)
    if not vocab:
        return {}
    counts = Counter(r.predicate for r in relations
                     if r.predicate and r.predicate_type != "DatatypeProperty")
    have = {v["name"] for v in vocab}
    names = sorted((n for n in counts if n not in GRAPH_VOCABULARY and n not in have),
                   key=lambda n: (-counts[n], n))
    if not names:
        return {}
    examples: dict[str, list[tuple[str, str]]] = {}
    for r in relations:
        if r.predicate in counts and len(examples.setdefault(r.predicate, [])) < 2:
            examples[r.predicate].append((r.subject, r.object))
    mapping = former.canonicalize_names({n: examples.get(n, []) for n in names}, vocab)
    unmatched = sorted(((n, counts[n]) for n in names if not mapping.get(n)), key=lambda kv: -kv[1])
    frequent = natural_cut(unmatched, min_k=1) if len(unmatched) > 1 else [n for n, _ in unmatched]
    added = former.define_from_names({n: examples.get(n, []) for n in frequent}, vocab) if frequent else []
    keep = {v["name"] for v in added}
    declared = {op.name for op in concepts.object_properties}
    for v in added:
        if v["name"] not in declared:
            concepts.object_properties.append(ObjectProperty(name=v["name"], description=v["definition"]))
            declared.add(v["name"])
    final = {n: (mapping.get(n) or RELATED_PREDICATE) for n in names if n not in keep}
    renamed = 0
    for r in relations:
        target = final.get(r.predicate)
        if target is None:
            continue
        if target == RELATED_PREDICATE:
            r.label = r.label or r.predicate
        r.predicate = target
        renamed += 1
    return {"names": len(names), "to_vocabulary": sum(1 for n in names if mapping.get(n)),
            "added_to_vocabulary": len(added), "renamed": renamed}
