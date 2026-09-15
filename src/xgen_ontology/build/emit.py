"""Serialize a built ontology to RDF.

* ``to_rdf_triples`` — materialize classes / properties / hierarchy / instances /
  relations / data values into ``RDFTriple`` objects, with deterministic IRIs
  (translations map names to ASCII local names; otherwise the name is kept as a
  Unicode-safe local name). Values are cleaned, dates ISO-normalized and xsd types
  inferred — the same hygiene the production builder applied.
* ``to_turtle`` — a tiny, dependency-free Turtle writer.
* ``to_owl_xml`` — RDF/XML via rdflib (optional ``[rdf]`` extra).
"""
from __future__ import annotations

import re

from ..models import Concepts, DataValue, Instance, RDFTriple, Relation
from ..text import safe_uri

SCHEMA_NS = "https://w3id.org/xgen-ontology#"
INSTANCE_NS = "https://w3id.org/xgen-ontology/instance#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

_PREFIXES = {
    "": SCHEMA_NS, "inst": INSTANCE_NS, "rdf": RDF_NS,
    "rdfs": RDFS_NS, "owl": OWL_NS, "xsd": XSD_NS,
}


def to_rdf_triples(
    concepts: Concepts,
    instances: list[Instance],
    relations: list[Relation],
    data_values: list[DataValue],
    *,
    translations: dict[str, str] | None = None,
    include_source_chunks: bool = True,
) -> list[RDFTriple]:
    translations = translations or {}

    def local(name: str, is_class: bool) -> str:
        eng = translations.get(name)
        if eng:
            cleaned = re.sub(r"[^A-Za-z0-9]", "", eng)
            if cleaned:
                # classes UpperCamelCase, properties lowerCamelCase (the OWL convention)
                return (cleaned[0].upper() if is_class else cleaned[0].lower()) + cleaned[1:]
        return safe_uri(name)

    def cls_uri(name: str) -> str:
        return SCHEMA_NS + local(name, True)

    def prop_uri(name: str) -> str:
        return SCHEMA_NS + local(name, False)

    def inst_uri(name: str) -> str:
        return INSTANCE_NS + safe_uri(name)

    t: list[RDFTriple] = []
    add = t.append

    def lit(s, p, value, datatype="", lang=""):
        add(RDFTriple(s, p, str(value), o_is_literal=True, datatype=datatype, lang=lang))

    class_names = {c.name for c in concepts.classes if c.name}

    # classes
    for c in concepts.classes:
        if not c.name:
            continue
        u = cls_uri(c.name)
        add(RDFTriple(u, RDF_NS + "type", OWL_NS + "Class"))
        lit(u, RDFS_NS + "label", c.name)
        if c.description:
            lit(u, RDFS_NS + "comment", c.description)
    for parent, child in concepts.class_hierarchy:
        if parent in class_names and child in class_names and parent != child:
            add(RDFTriple(cls_uri(child), RDFS_NS + "subClassOf", cls_uri(parent)))

    # object properties
    for op in concepts.object_properties:
        if not op.name:
            continue
        u = prop_uri(op.name)
        add(RDFTriple(u, RDF_NS + "type", OWL_NS + "ObjectProperty"))
        lit(u, RDFS_NS + "label", op.name)
        if op.domain and op.domain in class_names:
            add(RDFTriple(u, RDFS_NS + "domain", cls_uri(op.domain)))
        if op.range and op.range in class_names:
            add(RDFTriple(u, RDFS_NS + "range", cls_uri(op.range)))

    # datatype properties
    for dp in concepts.datatype_properties:
        if not dp.name:
            continue
        u = prop_uri(dp.name)
        add(RDFTriple(u, RDF_NS + "type", OWL_NS + "DatatypeProperty"))
        lit(u, RDFS_NS + "label", dp.name)
        if dp.domain and dp.domain in class_names:
            add(RDFTriple(u, RDFS_NS + "domain", cls_uri(dp.domain)))
        rng = dp.range or "xsd:string"
        if rng.startswith("xsd:"):
            add(RDFTriple(u, RDFS_NS + "range", XSD_NS + rng.split(":", 1)[1]))

    # instances
    known = {i.name for i in instances if i.name}
    for inst in instances:
        if not inst.name:
            continue
        u = inst_uri(inst.name)
        add(RDFTriple(u, RDF_NS + "type", OWL_NS + "NamedIndividual"))
        if inst.class_name:
            add(RDFTriple(u, RDF_NS + "type", cls_uri(inst.class_name)))
        lit(u, RDFS_NS + "label", inst.name)
        if include_source_chunks:
            for cid in inst.source_chunks or []:
                if cid:
                    lit(u, SCHEMA_NS + "sourceChunk", cid)

    # relations
    linked: set = set()
    for r in relations:
        if not (r.subject and r.predicate and r.object):
            continue
        su = inst_uri(r.subject)
        pu = prop_uri(r.predicate)
        if r.predicate_type == "DatatypeProperty":
            lit(su, pu, r.object)
            continue
        o = r.object.strip()
        if not o or re.match(r"^-?\d+\.?\d*$", o):
            continue
        ou = inst_uri(r.object)
        key = (su, ou)
        if key in linked:
            continue
        linked.add(key)
        if r.object not in known:
            add(RDFTriple(ou, RDF_NS + "type", OWL_NS + "NamedIndividual"))
            lit(ou, RDFS_NS + "label", r.object)
        add(RDFTriple(su, pu, ou))

    # data values
    for dv in data_values:
        if not dv.entity or not dv.property:
            continue
        value = dv.value
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        vs = clean_value(str(value).strip())
        if looks_like_date(vs):
            vs = normalize_date(vs)
        vtype = dv.value_type or "xsd:string"
        if vtype == "xsd:string" and vs:
            vtype = auto_xsd_type(vs)
        su = inst_uri(dv.entity)
        pu = prop_uri(dv.property)
        if "date" in vtype:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", vs):
                lit(su, pu, vs, datatype="xsd:date")
            else:
                lit(su, pu, vs)
        elif "integer" in vtype:
            try:
                iv = int(float(vs)) if "." in vs else int(vs)
                lit(su, pu, iv, datatype="xsd:integer")
            except (ValueError, TypeError):
                lit(su, pu, vs)
        elif "decimal" in vtype or "float" in vtype:
            try:
                lit(su, pu, float(vs), datatype="xsd:decimal")
            except (ValueError, TypeError):
                lit(su, pu, vs)
        elif "boolean" in vtype:
            lit(su, pu, "true" if vs.lower() in ("true", "1", "yes") else "false", datatype="xsd:boolean")
        else:
            lit(su, pu, vs)

    return t


# ───────────────────────── Turtle writer (zero-dep) ─────────────────────────


def to_turtle(triples: list[RDFTriple]) -> str:
    lines = [f"@prefix {p}: <{ns}> ." for p, ns in _PREFIXES.items()]
    lines.append("")
    for tr in triples:
        s = _term(tr.s)
        p = "a" if tr.p == RDF_NS + "type" else _term(tr.p)
        o = _literal(tr) if tr.o_is_literal else _term(tr.o)
        lines.append(f"{s} {p} {o} .")
    return "\n".join(lines) + "\n"


def _term(iri: str) -> str:
    for prefix, ns in _PREFIXES.items():
        if iri.startswith(ns):
            local = iri[len(ns):]
            if local and re.fullmatch(r"[A-Za-z0-9_\-]+", local):
                return f"{prefix}:{local}"
    return f"<{iri}>"


def _literal(tr: RDFTriple) -> str:
    body = '"' + tr.o.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "") + '"'
    if tr.datatype:
        return f"{body}^^{tr.datatype}"
    if tr.lang:
        return f"{body}@{tr.lang}"
    return body


# ───────────────────────── OWL / RDF-XML (optional) ─────────────────────────


def to_owl_xml(triples: list[RDFTriple]) -> str:
    """RDF/XML via rdflib. Requires the ``[rdf]`` extra (``pip install xgen-ontology[rdf]``)."""
    try:
        from rdflib import Graph, Literal, URIRef
        from rdflib.namespace import XSD
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("to_owl_xml needs rdflib: pip install 'xgen-ontology[rdf]'") from e
    g = Graph()
    for p, ns in _PREFIXES.items():
        g.bind(p or "base", ns)
    xsd_map = {f"xsd:{n}": getattr(XSD, n) for n in
               ("string", "integer", "decimal", "float", "boolean", "date", "dateTime")}
    for tr in triples:
        s = URIRef(tr.s)
        p = URIRef(tr.p)
        if tr.o_is_literal:
            dt = xsd_map.get(tr.datatype)
            o = Literal(tr.o, datatype=dt) if dt else (Literal(tr.o, lang=tr.lang) if tr.lang else Literal(tr.o))
        else:
            o = URIRef(tr.o)
        g.add((s, p, o))
    return g.serialize(format="xml")


# ───────────────────────── value hygiene ─────────────────────────


def auto_xsd_type(value: str) -> str:
    if not value:
        return "xsd:string"
    if re.match(r"^-?\d+$", value):
        return "xsd:integer"
    if re.match(r"^-?\d+\.\d+$", value):
        return "xsd:decimal"
    if looks_like_date(value):
        return "xsd:date"
    return "xsd:string"


def looks_like_date(value: str) -> bool:
    v = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$", v):
        return True
    if re.match(r"^\d{4}[/.]\d{1,2}[/.]\d{1,2}$", v):
        return True
    if re.match(r"^\d{4}년\s*\d{1,2}월\s*\d{1,2}일?$", v):
        return True
    if re.match(r"^\d{8}$", v):
        year, month = int(v[:4]), int(v[4:6])
        return 1900 <= year <= 2099 and 1 <= month <= 12
    return False


def normalize_date(value: str) -> str:
    v = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return v
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}", v)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$", v)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일?$", v)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", v)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return value


def clean_value(value: str) -> str:
    if not value:
        return value
    m = re.match(r"^(-?\d[\d,.]*)\s*\(.*\)\s*$", value)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{4}[-/]\d{2}[-/]\d{2})\s*\(.*\)\s*$", value)
    if m:
        return m.group(1)
    return value
