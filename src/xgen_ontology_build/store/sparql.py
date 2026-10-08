"""SparqlGraph — read/write any SPARQL 1.1 store (not Fuseki-only).

Stdlib-only (urllib). A build is pushed as Turtle and read back through the same
GraphStore methods PgGraph offers. Label search uses portable ``FILTER(CONTAINS(...))`` so it works on
Apache Jena Fuseki, GraphDB, Blazegraph, Virtuoso, rdflib-backed endpoints — any
SPARQL 1.1 service. Writes go through the SPARQL Graph Store Protocol (``gsp_url``)
when set, falling back to ``INSERT DATA`` on the update endpoint.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request

from xgen_ontology_build.models import Node
from xgen_ontology_build.text.tokens import tokenize

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"


class SparqlGraph:
    """GraphStore + GraphSink over a SPARQL 1.1 endpoint."""

    def __init__(
        self,
        query_url: str,
        graph_uri: str | None = None,
        *,
        update_url: str | None = None,
        gsp_url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        timeout: float = 30,
    ):
        self.query_url = query_url
        self.update_url = update_url
        self.gsp_url = gsp_url
        self.graph_uri = graph_uri
        self.timeout = timeout
        self._auth = (
            "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
            if user and password else None
        )

    # ── helpers ──

    def _g(self, body: str) -> str:
        return f"GRAPH <{self.graph_uri}> {{ {body} }}" if self.graph_uri else body

    def _post(self, url: str, data: bytes, content_type: str, accept: str = "") -> bytes:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
        if accept:
            req.add_header("Accept", accept)
        if self._auth:
            req.add_header("Authorization", self._auth)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _query(self, sparql: str) -> dict:
        data = urllib.parse.urlencode({"query": sparql}).encode()
        raw = self._post(self.query_url, data, "application/x-www-form-urlencoded",
                         "application/sparql-results+json")
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _local(uri: str) -> str:
        return uri.split("#")[-1].split("/")[-1]

    @staticmethod
    def _val(b: dict, key: str) -> str:
        return b.get(key, {}).get("value", "")

    # ── GraphStore (read) ──

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        terms = [t for t in tokenize(query) if len(t) >= 2][:8]
        if not terms:
            return []
        filt = " || ".join(f'CONTAINS(LCASE(STR(?l)), "{t}")' for t in terms)
        body = (f'?n <{_RDFS_LABEL}> ?l . OPTIONAL {{ ?n <{_RDF_TYPE}> ?ty . FILTER(?ty = <{_OWL_CLASS}>) }} '
                f'FILTER({filt})')
        q = f"SELECT DISTINCT ?n ?l ?ty WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            nid, lab = self._val(b, "n"), self._val(b, "l")
            kind = "class" if self._val(b, "ty") else "instance"
            ll = lab.lower()
            score = float(sum(t in ll for t in terms)) or 1.0
            out.append((Node(nid, lab, kind), score))
        out.sort(key=lambda x: -x[1])
        return out

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        body = (f'?i (<{_RDF_TYPE}>|<{_RDFS_SUBCLASS}>) <{class_id}> . '
                f'OPTIONAL {{ ?i <{_RDFS_LABEL}> ?l }}')
        q = f"SELECT DISTINCT ?i ?l WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            iid = self._val(b, "i")
            out.append(Node(iid, self._val(b, "l") or self._local(iid), "instance"))
        return out

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        body = (f'{{ BIND(<{node_id}> AS ?s) ?s ?p ?o }} UNION {{ BIND(<{node_id}> AS ?o) ?s ?p ?o }} '
                f'OPTIONAL {{ ?s <{_RDFS_LABEL}> ?sl }} OPTIONAL {{ ?o <{_RDFS_LABEL}> ?ol }} '
                f'FILTER(?p != <{_RDF_TYPE}> && ?p != <{_RDFS_LABEL}>)')
        q = f"SELECT ?s ?sl ?p ?o ?ol WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            sl = self._val(b, "sl") or self._local(self._val(b, "s"))
            ol = self._val(b, "ol") or (self._val(b, "o") if b.get("o", {}).get("type") == "literal"
                                        else self._local(self._val(b, "o")))
            out.append((sl, self._local(self._val(b, "p")), ol))
        return out

    def count_class(self, class_id: str) -> int:
        body = f'?i (<{_RDF_TYPE}>|<{_RDFS_SUBCLASS}>) <{class_id}>'
        q = f"SELECT (COUNT(DISTINCT ?i) AS ?c) WHERE {{ {self._g(body)} }}"
        bs = self._query(q).get("results", {}).get("bindings", [])
        return int(bs[0]["c"]["value"]) if bs else 0

    def get_node(self, node_id: str) -> Node | None:
        q = f"SELECT ?l WHERE {{ {self._g(f'<{node_id}> <{_RDFS_LABEL}> ?l')} }} LIMIT 1"
        bs = self._query(q).get("results", {}).get("bindings", [])
        return Node(node_id, (bs[0]["l"]["value"] if bs else self._local(node_id)))

    # ── GraphSink (write) ──

    def upload_turtle(self, ttl: str, *, graph: str | None = None, clear: bool = False) -> None:
        g = graph or self.graph_uri
        if self.gsp_url:
            url = self.gsp_url
            if g:
                url += ("&" if "?" in url else "?") + "graph=" + urllib.parse.quote(g, safe="")
            if clear:
                self._delete(url)
            method = "PUT" if clear else "POST"
            req = urllib.request.Request(url, data=ttl.encode("utf-8"),
                                         headers={"Content-Type": "text/turtle"}, method=method)
            if self._auth:
                req.add_header("Authorization", self._auth)
            with urllib.request.urlopen(req, timeout=self.timeout):
                return
        if not self.update_url:
            raise RuntimeError("upload_turtle needs gsp_url or update_url")
        raise RuntimeError(
            "Turtle upload via the update endpoint isn't supported without a Turtle parser; "
            "set gsp_url (SPARQL Graph Store Protocol) for writes."
        )

    def _delete(self, url: str) -> None:
        req = urllib.request.Request(url, method="DELETE")
        if self._auth:
            req.add_header("Authorization", self._auth)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                return
        except Exception:
            pass


def fuseki(base_url: str, dataset: str, **kwargs) -> SparqlGraph:
    """Convenience constructor for an Apache Jena Fuseki dataset.

    ``fuseki("http://localhost:3030", "ds")`` wires query/update/GSP urls."""
    base = base_url.rstrip("/")
    return SparqlGraph(
        query_url=f"{base}/{dataset}/query",
        update_url=f"{base}/{dataset}/update",
        gsp_url=f"{base}/{dataset}/data",
        **kwargs,
    )
