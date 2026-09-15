"""Swappable backends: zero-infra in-memory, a generic SPARQL 1.1 adapter, and the XGEN PostgreSQL graph tables."""
from .memory import InMemoryGraph, InMemoryGraphSink, InMemoryVector
from .postgres import PgGraph, graph_rows
from .sparql import SparqlGraph

__all__ = ["InMemoryGraph", "InMemoryGraphSink", "InMemoryVector", "SparqlGraph", "PgGraph", "graph_rows"]
