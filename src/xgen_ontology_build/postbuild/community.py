"""Community detection — Louvain modularity, pure Python (no networkx/igraph).

Greedy one-level Louvain: move each node to the neighboring community with the
largest modularity gain, repeat to convergence. Unlike label propagation, a single
bridge edge won't collapse two dense communities (modularity prevents it).
Deterministic: nodes are visited in sorted order with stable tie-breaks.
"""
from __future__ import annotations

from collections import defaultdict

from xgen_ontology_build.models import Relation


def louvain_communities(
    node_ids: list[str],
    edges: list[tuple[str, str]],
    max_iter: int = 30,
) -> dict[str, int]:
    """Returns ``{node_id: community_index}`` (0-based, largest community first)."""
    if not node_ids:
        return {}
    node_set = set(node_ids)
    adj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    deg: dict[str, int] = defaultdict(int)
    m = 0
    for s, t in edges:
        if s in node_set and t in node_set and s != t:
            adj[s][t] += 1
            adj[t][s] += 1
            deg[s] += 1
            deg[t] += 1
            m += 1

    ordered = sorted(node_ids)
    if m == 0:
        return {n: i for i, n in enumerate(ordered)}

    two_m = 2.0 * m
    comm = {n: n for n in ordered}
    comm_tot: dict[str, int] = defaultdict(int)
    for n in ordered:
        comm_tot[comm[n]] += deg[n]

    for _ in range(max_iter):
        moved = False
        for n in ordered:
            ki = deg[n]
            cn = comm[n]
            comm_tot[cn] -= ki
            nbr_w: dict[str, int] = defaultdict(int)
            for nb, w in adj[n].items():
                nbr_w[comm[nb]] += w
            best_c, best_gain = cn, nbr_w.get(cn, 0) - (comm_tot[cn] * ki) / two_m
            for c in sorted(nbr_w):
                gain = nbr_w[c] - (comm_tot[c] * ki) / two_m
                if gain > best_gain:
                    best_gain, best_c = gain, c
            comm_tot[best_c] += ki
            if best_c != cn:
                comm[n] = best_c
                moved = True
        if not moved:
            break

    groups: dict[str, list[str]] = defaultdict(list)
    for n in ordered:
        groups[comm[n]].append(n)
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    out: dict[str, int] = {}
    for idx, members in enumerate(sorted_groups):
        for n in members:
            out[n] = idx
    return out


def detect_communities(
    instances,
    relations: list[Relation],
    *,
    top_members: int = 5,
) -> list[dict]:
    """Cluster the instance-relation graph and summarize each community.

    Returns ``[{community, size, name, members}]`` (largest first)."""
    inst_names = {i.name for i in instances if i.name}
    edges: list[tuple[str, str]] = []
    degree: dict[str, int] = defaultdict(int)
    node_set: set[str] = set()
    for r in relations:
        if r.predicate_type == "DatatypeProperty":
            continue
        if r.subject in inst_names and r.object in inst_names and r.subject != r.object:
            edges.append((r.subject, r.object))
            degree[r.subject] += 1
            degree[r.object] += 1
            node_set.add(r.subject)
            node_set.add(r.object)
    if len(node_set) < 2 or not edges:
        return []

    comm_of = louvain_communities(list(node_set), edges)
    members_by_comm: dict[int, list[str]] = defaultdict(list)
    for n, c in comm_of.items():
        members_by_comm[c].append(n)

    out = []
    for c, members in members_by_comm.items():
        ranked = sorted(members, key=lambda n: (-degree.get(n, 0), n))
        out.append({
            "community": c,
            "size": len(members),
            "name": ranked[0] if ranked else f"community-{c}",
            "members": ranked[:top_members],
        })
    out.sort(key=lambda x: -x["size"])
    return out
