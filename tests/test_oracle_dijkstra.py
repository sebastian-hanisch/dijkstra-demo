"""Unabhängiges Orakel für den Kern: Entfernungen aus einer Matrix-Floyd-Warshall-Rechnung (anderer Rechenweg als jede Warteschlange), Mehrquellen, Nullkanten, Rohgraphen mit
Parallelkanten und Schleifen; Bellman-Ford-Referenz und Alarm gegen networkx; Anzahl falscher Netze gegen eine Neuberechnung mit networkx."""

import networkx as nx
import numpy as np
import pytest

import dj_algorithm as alg
import dj_evaluation as ev
from dj_graph import from_arcs, route_cost
from dj_queues import QUEUES


def _random_graph(rng, n, m, neg=0.0, zero=0.0, clean=True, directed=True):
    arcs = []
    for _ in range(m):
        u, v = (int(x) for x in rng.integers(0, n, 2))
        w = float(rng.integers(1, 9))
        if zero and rng.random() < zero:
            w = 0.0
        if neg and rng.random() < neg:
            w = -float(rng.integers(1, 5))
        arcs.append((u, v, w))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed, clean=clean)


def _matrix_distances(g, sources):
    D = np.full((g.n, g.n), np.inf)
    np.fill_diagonal(D, 0)
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            D[u, v] = min(D[u, v], w)
    for k in range(g.n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.min([D[s] for s in sources], axis=0)


@pytest.mark.parametrize("queue", list(QUEUES))
def test_dijkstra_equals_matrix_shortest_paths_on_small_random_graphs(queue):
    rng = np.random.default_rng(11)
    for it in range(60):
        n, m = int(rng.integers(2, 11)), int(rng.integers(1, 28))
        g = _random_graph(rng, n, m, zero=0.2 if it % 3 == 0 else 0.0, clean=bool(it % 2), directed=bool(it % 4 < 2))
        if g.m == 0:
            continue
        sources = [int(x) for x in rng.choice(n, int(rng.integers(1, min(n, 3) + 1)), replace=False)]
        ref = _matrix_distances(g, sources)
        res = alg.dijkstra(g, sources, queue=queue)
        assert np.array_equal(np.isfinite(res.dist), np.isfinite(ref))
        assert np.allclose(res.dist[np.isfinite(ref)], ref[np.isfinite(ref)])
        for v in np.flatnonzero(np.isfinite(ref)):
            route = res.route(int(v))
            assert route[0] in sources and route[-1] == v and route_cost(g, route) == pytest.approx(ref[v])
        targets = [int(x) for x in rng.choice(n, min(n, 2), replace=False)]
        early = alg.dijkstra(g, sources, targets, queue=queue)
        reachable = [t for t in targets if np.isfinite(ref[t])]
        if reachable:
            assert early.found in targets and ref[early.found] == min(ref[t] for t in reachable)
        else:
            assert early.found == -1


def test_bellman_ford_and_alarm_against_networkx_with_negative_edges():
    rng = np.random.default_rng(5)
    for _ in range(80):
        n = int(rng.integers(2, 9))
        g = _random_graph(rng, n, int(rng.integers(1, 22)), neg=0.25)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v, w in zip(g.out(u), g.out_weights(u)):
                G.add_edge(u, int(v), weight=float(w))
        bf = alg.bellman_ford_reference(g, 0)
        try:
            ref = nx.single_source_bellman_ford_path_length(G, 0)
        except nx.NetworkXUnbounded:
            assert bf.negative_cycle
            continue
        assert not bf.negative_cycle
        for v in range(n):
            assert (bf.dist[v] == pytest.approx(ref[v])) if v in ref else not np.isfinite(bf.dist[v])
        res = alg.dijkstra(g, [0])
        if any(abs(res.dist[v] - ref[v]) > 1e-9 for v in ref):
            assert res.alarms                                   # falsche Kosten ohne Alarm darf es nicht geben
        assert all(res.dist[v] >= ref[v] - 1e-9 for v in ref)  # Dijkstra findet echte Wege: nie unter dem Optimum


def test_negative_share_counts_equal_an_independent_recount():
    n, m, trials, seeds = 15, 40, 10, (1, 2)
    for row in ev.negative_share(fractions=(0.0, 0.05, 0.2), n=n, m=m, trials=trials, seeds=seeds):
        wrong = right = cycle = 0
        for sd in seeds:
            rng = np.random.default_rng([sd, 606])
            for _ in range(trials):
                arcs = []
                for _ in range(m):
                    u, v = (int(x) for x in rng.integers(0, n, 2))
                    arcs.append((u, v, -float(rng.integers(1, 5)) if rng.random() < row["fraction"] else float(rng.integers(1, 10))))
                G = nx.DiGraph()
                G.add_nodes_from(range(n))
                for u, v, w in arcs:
                    if u != v and (not G.has_edge(u, v) or w < G[u][v]["weight"]):
                        G.add_edge(u, v, weight=w)
                try:
                    ref = nx.single_source_bellman_ford_path_length(G, 0)
                except nx.NetworkXUnbounded:
                    cycle += 1
                    continue
                d = alg.dijkstra(from_arcs(n, arcs, np.zeros((n, 2)), directed=True), [0]).dist
                if any(abs(d[v] - ref[v]) > 1e-9 for v in ref):
                    wrong += 1
                else:
                    right += 1
        assert (row["wrong"], row["right"], row["cycle"]) == (wrong, right, cycle)
