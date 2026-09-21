"""Dijkstra mit allen Warteschlangen, Bellman-Ford-Referenz, BFS-Vergleich und die Warteschlangen selbst - gegen networkx und gegen ein einfaches Modell."""

import random

import networkx as nx
import numpy as np
import pytest

import dj_algorithm as alg
from dj_graph import from_arcs, route_cost
from dj_queues import QUEUES


def random_graph(n, m, seed, directed=False, integer=True, negative=0.0):
    rng = np.random.default_rng(seed)
    arcs = []
    for _ in range(m):
        u, v = rng.integers(0, n, 2)
        w = float(rng.integers(1, 9)) if integer else 0.5 + 5 * rng.random()
        if negative and rng.random() < negative:
            w = -float(rng.integers(1, 4))
        arcs.append((u, v, w))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)


def to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


# --- Dijkstra gegen networkx --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("queue", list(QUEUES))
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("directed", [False, True])
def test_distances_equal_networkx_for_every_queue(queue, seed, directed):
    g = random_graph(60, 140, seed, directed)
    ref = nx.single_source_dijkstra_path_length(to_nx(g), 0)
    res = alg.dijkstra(g, [0], queue=queue)
    for v in range(g.n):
        assert (res.dist[v] == pytest.approx(ref[v])) if v in ref else not np.isfinite(res.dist[v])
    assert set(res.order) == set(ref) and not res.alarms


@pytest.mark.parametrize("queue", list(QUEUES))
def test_routes_are_real_paths_with_the_reported_cost(queue):
    g = random_graph(50, 110, 3, integer=False if queue != "dial" else True)
    G = to_nx(g)
    for t in range(1, 50, 4):
        res = alg.dijkstra(g, [0], [t], queue=queue)
        if not nx.has_path(G, 0, t):
            assert res.found == -1 and res.route(t) == []
            continue
        route = res.route(t)
        assert route[0] == 0 and route[-1] == t and route_cost(g, route) == pytest.approx(res.dist[t]) == pytest.approx(nx.dijkstra_path_length(G, 0, t))


def test_all_queues_agree_on_distances_and_settle_the_same_nodes():
    g = random_graph(80, 200, 7)
    results = {q: alg.dijkstra(g, [0], queue=q) for q in QUEUES}
    base = results["lazy"]
    for q, r in results.items():
        assert np.allclose(np.where(np.isfinite(r.dist), r.dist, -1), np.where(np.isfinite(base.dist), base.dist, -1)), q
        assert sorted(r.order) == sorted(base.order)
        assert [base.dist[v] for v in r.order] == sorted(base.dist[v] for v in r.order)      # Festlegung in nichtfallender Entfernung


def test_settle_order_is_nondecreasing_in_distance():
    g = random_graph(100, 300, 2, integer=False)
    for q in ("binary", "fibonacci", "array", "lazy"):
        d = [alg.dijkstra(g, [0], queue=q).dist[v] for v in alg.dijkstra(g, [0], queue=q).order]
        assert d == sorted(d)


def test_early_stop_settles_fewer_nodes_and_keeps_the_target_distance():
    g = random_graph(200, 500, 4, integer=False)
    full = alg.dijkstra(g, [0])
    t = full.order[len(full.order) // 3]
    early = alg.dijkstra(g, [0], [t])
    assert early.found == t and early.dist[t] == pytest.approx(full.dist[t]) and len(early.order) == full.order.index(t) + 1 and len(early.order) < len(full.order)


def test_multiple_sources_and_targets():
    g = from_arcs(7, [(i, i + 1, 1.0) for i in range(6)], np.zeros((7, 2)))
    res = alg.dijkstra(g, [0, 6], [3, 4])
    assert res.found == 4 and res.dist[4] == 2
    assert alg.dijkstra(g, [2], [2]).found == 2 and alg.dijkstra(g, [2], [2]).order == [2]


def test_unreachable_target_is_reported():
    g = from_arcs(4, [(0, 1, 1.0), (2, 3, 1.0)], np.zeros((4, 2)))
    res = alg.dijkstra(g, [0], [3])
    assert res.found == -1 and res.route(3) == [] and not np.isfinite(res.dist[3])


# --- Zähler -------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("queue", list(QUEUES))
def test_counter_invariants(queue):
    g = random_graph(70, 180, 5)
    res = alg.dijkstra(g, [0], queue=queue)
    c = res.counters
    reached = int(np.isfinite(res.dist).sum())
    assert c["pushes"] == reached and c["pops"] == len(res.order) == reached          # jeder erreichte Knoten wird einmal eingefügt und einmal entnommen
    assert c["relaxations"] == int(sum(len(g.out(u)) for u in res.order))
    assert c["decrease_keys"] <= c["relaxations"] - reached + 1 and len(res.front) == len(res.order) and res.front[-1] == 0


def test_array_queue_work_is_quadratic_and_heaps_are_much_cheaper():
    g = random_graph(300, 900, 1)
    work = {q: alg.dijkstra(g, [0], queue=q).counters["work"] for q in ("array", "binary", "fibonacci")}
    assert work["array"] > 5 * work["binary"] and work["array"] > 5 * work["fibonacci"] and work["binary"] > 0


def test_lazy_heap_pushes_stale_entries_instead_of_decreasing():
    g = random_graph(150, 600, 2)
    lazy = alg.dijkstra(g, [0], queue="lazy").counters
    assert lazy["stale_pops"] > 0 and lazy["stale_pops"] <= lazy["decrease_keys"]


# --- Warteschlangen gegen ein einfaches Modell ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["array", "binary", "lazy", "fibonacci"])
@pytest.mark.parametrize("seed", range(4))
def test_queue_matches_a_dict_model_under_random_operations(kind, seed):
    rnd = random.Random(seed)
    q, model, popped = QUEUES[kind](100, 1000), {}, set()
    for _ in range(600):
        op = rnd.random()
        if op < 0.5 and len(model) < 60:
            node = rnd.randrange(100)
            if node in popped:
                continue
            key = rnd.randrange(1000) if node not in model else rnd.randrange(model[node] + 1)
            model[node] = key if node not in model else min(key, model[node])
            q.push_or_decrease(node, model[node])
        elif model:
            key, node = q.pop_min()
            assert key == min(model.values()) and model[node] == key
            del model[node]
            popped.add(node)
        assert len(q) == len(model)
    while model:
        key, node = q.pop_min()
        assert key == min(model.values())
        del model[node]
    assert len(q) == 0


def test_dial_queue_rejects_non_integer_keys_and_handles_bucket_wraparound():
    q = QUEUES["dial"](40, 3)                                    # größte Kosten 3 -> vier Eimer im Kreis
    with pytest.raises(ValueError):
        q.push_or_decrease(1, 1.5)
    q.push_or_decrease(0, 0)
    q.push_or_decrease(100, 2)                                   # ein zweiter Eintrag, der später mit kleinerem Schlüssel überholt wird
    q.push_or_decrease(100, 1)
    popped = []
    for i in range(12):
        key, node = q.pop_min()
        popped.append((key, node))
        if node < 30:
            q.push_or_decrease(node + 1, key + 3)
    assert [k for k, _ in popped] == sorted(k for k, _ in popped) and (1, 100) in popped and (2, 100) not in popped
    assert [k for k, n in popped if n < 30] == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    assert len(q) == 1 or len(q) == 0


# --- Negative Kanten ----------------------------------------------------------------------------------------------------------------------------

def _negative_graph():
    # 0 -> 1 (5), 0 -> 2 (6), 2 -> 1 (-4), 1 -> 3 (1): über 2 kostet Knoten 1 nur 2 statt 5, Dijkstra legt 1 aber schon mit 5 fest
    return from_arcs(4, [(0, 1, 5.0), (0, 2, 6.0), (2, 1, -4.0), (1, 3, 1.0)], np.zeros((4, 2)), directed=True)


def test_negative_edge_gives_a_wrong_answer_and_an_alarm():
    g = _negative_graph()
    bf = alg.bellman_ford_reference(g, 0)
    res = alg.dijkstra(g, [0], queue="binary")
    assert bf.dist[3] == 3.0 and bf.route(3) == [0, 2, 1, 3] and not bf.negative_cycle
    assert res.dist[3] == 6.0 and res.route(3) == [0, 1, 3]                                 # Dijkstra: zu teuer
    assert res.alarms == [(1, 5.0, 2.0)]


@pytest.mark.parametrize("queue", ["array", "binary", "lazy", "fibonacci"])
def test_negative_edges_alarm_is_the_same_for_every_queue(queue):
    res = alg.dijkstra(_negative_graph(), [0], queue=queue)
    assert res.alarms == [(1, 5.0, 2.0)]


@pytest.mark.parametrize("seed", range(6))
def test_bellman_ford_equals_networkx_with_negative_edges(seed):
    g = random_graph(40, 70, seed, directed=True, negative=0.15)
    G = to_nx(g)
    try:
        ref = nx.single_source_bellman_ford_path_length(G, 0)
    except nx.NetworkXUnbounded:
        assert alg.bellman_ford_reference(g, 0).negative_cycle
        return
    bf = alg.bellman_ford_reference(g, 0)
    assert not bf.negative_cycle
    for v in range(g.n):
        assert (bf.dist[v] == pytest.approx(ref[v])) if v in ref else not np.isfinite(bf.dist[v])


def test_bellman_ford_detects_a_negative_cycle():
    g = from_arcs(3, [(0, 1, 1.0), (1, 2, -3.0), (2, 1, 1.0)], np.zeros((3, 2)), directed=True)
    assert alg.bellman_ford_reference(g, 0).negative_cycle


def test_no_alarm_without_negative_edges():
    for seed in range(5):
        assert not alg.dijkstra(random_graph(60, 150, seed, integer=False), [0]).alarms


# --- Vergleich mit BFS und Rohdaten -------------------------------------------------------------------------------------------------------------

def test_with_equal_costs_dijkstra_settles_exactly_the_bfs_layers():
    rng = np.random.default_rng(3)
    g = from_arcs(80, [(int(u), int(v), 1.0) for u, v in rng.integers(0, 80, (200, 2))], np.zeros((80, 2)))
    d, b = alg.dijkstra(g, [0]), alg.bfs(g, [0])
    for k in range(int(b.dist.max()) + 1):
        assert {v for v in d.order if d.dist[v] == k} == {v for v in range(g.n) if b.dist[v] == k}


def test_bfs_route_can_cost_more_than_the_dijkstra_route():
    g = from_arcs(4, [(0, 3, 10.0), (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], np.zeros((4, 2)))
    b, d = alg.bfs(g, [0], [3]).route(3), alg.dijkstra(g, [0], [3]).route(3)
    assert b == [0, 3] and route_cost(g, b) == 10.0 and d == [0, 1, 2, 3] and route_cost(g, d) == 3.0


def test_raw_parallel_edges_are_handled_by_the_multigraph_but_not_by_a_naive_dictionary():
    raw = [(0, 1, 5.0), (0, 1, 1.0), (1, 2, 2.0), (1, 1, 1.0)]
    g_raw = from_arcs(3, raw, np.zeros((3, 2)), directed=True, clean=False)
    g_clean = from_arcs(3, raw, np.zeros((3, 2)), directed=True, clean=True)
    assert g_raw.m == 4 and g_clean.m == 2 and alg.dijkstra(g_raw, [0], [2]).dist[2] == 3.0 == alg.dijkstra(g_clean, [0], [2]).dist[2]
    last_wins, first_wins = {}, {}
    for u, v, c in raw:
        last_wins.setdefault(u, {})[v] = c
        first_wins.setdefault(u, {}).setdefault(v, c)
    assert alg.dijkstra_dict(last_wins, 0, 2) == 3.0 and alg.dijkstra_dict(first_wins, 0, 2) == 7.0     # je nach Reihenfolge der Rohdaten: 3 oder 7
