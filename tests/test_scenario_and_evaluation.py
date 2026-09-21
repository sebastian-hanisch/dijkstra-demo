"""Netze (Lieferwege, Tausch, Stadtnetz, Labyrinth, Toronto), Kennzahlen, Verteilung über Paare, Experimente, Tabellenzustand."""

import numpy as np
import pytest

import dj_algorithm as alg
import dj_constants as C
import dj_evaluation as ev
import dj_scenario as sc
from dj_graph import route_cost


def _connected(g):
    return (alg.bfs(g, [0]).dist >= 0).all()


# --- Stadtnetz und Labyrinth -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("reach,blocked", [(1.0, 60), (2.3, 20), (3.2, 60)])
def test_city_is_connected_symmetric_with_whole_number_costs(seed, reach, blocked):
    g = sc.build_city(12, reach, 1.0, blocked, seed)
    assert g.n == 144 and _connected(g)
    assert (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight))                 # ganze Meter: alle Warteschlangen (auch Dial) laufen auf denselben Netzen
    for u in range(g.n):
        for v in g.out(u):
            assert g.weight[g.arc(u, int(v))] == g.weight[g.arc(int(v), u)]


def test_city_is_deterministic_per_seed():
    a, b, c = (sc.build_city(10, 2.3, 1.0, 20, s) for s in (1, 1, 2))
    assert np.array_equal(a.weight, b.weight) and not np.array_equal(a.weight, c.weight)


@pytest.mark.parametrize("side,walls,seed", [(6, 45, 0), (20, 35, 1), (40, 10, 2), (40, 45, 3)])
def test_maze_start_and_goal_are_connected(side, walls, seed):
    g, s, t = sc.build_maze(side, walls, seed)
    assert alg.bfs(g, [s], [t]).found == t and (g.weight == 1.0).all()


def test_toronto_data_is_the_expected_excerpt_with_whole_number_costs():
    net = sc.toronto_network()
    g = net.graph
    assert (g.n, g.m) == (5072, 14503) and g.directed and net.unit == "m" and not net.negative
    assert (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight))
    src = np.repeat(np.arange(g.n), g.degree())
    assert (src != g.indices).all() and len(set(zip(src.tolist(), g.indices.tolist()))) == g.m


# --- Lieferwege ------------------------------------------------------------------------------------------------------------------------------

def test_delivery_graph_matches_its_definition_and_is_undirected():
    net = sc.delivery_network()
    g = net.graph
    assert g.n == 7 and g.m == 2 * len(sc.DELIVERY_LINKS) and not g.directed and not g.has_negative()
    for a, b, w in sc.DELIVERY_LINKS:
        ia, ib = g.names.index(a), g.names.index(b)
        assert g.weight[g.arc(ia, ib)] == g.weight[g.arc(ib, ia)] == w


def test_delivery_fewest_edges_is_not_cheapest_and_greedy_is_far_off():
    a = ev.analyse(sc.delivery_network())
    g, r, m = a.net.graph, a.routes, a.metrics
    names = lambda route: [g.names[i] for i in route]
    assert names(r["dijkstra"]) == ["Depot", "Hafen", "Brücke", "Kunde"] and m["cost_dijkstra"] == 9
    assert names(r["bfs"]) == ["Depot", "Brücke", "Kunde"] and m["cost_bfs"] == 12 and m["hops_bfs"] == 2 < m["hops_dijkstra"] == 3
    assert names(r["greedy"]) == ["Depot", "Hafen", "Werk", "Fähre", "Kunde"] and m["cost_greedy"] == 30
    assert ev.verdict(a) == "bfs_worse"


def test_delivery_table_has_two_improvements_of_tentative_distances():
    a = ev.analyse(sc.delivery_network())
    improved = [(a.net.graph.names[v], old, new) for _, _, updates in a.result.events for v, old, new in updates if np.isfinite(old)]
    assert improved == [("Brücke", 9.0, 6.0), ("Kunde", 14.0, 9.0)]                        # zwei Verbesserungen bereits bekannter Entfernungen


# --- Tausch mit Rückzahlung -----------------------------------------------------------------------------------------------------------------

def test_exchange_has_one_negative_edge_and_dijkstra_pays_too_much():
    a = ev.analyse(sc.exchange_network())
    m, g = a.metrics, a.net.graph
    assert g.directed and g.has_negative() and int((g.weight < 0).sum()) == 1 and a.net.negative
    assert m["cost_dijkstra"] == 16 and m["cost_bellman_ford"] == 13 and m["alarms"] == 1
    assert m["overpay"] == pytest.approx(3 / 13) and ev.verdict(a) == "negative_wrong"
    assert [g.names[i] for i in a.routes["bellman_ford"]] == ["Fahrradlampe", "Werkzeugset", "Klingel", "Laufrad", "Rennrad"]
    assert a.result.alarms == [(g.names.index("Klingel"), 5.0, 2.0)] and not alg.bellman_ford_reference(g, 0).negative_cycle


def test_table_state_walks_through_the_steps_and_alarm_appears_at_step_three():
    a = ev.analyse(sc.exchange_network())
    n_steps = len(a.result.order)
    assert n_steps == 6
    for k in range(n_steps + 1):
        st = ev.table_state(a, k)
        assert len(st["rows"]) == 6 and sum(r["Status"].endswith("festgelegt") for r in st["rows"]) == k
        assert (len(st["alarms"]) == 1) == (k >= 3)
    first = ev.table_state(a, 1)
    assert [r["Entfernung"] for r in first["rows"]][:3] == [0.0, 5.0, 6.0] and first["queue"] == [(5.0, "Klingel"), (6.0, "Werkzeugset")]
    assert ev.table_state(a, n_steps)["queue"] == []


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
def test_analysis_invariants(key):
    a = ev.analyse(sc.make_network(key))
    m, net = a.metrics, a.net
    assert m["reachable"] and m["settled"] <= m["n"] and m["front_max"] >= 1 and a.routes["dijkstra"][0] in net.sources and a.routes["dijkstra"][-1] in net.targets
    assert m["counters"]["pops"] == m["settled"] == len(a.result.order)
    if not net.negative:
        assert m["cost_dijkstra"] <= m["cost_bfs"] + 1e-9 and m["detour_bfs"] >= -1e-12 and m["alarms"] == 0
        assert m["cost_dijkstra"] == pytest.approx(route_cost(net.graph, a.routes["dijkstra"]))
    if not net.weighted:
        assert m["detour_bfs"] == 0 and ev.verdict(a) == "unweighted"


def test_dijkstra_equals_the_networkx_distance_on_the_shown_pair():
    import networkx as nx
    for key in ("city", "toronto", "delivery"):
        net = sc.make_network(key)
        g = net.graph
        G = nx.DiGraph()
        for u in range(g.n):
            for v, w in zip(g.out(u), g.out_weights(u)):
                G.add_edge(u, int(v), weight=float(w))
        assert ev.analyse(net).metrics["cost_dijkstra"] == pytest.approx(nx.dijkstra_path_length(G, net.sources[0], net.targets[0]))


def test_all_queues_give_the_same_analysis_costs():
    net = sc.make_network("city", 12)
    costs = {q: ev.analyse(net, queue=q).metrics["cost_dijkstra"] for q in ("array", "binary", "lazy", "dial", "fibonacci")}
    assert len(set(costs.values())) == 1


def test_pair_stats_fields_and_determinism():
    net = sc.make_network("city")
    a, b = ev.pair_stats(net, 40, 3), ev.pair_stats(net, 40, 3)
    assert a["n_pairs"] == 40 and np.array_equal(a["share"], b["share"]) and not np.array_equal(a["share"], ev.pair_stats(net, 40, 4)["share"])
    assert 0 < a["share_median"] <= a["share_p90"] <= a["share_max"] <= 1 and (a["detour"] >= -1e-12).all()


def test_settled_share_of_random_pairs_averages_about_one_half():
    for key in ("city", "maze", "toronto"):
        ps = ev.pair_stats(sc.make_network(key), 150, 1)
        assert 0.4 < float(np.mean(ps["share"])) < 0.6, key                                 # Rangzahl eines zufälligen Ziels ist gleichverteilt


def test_unweighted_networks_have_no_detour_over_random_pairs():
    ps = ev.pair_stats(sc.make_network("maze"), 60, 1)
    assert ps["detour_zero"] == 1.0 and ps["detour_max"] == 0.0


def test_city_sweep_rows_carry_all_fields():
    rows = ev.city_sweep("spread", (0.0, 2.0), dict(side=8, reach=2.3, spread=1.0, blocked=20), pairs=10, seeds=C.SWEEP_SEEDS[:2])
    assert [r["value"] for r in rows] == [0.0, 2.0] and all({"share_median", "detour_median", "share_p90"} <= set(r) for r in rows) and rows[0]["detour_median"] < rows[1]["detour_median"]


# --- Experimente -----------------------------------------------------------------------------------------------------------------------------

def test_radius_growth_has_slope_about_two_and_is_deterministic():
    a, b = ev.radius_growth(side=40), ev.radius_growth(side=40)
    assert a["slope"] == b["slope"] and 1.8 < a["slope"] < 2.3 and a["counts"][-1] <= a["n"] and list(a["counts"]) == sorted(a["counts"])


def test_queue_table_rows_cover_all_queues_and_sizes_with_equal_distances():
    rows = ev.queue_table(sides=(8, 12))
    assert len(rows) == 10 and {r["queue"] for r in rows} == {"array", "binary", "lazy", "dial", "fibonacci"}
    for r in rows:
        assert r["pushes"] == r["n"] and r["decrease_keys"] > 0
    small = {r["queue"]: r for r in rows if r["side"] == 12}
    assert small["array"]["work"] > small["binary"]["work"] and small["lazy"]["work"] == 0 and small["lazy"]["stale_pops"] > 0


def test_negative_share_has_no_errors_without_negative_edges_and_grows_with_the_share():
    rows = ev.negative_share(fractions=(0.0, 0.05), trials=15, seeds=C.SWEEP_SEEDS[:2])
    assert rows[0]["wrong"] == 0 and rows[0]["cycle"] == 0 and rows[0]["right"] == rows[0]["total"] == 30
    assert rows[1]["wrong"] > 0 and rows[1]["wrong"] + rows[1]["right"] + rows[1]["cycle"] == 30


def test_raw_trap_depends_on_the_implementation():
    rt = ev.raw_trap()
    assert rt["multigraph"] == rt["cleaned"] == rt["dict_last_wins"] == 3.0 and rt["dict_first_wins"] == 7.0 and (rt["arcs_raw"], rt["arcs_clean"]) == (4, 2)
