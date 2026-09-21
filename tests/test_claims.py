"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier belegt (Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft; Kosten der erzeugten Netze ±1 %, Zähler ±3 %).
Positive UND negative Aussagen: wo Dijkstra gewinnt (Kosten korrekt, Labyrinth = Breitensuche), steht hier ebenso wie dort, wo es verliert (negative Kanten, ganzes Netz festlegen)."""

from functools import lru_cache

import numpy as np
import pytest

import dj_algorithm as alg
import dj_constants as C
import dj_evaluation as ev
import dj_scenario as sc

TOL = 0.02
CITY = dict(side=C.DEFAULT_SIDE, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, blocked=C.DEFAULT_BLOCKED)


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


def rel(value, expected, tol=0.01):
    assert abs(value - expected) <= tol * abs(expected), f"{value} statt {expected}"


@lru_cache(maxsize=None)
def analysis(key):
    return ev.analyse(sc.make_network(key))


@lru_cache(maxsize=None)
def sweep(parameter, values):
    return ev.city_sweep(parameter, values, CITY)


@lru_cache(maxsize=None)
def pairs(key):
    return ev.pair_stats(sc.make_network(key), C.PAIRS, C.DEFAULT_SEED)


# --- Preset-Hilfen ------------------------------------------------------------------------------------------------------------------------

def test_delivery_preset_numbers():
    m = analysis("delivery").metrics
    assert (m["cost_bfs"], m["cost_dijkstra"], m["hops_dijkstra"], m["hops_bfs"]) == (12, 9, 3, 2)                 # 12 Minuten gegen 9, über drei Strecken
    assert m["cost_greedy"] == 30


def test_exchange_preset_numbers():
    m = analysis("exchange").metrics
    assert (m["cost_dijkstra"], m["cost_bellman_ford"], m["alarms"]) == (16, 13, 1)
    near(m["overpay"], 0.23)                                                                                        # Grenzen-Tabelle: 23 % zu viel


def test_city_preset_numbers():
    m = analysis("city").metrics
    rel(m["cost_dijkstra"], 3186)
    near(m["detour_bfs"], 0.27)
    assert m["settled"] == m["n"] == 400


def test_toronto_preset_and_limits_numbers():
    m = analysis("toronto").metrics
    rel(m["cost_dijkstra"], 891)
    rel(m["cost_bfs"], 1129)
    near(m["detour_bfs"], 0.27)
    assert m["n"] == 5072 and abs(m["settled"] - 2513) <= 10 and 0.45 < m["settled_share"] < 0.55                 # "rund die Hälfte", 2 513 von 5 072
    assert m["hops_dijkstra"] == 57 and m["hops_bfs"] == 29


def test_maze_preset_is_the_breadth_first_search():
    m = analysis("maze").metrics
    assert m["hops_dijkstra"] == m["hops_bfs"] == 42 and m["cost_dijkstra"] == m["cost_bfs"] and m["detour_bfs"] == 0
    for seed in C.SWEEP_SEEDS:
        net = sc.make_network("maze", 20, 2.3, 1.0, 20, 35, seed)
        d, b = alg.dijkstra(net.graph, net.sources), alg.bfs(net.graph, net.sources)
        for k in range(int(b.dist.max()) + 1):
            assert {v for v in d.order if d.dist[v] == k} == {v for v in range(net.graph.n) if b.dist[v] == k}


# --- Seitenleiste --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,detour,share", [(6, 0.10, 0.52), (10, 0.22, 0.48), (20, 0.28, 0.50), (30, 0.28, 0.46), (40, 0.30, 0.46)])
def test_side_sweep(value, detour, share):
    row = next(r for r in sweep("side", (6, 10, 20, 30, 40)) if r["value"] == value)
    near(row["detour_median"], detour)
    near(row["share_median"], share)


@pytest.mark.parametrize("value,detour", [(1.0, 0.07), (1.5, 0.18), (2.3, 0.28), (3.2, 0.34)])
def test_reach_sweep(value, detour):
    near(next(r for r in sweep("reach", (1.0, 1.5, 2.3, 3.2)) if r["value"] == value)["detour_median"], detour)


@pytest.mark.parametrize("value,detour", [(0.0, 0.11), (0.5, 0.18), (1.0, 0.28), (2.0, 0.47), (3.0, 0.62)])
def test_spread_sweep(value, detour):
    near(next(r for r in sweep("spread", (0.0, 0.5, 1.0, 2.0, 3.0)) if r["value"] == value)["detour_median"], detour)


@pytest.mark.parametrize("value,detour", [(0, 0.30), (20, 0.28), (40, 0.22), (60, 0.15)])
def test_blocked_sweep(value, detour):
    near(next(r for r in sweep("blocked", (0, 20, 40, 60)) if r["value"] == value)["detour_median"], detour)


def test_settled_share_stays_around_one_half_whatever_the_slider():
    for param, values in (("reach", (1.0, 1.5, 2.3, 3.2)), ("spread", (0.0, 0.5, 1.0, 2.0, 3.0)), ("blocked", (0, 20, 40, 60))):
        for r in sweep(param, values):
            assert 0.42 < r["share_median"] < 0.56, (param, r)


# --- Verteilung über Paare und Grenzen-Tabelle ---------------------------------------------------------------------------------------------

def test_random_pairs_settle_about_half_the_network():
    city, tor = pairs("city"), pairs("toronto")
    near(city["share_median"], 0.51)
    near(tor["share_median"], 0.51)
    assert city["share_p90"] > 0.85 and tor["share_p90"] > 0.85                                                     # "bei einem Zehntel der Paare über 85 %"
    assert city["detour_zero"] < 0.1 and tor["detour_zero"] < 0.1 and tor["detour_max"] > 1.0                      # Breitensuche liegt fast immer daneben, im schlimmsten Fall mehr als das Doppelte


def test_bfs_has_a_detour_over_random_pairs_while_dijkstra_has_none():
    ps = pairs("city")
    near(ps["detour_median"], 0.24)                                                                                 # README: 24 % (Seed 7); im Mittel über fünf Netze 28 % (Sweep oben)
    assert ps["detour_p90"] > ps["detour_median"]


def test_radius_growth_slope_is_about_two():
    near(ev.radius_growth()["slope"], 2.08, 0.06)
    for reach in (1.0, 1.5):
        assert 1.9 < ev.radius_growth(reach=reach)["slope"] < 2.2


# --- Warteschlangen ------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def queue_rows():
    return ev.queue_table()


def _slope(queue):
    sel = [r for r in queue_rows() if r["queue"] == queue]
    return float(np.polyfit(np.log([r["n"] for r in sel]), np.log([r["work"] for r in sel]), 1)[0])


def test_every_node_is_pushed_once_and_about_half_get_a_decrease_key():
    for r in queue_rows():
        assert r["pushes"] == r["n"]
        assert 0.35 < r["decrease_keys"] / r["n"] < 0.65


def test_fibonacci_needs_fewer_comparisons_than_the_binary_heap_and_dial_fewest_steps():
    top = {r["queue"]: r for r in queue_rows() if r["n"] == 3600}
    assert top["fibonacci"]["work"] < top["binary"]["work"] < top["array"]["work"] and top["dial"]["work"] < top["binary"]["work"]
    near(1 - top["fibonacci"]["work"] / top["binary"]["work"], 0.17, 0.05)                                       # README: rund 17 % weniger Vergleiche


def test_array_work_grows_much_faster_than_heap_work():
    assert 1.3 < _slope("array") < 1.8 and 1.0 < _slope("binary") < 1.35 and _slope("array") > _slope("binary") + 0.25


def test_lazy_heap_leaves_a_dead_entry_for_every_decrease_key():
    for r in queue_rows():
        if r["queue"] == "lazy":
            assert 0.9 * r["decrease_keys"] <= r["stale_pops"] <= r["decrease_keys"]


# --- Negative Kanten und Rohdaten ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def negative_rows():
    return {round(r["fraction"], 2): r for r in ev.negative_share()}


def test_dijkstra_is_always_right_without_negative_edges_and_often_wrong_with_few():
    rows = negative_rows()
    assert rows[0.0]["wrong"] == 0 and rows[0.0]["cycle"] == 0 and rows[0.0]["right"] == rows[0.0]["total"] == 200
    two = rows[0.02]
    assert 0.15 < two["wrong"] / two["total"] < 0.25                                                                 # "in etwa jedem fünften Netz"
    assert two["cycle"] / two["total"] < 0.1


def test_with_many_negative_edges_negative_cycles_dominate():
    rows = negative_rows()
    assert rows[0.2]["cycle"] / rows[0.2]["total"] > 0.5
    cycles = [rows[f]["cycle"] for f in (0.0, 0.02, 0.05, 0.1, 0.2)]
    assert cycles == sorted(cycles)


def test_raw_trap_numbers():
    rt = ev.raw_trap()
    assert (rt["multigraph"], rt["cleaned"], rt["dict_last_wins"], rt["dict_first_wins"]) == (3.0, 3.0, 3.0, 7.0)
