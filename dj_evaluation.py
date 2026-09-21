"""Läufe, Kennzahlen, Verteilung über zufällige Start-Ziel-Paare, Experimente (Suchfläche, Warteschlangen, negative Kanten, Rohdaten) und das Urteil für die App."""

import time
from dataclasses import dataclass

import numpy as np

import dj_algorithm as alg
import dj_constants as C
from dj_graph import from_arcs, route_cost
from dj_scenario import RAW_TRAP_ARCS, make_network

@dataclass(frozen=True)
class Analysis:
    net: object
    result: alg.Dijkstra           # Dijkstra mit Abbruch beim Ziel
    bfs: alg.Bfs
    routes: dict                   # "dijkstra", "bfs", "greedy", "bellman_ford" -> Knotenfolge
    metrics: dict
    seconds: dict


def _cost(net, route):
    return route_cost(net.graph, route) if route else float("nan")


def analyse(net, queue="binary"):
    """Dijkstra für Start und Ziel des Netzes, dazu BFS (Kanten zählen), bei kleinen Netzen die Greedy-Gegenprobe und bei negativen Kanten die Bellman-Ford-Referenz."""
    g, S, T = net.graph, net.sources, net.targets
    small = bool(g.names)                                  # kleine, benannte Netze: jede Festlegung wird protokolliert (Tabelle in der App), dazu die Greedy-Gegenprobe
    t0 = time.perf_counter()
    d = alg.dijkstra(g, S, T, queue=queue, trace=small)
    t_dij = time.perf_counter() - t0
    b = alg.bfs(g, S, T)
    reachable = d.found >= 0
    routes = {"dijkstra": d.route(d.found) if reachable else [], "bfs": b.route(b.found) if b.found >= 0 else [], "greedy": [], "bellman_ford": []}
    if small and reachable:
        routes["greedy"] = alg.greedy_route(g, S[0], T[0])
    bf = None
    if net.negative:
        bf = alg.bellman_ford_reference(g, S[0])
        routes["bellman_ford"] = bf.route(T[0])
    cost = {k: _cost(net, r) for k, r in routes.items()}
    hops = {k: len(r) - 1 if r else 0 for k, r in routes.items()}
    detour = cost["bfs"] / cost["dijkstra"] - 1.0 if reachable and routes["bfs"] and cost["dijkstra"] > 0 else float("nan")
    greedy_detour = cost["greedy"] / cost["dijkstra"] - 1.0 if routes["greedy"] and cost["dijkstra"] > 0 else float("nan")
    overpay = float("nan")
    if bf is not None and routes["bellman_ford"] and cost["bellman_ford"] != 0:
        overpay = cost["dijkstra"] / cost["bellman_ford"] - 1.0
    metrics = {"reachable": reachable, "n": g.n, "m": g.m, "cost_dijkstra": cost["dijkstra"], "hops_dijkstra": hops["dijkstra"], "cost_bfs": cost["bfs"], "hops_bfs": hops["bfs"],
               "detour_bfs": detour, "cost_greedy": cost["greedy"], "hops_greedy": hops["greedy"], "greedy_detour": greedy_detour, "cost_bellman_ford": cost["bellman_ford"],
               "overpay": overpay, "settled": len(d.order), "settled_share": len(d.order) / g.n, "discovered_bfs": len(b.discovered), "front_max": max(d.front) if d.front else 0,
               "alarms": len(d.alarms), "counters": d.counters, "negative": net.negative}
    return Analysis(net, d, b, routes, metrics, {"dijkstra": t_dij})


# --- Verteilung über zufällige Start-Ziel-Paare ----------------------------------------------------------------------------------------------

def pair_stats(net, pairs=C.PAIRS, seed=0):
    """Über zufällige erreichbare Start-Ziel-Paare: welcher Anteil des Netzes wird festgelegt, bevor das Ziel gefunden ist, und wie weit liegt die Kanten-Route (BFS) daneben.
    Nur für Netze ohne negative Kanten."""
    rng = np.random.default_rng([int(seed), 505])
    g = net.graph
    share, detour = [], []
    tries = 0
    while len(share) < pairs and tries < pairs * 20:
        tries += 1
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s == t:
            continue
        d = alg.dijkstra(g, (s,), (t,), queue="lazy")
        b = alg.bfs(g, (s,), (t,))
        if d.found < 0 or b.found < 0 or d.dist[t] <= 0:
            continue
        share.append(len(d.order) / g.n)
        detour.append(route_cost(g, b.route(t)) / d.dist[t] - 1.0)
    sh, dt = np.array(share), np.array(detour)
    empty = float("nan")
    if not len(sh):
        return {"share": sh, "detour": dt, "n_pairs": 0, "share_median": empty, "share_p90": empty, "share_max": empty, "detour_median": empty, "detour_p90": empty, "detour_max": empty,
                "detour_zero": empty}
    return {"share": sh, "detour": dt, "n_pairs": len(sh), "share_median": float(np.median(sh)), "share_p90": float(np.quantile(sh, 0.9)), "share_max": float(sh.max()),
            "detour_median": float(np.median(dt)), "detour_p90": float(np.quantile(dt, 0.9)), "detour_max": float(dt.max()), "detour_zero": float((dt < 1e-9).mean())}


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

def radius_growth(side=60, reach=2.3, seed=C.SWEEP_SEEDS[0]):
    """Blind in alle Richtungen: von der Mitte eines Stadtnetzes aus legt Dijkstra alle Knoten bis zum Kostenradius L fest - wie viele sind das je L? Log-Log-Steigung im Inneren
    (Radius zwischen dem 2. und 25. Prozentil der Entfernungen, damit der Rand des Netzes noch nicht mitzählt); im idealen Kreis wäre die Fläche ~ L²."""
    g = make_network("city", side, reach, C.DEFAULT_SPREAD, 0, C.DEFAULT_WALLS, seed).graph
    center = (side // 2) * side + side // 2
    d = np.sort(alg.dijkstra(g, [center], queue="lazy").dist)
    d = d[np.isfinite(d)]
    radii = np.linspace(d[int(len(d) * 0.02)], d[int(len(d) * 0.25)], 12)
    counts = np.array([(d <= r).sum() for r in radii])
    slope = float(np.polyfit(np.log(radii), np.log(counts), 1)[0])
    all_r = np.linspace(d[1], d[-1], 40)
    return {"side": side, "n": g.n, "slope": slope, "radii": all_r, "counts": np.array([(d <= r).sum() for r in all_r])}


def queue_table(sides=(10, 20, 40, 60), reach=1.5, seed=5):
    """Alle fünf Warteschlangen auf denselben Stadtnetzen: Zähler (deterministisch) und Laufzeit (nur Messwert dieses Laufs)."""
    rows = []
    for side in sides:
        g = make_network("city", side, reach, C.DEFAULT_SPREAD, 0, C.DEFAULT_WALLS, seed).graph
        base = None
        for q in ("array", "binary", "lazy", "dial", "fibonacci"):
            best = float("inf")
            for _ in range(2):
                t0 = time.perf_counter()
                r = alg.dijkstra(g, [0], queue=q)
                best = min(best, time.perf_counter() - t0)
            base = r.dist if base is None else base
            assert np.array_equal(base, r.dist)
            c = r.counters
            rows.append({"side": side, "n": g.n, "m": g.m, "queue": q, "pushes": c["pushes"], "decrease_keys": c["decrease_keys"], "work": c["work"], "stale_pops": c.get("stale_pops", 0),
                         "ms": best * 1000})
    return rows


def negative_share(fractions=(0.0, 0.02, 0.05, 0.1, 0.2), n=40, m=120, trials=40, seeds=C.SWEEP_SEEDS):
    """Zufällige gerichtete Netze mit einem Anteil negativer Kanten: in wie vielen Instanzen liefert Dijkstra falsche Entfernungen (gegen Bellman-Ford), in wie vielen stimmt es zufällig,
    und in wie vielen gibt es einen negativen Zyklus (dann existiert gar keine kürzeste Route)?"""
    rows = []
    for frac in fractions:
        wrong = right = cycle = 0
        for sd in seeds:
            rng = np.random.default_rng([int(sd), 606])
            for _ in range(trials):
                arcs = []
                for _ in range(m):
                    u, v = rng.integers(0, n, 2)
                    w = -float(rng.integers(1, 5)) if rng.random() < frac else float(rng.integers(1, 10))
                    arcs.append((u, v, w))
                g = from_arcs(n, arcs, np.zeros((n, 2)), directed=True)
                bf = alg.bellman_ford_reference(g, 0)
                if bf.negative_cycle:
                    cycle += 1
                    continue
                d = alg.dijkstra(g, [0])
                fin = np.isfinite(bf.dist)
                if np.any(np.abs(d.dist[fin] - bf.dist[fin]) > 1e-9):
                    wrong += 1
                else:
                    right += 1
        total = wrong + right + cycle
        rows.append({"fraction": frac, "wrong": wrong, "right": right, "cycle": cycle, "total": total})
    return rows


def raw_trap():
    """Rohdaten mit Parallelkanten und Selbstschleife: dasselbe Netz, vier Umsetzungen. Kosten von A nach C."""
    arcs = RAW_TRAP_ARCS
    xy = np.zeros((3, 2))
    last, first = {}, {}
    for u, v, c in arcs:
        last.setdefault(u, {})[v] = c
        first.setdefault(u, {}).setdefault(v, c)
    raw = from_arcs(3, arcs, xy, directed=True, clean=False)
    clean = from_arcs(3, arcs, xy, directed=True, clean=True)
    return {"raw_arcs": arcs, "multigraph": alg.dijkstra(raw, [0], [2]).dist[2], "cleaned": alg.dijkstra(clean, [0], [2]).dist[2],
            "dict_last_wins": alg.dijkstra_dict(last, 0, 2), "dict_first_wins": alg.dijkstra_dict(first, 0, 2), "arcs_raw": raw.m, "arcs_clean": clean.m}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

def verdict(a):
    """Code für die App: unreachable / negative_wrong / negative_ok / unweighted / same_route / bfs_worse."""
    m = a.metrics
    if not m["reachable"]:
        return "unreachable"
    if m["negative"]:
        return "negative_wrong" if (m["alarms"] > 0 and m["overpay"] > 1e-9) else "negative_ok"
    if not a.net.weighted:
        return "unweighted"
    return "same_route" if m["detour_bfs"] < 1e-9 else "bfs_worse"


# --- Zustand für die Tabelle bei kleinen Netzen ------------------------------------------------------------------------------------------------

def table_state(a, k):
    """Zustand nach `k` Festlegungen (nur mit Protokoll, also bei kleinen Netzen): je Knoten Entfernung, Vorgänger, Status; welche Zellen sich in der letzten Festlegung geändert haben;
    Alarme bis dahin (ein bereits festgelegter Knoten hätte eine bessere Route gehabt)."""
    net, res = a.net, a.result
    g = net.graph
    dist = {int(s): 0.0 for s in net.sources}
    parent, changed = {}, set()
    for u, d, updates in res.events[:k]:
        changed = set()
        for v, old, new in updates:
            dist[v], parent[v] = new, u
            changed.add(v)
    settled = set(res.order[:k])
    last = res.order[k - 1] if k > 0 else None
    rows = []
    for v in range(g.n):
        if v in settled:
            status = "gerade festgelegt" if v == last else "festgelegt"
        elif v in dist:
            status = "in der Warteschlange"
        else:
            status = "noch unbekannt"
        rows.append({"Knoten": g.names[v] if g.names else str(v), "Entfernung": dist.get(v), "Vorgänger": g.names[parent[v]] if v in parent else "–", "Status": status, "geändert": v in changed})
    alarms = [(g.names[v], old, new) for (v, old, new), st in zip(res.alarms, res.alarm_steps) if st <= k]
    queue = sorted(((dist[v], g.names[v]) for v in dist if v not in settled))
    return {"rows": rows, "alarms": alarms, "queue": queue}


def city_sweep(parameter, values, base, pairs=40, seeds=C.SWEEP_SEEDS):
    """Ein Regler des Stadtnetzes durchgefahren, alle anderen wie in `base` (dict mit side, reach, spread, blocked): festgelegter Anteil des Netzes (Median über Paare) und Umweg der
    Breitensuche-Route (Median), jeweils Mittel über die fünf festen Sweep-Datensätze (getrennt vom Seed der Seitenleiste)."""
    rows = []
    for v in values:
        kw = {**base, parameter: v}
        st = [pair_stats(make_network("city", kw["side"], kw["reach"], kw["spread"], kw["blocked"], C.DEFAULT_WALLS, sd), pairs, sd) for sd in seeds]
        rows.append({"value": v, "share_median": float(np.mean([x["share_median"] for x in st])), "detour_median": float(np.mean([x["detour_median"] for x in st])),
                     "share_p90": float(np.mean([x["share_p90"] for x in st]))})
    return rows
