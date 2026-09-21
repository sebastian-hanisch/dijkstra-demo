"""Dijkstra mit austauschbarer Warteschlange, Bellman-Ford als Messreferenz für negative Kanten, Breitensuche als Vergleich (Stück 1 der Linie), naive Wörterbuch-Variante für die Rohdaten-Falle.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus dj_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from dj_graph import route_cost
from dj_queues import QUEUES

INF = float("inf")


@dataclass
class Dijkstra:
    dist: np.ndarray                          # Kosten vom nächsten Start; bei festgelegten Knoten endgültig, sonst der letzte vorläufige Wert (inf = nie erreicht)
    parent: np.ndarray                        # Vorgänger, -1 = Start oder nicht erreicht
    order: list = field(default_factory=list)          # festgelegte Knoten in der Reihenfolge ihrer Festlegung
    disc_step: np.ndarray = None              # bei welcher Festlegung (1-basiert) der Knoten zum ersten Mal einen endlichen Wert bekam (0 = Start, -1 = nie)
    front: list = field(default_factory=list)          # Warteschlangengröße nach jeder Festlegung
    found: int = -1                           # zuerst festgelegtes Ziel
    alarms: list = field(default_factory=list)         # (Knoten, festgelegter Wert, besserer Wert): ein festgelegter Knoten hätte verbessert werden können - nur mit negativen Kanten
    counters: dict = field(default_factory=dict)       # pushes, decrease_keys, pops, work (siehe dj_queues) plus relaxations
    events: list = field(default_factory=list)         # nur mit trace=True: je Festlegung (Knoten, Wert, [(Nachbar, alt, neu)])
    alarm_steps: list = field(default_factory=list)    # bei welcher Festlegung (1-basiert) der jeweilige Alarm auftrat

    def settled_mask(self):
        m = np.zeros(len(self.dist), dtype=bool)
        m[self.order] = True
        return m

    def route(self, target):
        if not np.isfinite(self.dist[target]):
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def dijkstra(g, sources, targets=(), queue="lazy", trace=False):
    """Dijkstra ab einem oder mehreren Starts, Abbruch, sobald ein Ziel festgelegt wird. Ein Knoten wird genau einmal festgelegt; findet sich danach eine bessere Route zu ihm
    (nur mit negativen Kanten möglich), wird das als Alarm gezählt und NICHT nachgebessert - so verhält sich das Verfahren, wie es gelehrt wird, und so falsch wird sein Ergebnis."""
    n = g.n
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    max_w = int(max(w)) if (w and queue == "dial") else 1
    q = QUEUES[queue](n, max_w)
    targets = set(int(t) for t in targets)
    dist, parent = [INF] * n, [-1] * n
    settled = [False] * n
    disc = [-1] * n
    for s in sources:
        s = int(s)
        dist[s] = 0.0 if queue != "dial" else 0
        disc[s] = 0
        q.push_or_decrease(s, dist[s])
    order, front, alarms, alarm_steps, events = [], [], [], [], []
    found, relaxations = -1, 0
    while len(q):
        d, u = q.pop_min()
        if settled[u]:
            continue
        settled[u] = True
        order.append(u)
        step = len(order)
        front.append(len(q))
        if u in targets:
            found = u
            if trace:
                events.append((u, d, []))
            break
        updates = []
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            relaxations += 1
            nd = d + w[k]
            if nd < dist[v]:
                if settled[v]:
                    alarms.append((v, dist[v], nd))
                    alarm_steps.append(step)
                    continue
                if trace:
                    updates.append((v, dist[v], nd))
                dist[v], parent[v] = nd, u
                if disc[v] < 0:
                    disc[v] = step
                q.push_or_decrease(v, nd)
        if trace:
            events.append((u, d, updates))
    counters = q.as_dict()
    counters["relaxations"] = relaxations
    if hasattr(q, "stale_pops"):
        counters["stale_pops"] = q.stale_pops
    return Dijkstra(np.array(dist, dtype=float), np.array(parent), order, np.array(disc), front, found, alarms, counters, events, alarm_steps)


@dataclass
class BellmanFord:
    dist: np.ndarray
    parent: np.ndarray
    negative_cycle: bool
    rounds: int

    def route(self, target):
        if not np.isfinite(self.dist[target]) or self.negative_cycle:
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def bellman_ford_reference(g, source):
    """Messreferenz für Netze mit negativen Kanten: bis zu n - 1 Runden über alle Kanten (Ford 1956, Bellman 1958); eine weitere Verbesserung in Runde n zeigt einen negativen Zyklus an.
    Das Verfahren selbst ist ein späteres Stück der Linie - hier nur, um zu wissen, was Dijkstra hätte finden müssen."""
    src = np.repeat(np.arange(g.n), g.degree()).tolist()
    dst, w = g.indices.tolist(), g.weight.tolist()
    dist, parent = [INF] * g.n, [-1] * g.n
    dist[int(source)] = 0.0
    rounds = 0
    for rounds in range(1, g.n + 1):
        changed = False
        for u, v, c in zip(src, dst, w):
            if dist[u] + c < dist[v]:
                dist[v], parent[v] = dist[u] + c, u
                changed = True
        if not changed:
            return BellmanFord(np.array(dist), np.array(parent), False, rounds)
    return BellmanFord(np.array(dist), np.array(parent), True, rounds)


@dataclass
class Bfs:
    dist: np.ndarray
    parent: np.ndarray
    discovered: list = field(default_factory=list)
    found: int = -1

    def route(self, target):
        if self.dist[target] < 0:
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def bfs(g, sources, targets=()):
    """Breitensuche (Stück 1 der Linie) als Vergleich: die Route mit den wenigsten Kanten, Abbruch beim ersten entdeckten Ziel."""
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    dist, parent = [-1] * g.n, [-1] * g.n
    targets = set(int(t) for t in targets)
    queue, discovered, found = deque(), [], -1
    for s in sources:
        s = int(s)
        if dist[s] < 0:
            dist[s] = 0
            queue.append(s)
            discovered.append(s)
            if found < 0 and s in targets:
                found = s
    while queue and found < 0:
        u = queue.popleft()
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            if dist[v] < 0:
                dist[v], parent[v] = dist[u] + 1, u
                queue.append(v)
                discovered.append(v)
                if v in targets:
                    found = v
                    break
    return Bfs(np.array(dist), np.array(parent), discovered, found)


def dijkstra_dict(adjacency, source, target):
    """Naive Wörterbuch-Variante (adjacency[u][v] = Kosten, wie in vielen Lehrbuch-Umsetzungen): kennt nur EINE Kante je Knotenpaar. Nur für die Rohdaten-Falle."""
    import heapq
    dist = {source: 0.0}
    heap = [(0.0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == target:
            return d
        for v, c in adjacency.get(u, {}).items():
            nd = d + c
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return INF


def route_summary(g, route):
    """(Kanten, Kosten) einer Route; (0, 0.0) bei leerer Route."""
    if not route:
        return 0, 0.0
    return len(route) - 1, route_cost(g, route)


def greedy_route(g, source, target):
    """'Immer die billigste noch unbesuchte Kante nehmen' - der naheliegende Ansatz, der in die Irre läuft (Gegenprobe): Route oder [] bei Sackgasse."""
    route, seen = [int(source)], {int(source)}
    u = int(source)
    while u != int(target):
        options = [(w, v) for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()) if v not in seen]
        if not options:
            return []
        _, v = min(options)
        route.append(v)
        seen.add(v)
        u = v
    return route
