"""Die Netze der Demo: kleines Lieferwegenetz und Tauschnetz mit Rückzahlung (eigene Graphen in der Art der Lehrbuch-Beispiele), erzeugtes Stadtnetz, Labyrinth und Toronto Campus (echte OpenStreetMap-Daten).
Alle Kosten sind ganze Zahlen (Meter, Euro): so laufen alle Warteschlangen, auch die Dial-Eimer, auf denselben Netzen."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import dj_constants as C
from dj_graph import Graph, from_arcs

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    sources: tuple                 # Startknoten (bei mehreren: der nächste zählt)
    targets: tuple                 # Zielknoten (bei mehreren: der nächste zählt)
    weighted: bool                 # False: alle Kanten kosten dasselbe, Kanten zählen = Kosten zählen
    unit: str                      # Einheit der Kosten
    title: str
    start_label: str
    goal_label: str
    note: str = ""
    side: int = 0                  # Kantenlänge des Rasters (nur erzeugte Netze)
    negative: bool = False         # enthält Kanten mit negativen Kosten


# --- Stadtnetz ---------------------------------------------------------------------------------------------------------------------------

def _primitive_offsets(reach):
    """Verbindungen (dy, dx) auf dem Raster mit Länge <= reach, nur die primitiven (keine Verbindung überspringt einen Knoten, der auf ihr liegt), nur eine Richtung je Paar."""
    r = int(math.floor(reach + 1e-9))
    out = []
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if (dy == 0 and dx <= 0) or dx * dx + dy * dy > reach * reach + 1e-9 or math.gcd(abs(dx), dy) != 1:
                continue
            out.append((dy, dx))
    return out


def build_city(side, reach, spread, blocked_pct, seed):
    """Gestörtes Raster: Kreuzungen im Abstand SPACING mit Lageabweichung, verbunden mit allen Nachbarn bis zur Reichweite `reach` (in Blocklängen). Kosten einer Straße = ihre Länge in Metern
    mal (1 + spread * Zufall): Ampeln, Steigung, Belag. Ein Teil der Straßen ist gesperrt, das Netz bleibt aber zusammenhängend (Spannbaum bleibt geschützt)."""
    rng = np.random.default_rng([int(seed), 101])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)                      # (Zeile, Spalte)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    pairs = []
    for dy, dx in _primitive_offsets(reach):
        for i in range(side):
            for j in range(side):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < side and 0 <= j2 < side:
                    pairs.append((i * side + j, i2 * side + j2))
    pairs = np.array(pairs)
    length = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    cost = np.maximum(1.0, np.rint(length * (1.0 + spread * rng.random(len(pairs)))))
    order = rng.permutation(len(pairs))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    in_tree = np.zeros(len(pairs), dtype=bool)
    for k in order:
        a, b = find(pairs[k, 0]), find(pairs[k, 1])
        if a != b:
            parent[a] = b
            in_tree[k] = True
    removable = [k for k in order if not in_tree[k]]
    drop = set(removable[: int(round(len(pairs) * blocked_pct / 100.0))])
    arcs = [(pairs[k, 0], pairs[k, 1], cost[k]) for k in range(len(pairs)) if k not in drop]
    return from_arcs(n, arcs, xy)


def city_network(side, reach, spread, blocked_pct, seed):
    g = build_city(side, reach, spread, blocked_pct, seed)
    return Network("city", g, (0,), (g.n - 1,), True, "m", "Stadtnetz", "unten links", "oben rechts",
                   "Erzeugtes Stadtnetz: Kreuzungen auf einem gestörten Raster; die Reichweite bestimmt, wie weit eine Straße zwischen zwei Kreuzungen reichen darf.", side)


# --- Labyrinth ---------------------------------------------------------------------------------------------------------------------------

def _components(open_, side):
    label = -np.ones((side, side), dtype=int)
    count = 0
    for i in range(side):
        for j in range(side):
            if open_[i, j] and label[i, j] < 0:
                label[i, j] = count
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for a2, b2 in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= a2 < side and 0 <= b2 < side and open_[a2, b2] and label[a2, b2] < 0:
                            label[a2, b2] = count
                            stack.append((a2, b2))
                count += 1
    return label


def build_maze(side, walls_pct, seed):
    """Quadratisches Raster mit zufälligen Wänden; Start unten links, Ziel oben rechts, garantiert verbunden (falls nötig werden zufällige Wände geöffnet, die zwei getrennte Gebiete berühren)."""
    rng = np.random.default_rng([int(seed), 202])
    open_ = rng.random((side, side)) >= walls_pct / 100.0
    open_[0, 0] = open_[side - 1, side - 1] = True
    while True:
        label = _components(open_, side)
        if label[0, 0] == label[side - 1, side - 1]:
            break
        cand, near_start = [], []
        for i in range(side):
            for j in range(side):
                if open_[i, j]:
                    continue
                labs = {label[a, b] for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)) if 0 <= a < side and 0 <= b < side and label[a, b] >= 0}
                if len(labs) >= 2:
                    cand.append((i, j))
                if label[0, 0] in labs:
                    near_start.append((i, j))
        pool = cand or near_start
        i, j = pool[int(rng.integers(len(pool)))]
        open_[i, j] = True
    node = -np.ones((side, side), dtype=int)
    cells = np.argwhere(open_)
    node[open_] = np.arange(len(cells))
    arcs = []
    for (i, j) in cells:
        for i2, j2 in ((i + 1, j), (i, j + 1)):
            if i2 < side and j2 < side and open_[i2, j2]:
                arcs.append((node[i, j], node[i2, j2], 1.0))
    xy = np.stack([cells[:, 1], cells[:, 0]], axis=1).astype(float)
    return from_arcs(len(cells), arcs, xy), int(node[0, 0]), int(node[side - 1, side - 1])


def maze_network(side, walls_pct, seed):
    g, s, t = build_maze(side, walls_pct, seed)
    return Network("maze", g, (s,), (t,), False, "Schritte", "Labyrinth", "unten links", "oben rechts",
                   "Erzeugtes Labyrinth: jeder Schritt zu einer freien Nachbarzelle kostet dasselbe - hier legt Dijkstra genau die Schichten der Breitensuche fest.", side)


# --- Toronto Campus ----------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _toronto_payload():
    return json.loads((DATA / "toronto_campus.json").read_text(encoding="utf-8"))


def toronto_network():
    p = _toronto_payload()
    xy = np.array(p["nodes_xy_m"], dtype=float)
    g = from_arcs(len(xy), [(u, v, max(1.0, round(w))) for u, v, w in p["arcs"]], xy, directed=True)
    return Network("toronto", g, (p["start"]["node"],), (p["goal"]["node"],), True, "m", "Toronto Campus", p["start"]["label"], p["goal"]["label"],
                   "Straßen- und Wegenetz um den St. George Campus der University of Toronto, 1 km Umkreis; Einbahnstraßen sind gerichtete Kanten.")


# --- Eigene Lehrbuch-artige Netze ------------------------------------------------------------------------------------------------------------

DELIVERY_STOPS = [("Depot", 0.0, 2.0), ("Hafen", 2.0, 3.4), ("Brücke", 4.0, 1.4), ("Tunnel", 2.4, 0.2), ("Werk", 4.2, 4.2), ("Fähre", 6.2, 4.2), ("Kunde", 6.6, 1.6)]
DELIVERY_LINKS = [("Depot", "Hafen", 2), ("Depot", "Brücke", 9), ("Depot", "Tunnel", 4), ("Hafen", "Werk", 3), ("Hafen", "Brücke", 4), ("Brücke", "Kunde", 3), ("Tunnel", "Kunde", 10),
                  ("Werk", "Fähre", 5), ("Fähre", "Kunde", 20)]


def delivery_network():
    names = [s[0] for s in DELIVERY_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in DELIVERY_LINKS], [(s[1], s[2]) for s in DELIVERY_STOPS], names)
    return Network("delivery", g, (idx["Depot"],), (idx["Kunde"],), True, "Minuten", "Lieferwege", "Depot", "Kunde",
                   "Ein eigenes kleines Liefernetz in der Art des Beispiels aus Grokking Algorithms (Kap. 9): die Zahl an einer Strecke sind Fahrminuten. Die Route mit den wenigsten Strecken ist nicht die schnellste.")


EXCHANGE_ITEMS = [("Fahrradlampe", 0.0, 1.6), ("Klingel", 2.2, 2.8), ("Werkzeugset", 2.2, 0.4), ("Sattel", 4.4, 3.2), ("Laufrad", 4.4, 1.2), ("Rennrad", 6.6, 2.2)]
EXCHANGE_LINKS = [("Fahrradlampe", "Klingel", 5), ("Fahrradlampe", "Werkzeugset", 6), ("Werkzeugset", "Klingel", -4), ("Klingel", "Laufrad", 7), ("Klingel", "Sattel", 3),
                  ("Laufrad", "Rennrad", 4), ("Sattel", "Rennrad", 9)]


def exchange_network():
    names = [s[0] for s in EXCHANGE_ITEMS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in EXCHANGE_LINKS], [(s[1], s[2]) for s in EXCHANGE_ITEMS], names, directed=True)
    return Network("exchange", g, (idx["Fahrradlampe"],), (idx["Rennrad"],), True, "Euro", "Tausch mit Rückzahlung", "Fahrradlampe", "Rennrad",
                   "Ein eigenes Tauschnetz in der Art des Klavier-Tauschs aus Grokking Algorithms (Kap. 9): jede Zahl ist die Zuzahlung in Euro für einen Tausch. Ein Tausch mit Rückzahlung hat negative Kosten - das bricht Dijkstra.",
                   negative=True)


# Rohdaten mit Parallelkanten und einer Selbstschleife (wie sie OSM-Netze enthalten): (von, nach, Kosten) in der Reihenfolge der Rohdaten
RAW_TRAP_NODES = ("A", "B", "C")
RAW_TRAP_ARCS = [(0, 1, 5.0), (0, 1, 1.0), (1, 1, 1.0), (1, 2, 2.0)]


# --- Zusammenbau -------------------------------------------------------------------------------------------------------------------------

def make_network(net, side=C.DEFAULT_SIDE, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, blocked=C.DEFAULT_BLOCKED, walls=C.DEFAULT_WALLS, seed=C.DEFAULT_SEED):
    if net == "city":
        return city_network(int(side), float(reach), float(spread), int(blocked), int(seed))
    if net == "maze":
        return maze_network(int(side), int(walls), int(seed))
    if net == "toronto":
        return toronto_network()
    if net == "delivery":
        return delivery_network()
    if net == "exchange":
        return exchange_network()
    raise ValueError(net)
