"""Konstanten, Grenzen der Regler und Presets. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("delivery", "exchange", "city", "maze", "toronto")
NET_LABELS = {
    "delivery": "🚚 Lieferwege (klein)",
    "exchange": "🔁 Tausch mit Rückzahlung (negative Kante)",
    "city": "🏙️ Stadtnetz (erzeugt)",
    "maze": "🧱 Labyrinth (gleiche Kosten)",
    "toronto": "🍁 Toronto Campus (OpenStreetMap)",
}
GRID_NETS = ("city", "maze")       # Netze mit Größenregler
SMALL_NETS = ("delivery", "exchange")

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 6, 40, 20
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 2.3
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
BLOCKED_MIN, BLOCKED_MAX, DEFAULT_BLOCKED = 0, 60, 20          # Prozent der Straßen
WALLS_MIN, WALLS_MAX, DEFAULT_WALLS = 10, 45, 35               # Prozent der Zellen
DEFAULT_SEED = 7
DEFAULT_NET = "delivery"

SWEEP_SEEDS = tuple(range(100000, 100005))
PAIRS = 200                        # zufällige Start-Ziel-Paare je Netz für die Verteilungen

COLORS = {"dijkstra": "#d62728", "bfs": "#1f77b4", "start": "#111111", "goal": "#ff7f0e", "front": "#ff7f0e"}

# Jedes Preset setzt alle Regler; bei den festen Netzen (Lieferwege, Tausch, Toronto) haben Größe, Reichweite, Streuung, Sperrungen und Wände keine Wirkung (die Regler sind dort ausgeblendet) und werden auf die Standardwerte gesetzt.
_BASE = dict(side=DEFAULT_SIDE, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, blocked=DEFAULT_BLOCKED, walls=DEFAULT_WALLS, seed=DEFAULT_SEED)
PRESETS = {
    "🚚 Lieferwege": {**_BASE, "net": "delivery"},
    "🔁 Tausch mit Rückzahlung": {**_BASE, "net": "exchange"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
    "🧱 Labyrinth": {**_BASE, "net": "maze"},
    "🍁 Toronto Campus": {**_BASE, "net": "toronto"},
}
PRESET_HELP = {
    "🚚 Lieferwege": "Kleines eigenes Liefernetz in der Art des Beispiels aus Grokking Algorithms: die Route mit den wenigsten Strecken braucht 12 Minuten, die schnellste 9 (über drei Strecken). Die Tabelle zeigt, wie sich Entfernungen und Vorgänger Schritt für Schritt ändern.",
    "🔁 Tausch mit Rückzahlung": "Eigenes Tauschnetz in der Art des Klavier-Tauschs aus Grokking Algorithms: ein Tausch mit Rückzahlung hat negative Kosten. Dijkstra findet 16 Euro, die beste Route kostet 13 - und meldet einen Alarm.",
    "🏙️ Stadtnetz": "Erzeugtes Stadtnetz (20 × 20 Kreuzungen, Straßen bis 2.3 Blocklängen, Kosten mit Streuung 1.0): Dijkstra findet die kürzeste Route (3 186 m); die Route der Breitensuche ist 27 % länger. Vorher muss es aber das ganze Netz festlegen.",
    "🧱 Labyrinth": "Erzeugtes Labyrinth (35 % Wände, 20 × 20): jeder Schritt kostet dasselbe, Dijkstra legt genau die Schichten der Breitensuche fest.",
    "🍁 Toronto Campus": "Echte OpenStreetMap-Daten (dasselbe Paar wie in der Breitensuche-Demo, Kosten in ganzen Metern): Dijkstra findet 891 m statt 1 129 m der Breitensuche und legt dafür rund die Hälfte des Netzes fest.",
}
