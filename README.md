# Dijkstra – Kosten korrekt, aber blind in alle Richtungen – Streamlit-Demo

Zweites Stück der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung der [Breitensuche-Demo](../bfs-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Dijkstra-Algorithmus** – an einem wachsenden Beispiel.
Die Breitensuche zählte Kanten, Dijkstra zählt Kosten: er legt die Knoten nacheinander **endgültig fest**, immer den mit den kleinsten bisher bekannten Kosten, und verbessert dabei seine Nachbarn.
Jeder festgelegte Knoten steht damit am Ende seiner kürzesten Route – **solange keine Kante negative Kosten hat**. Der Preis: Dijkstra kennt die Richtung des Ziels nicht und legt alles fest, was näher am Start liegt als das Ziel.

**Einordnung in die Reihe (die Kanten des Graphen):** Dijkstra löst die Schwäche der Wurzel (Kosten werden ignoriert) und bringt eigene mit, die die nächsten Stücke aufgreifen:
die Suche in alle Richtungen (→ Bidirektionale Suche, A\*), negative Kanten (→ Bellman-Ford), jede Anfrage von vorn (→ Contraction Hierarchies). Bisher gebaut: die Wurzel und dieses Stück.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                              [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)               [dieses Stück]
       ├─ Bidirektionale Suche → Contraction Hierarchies                    [nicht gebaut]
       ├─ Bellman-Ford + Floyd-Warshall → Johnson (Konvergenz: Umgewichtung) [nicht gebaut]
       └─ Mehrkriterien-Routing (Zeit gegen CO₂, Pareto)                    [nicht gebaut]
A* steht einmal in der Baumsuche-Linie und wird von hier aus nur verlinkt.
```

## Beispiele in der Art der Lehrbücher

| Netz in der Demo | Quelle |
|---|---|
| **Lieferwege** (7 Orte, Fahrminuten; die Route mit den wenigsten Strecken ist nicht die schnellste, "immer die billigste Kante" läuft in die Irre; Tabelle Entfernung/Vorgänger je Schritt) | in der Art des Beispiels aus *Grokking Algorithms* (A. Bhargava), Kap. 9 – eigener Graph mit eigenen Zahlen |
| **Tausch mit Rückzahlung** (Fahrradteile tauschen, ein Tausch mit Rückzahlung = negative Kante) | in der Art des Klavier-Tauschs aus *Grokking Algorithms*, Kap. 9 – eigener Graph mit eigenen Gegenständen und Preisen |
| **Rohdaten-Falle** (Parallelkanten und Schleifen) | Idee aus *Optimization Algorithms* (A. Khamis), Kap. 3 – eigenes 3-Knoten-Beispiel |
| **Toronto Campus** (Reiterstandbild → Bahen Centre, dasselbe Paar wie in der Breitensuche-Demo) | Szenario aus *Optimization Algorithms*, Kap. 3 – **echte OpenStreetMap-Daten**, Kosten auf ganze Meter gerundet |
| **Stadtnetz**, **Labyrinth** (gleiche Kosten) | erzeugt, mit Größen- und Kostenreglern |

Aus den Büchern stammt nur die Idee der Beispiele; Text, Abbildungen, Code, Graphen und Zahlen der Bücher sind nicht übernommen.

**Daten und Lizenz:** Kartendaten © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Open Database License (ODbL) 1.0. `data/toronto_campus.json` ist ein Auszug daraus und steht deshalb ebenfalls unter der ODbL – siehe [data/LICENSE-ODbL.md](data/LICENSE-ODbL.md).

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Lieferwege | ✅ Dijkstra findet **9 Minuten** über drei Strecken; die Breitensuche (zwei Strecken) käme auf **12**, "immer die billigste Kante" auf **30**. Zwei vorläufige Entfernungen werden unterwegs verbessert (Brücke 9 → 6, Kunde 14 → 9) |
| Stadtnetz (20 × 20, Reichweite 2.3, Streuung 1.0) | ✅ Dijkstra findet die kürzeste Route (3 186 m), die Route der Breitensuche ist **27 % länger**; Median-Umweg der Breitensuche über 200 Paare 24 % (im Mittel über fünf feste Netze 28 %; Dijkstra: immer 0). ❌ Preis: bevor das Ziel feststeht, ist bei einem typischen Paar **rund die Hälfte** des Netzes festgelegt (Median 51 %, bei einem Zehntel der Paare über 85 %) |
| Toronto Campus | ✅ **891 m** statt 1 129 m der Breitensuche (27 % weniger) – dasselbe Paar wie in der Breitensuche-Demo. ❌ Dafür sind **2 513 von 5 072** Knoten festgelegt; über 200 Paare Median 51 % |
| Labyrinth (gleiche Kosten) | ✅ Dijkstra legt genau die **Schichten der Breitensuche** fest (in fünf festen Netzen geprüft), gleiche Route mit 42 Schritten |
| Blind in alle Richtungen | ❌ Von der Mitte eines Netzes aus wächst die Zahl der festgelegten Knoten mit **L hoch 2.08** (Steigung im Log-Log, im Inneren gemessen) – wie die Fläche eines Kreises. Der festgelegte Anteil ist bei zufälligen Paaren etwa gleichverteilt, im Mittel also die Hälfte, unabhängig von Größe und Form |
| Tausch mit Rückzahlung | ❌ Dijkstra findet **16 Euro**, die beste Route kostet **13** (23 % zu viel); der Alarm zeigt, wo die Festlegung nicht endgültig war |
| Negative Kanten in Zufallsnetzen | ❌ Ohne negative Kanten nie falsch; bei **2 %** negativer Kanten liefert Dijkstra in etwa jedem fünften Netz (20 %) falsche Kosten, mit vielen negativen Kanten überwiegen negative Zyklen (200 Netze je Anteil, 5 feste Datensätze) |
| Warteschlangen | ✅ fünf Umsetzungen, dieselben Entfernungen: **Feld**, **Binärheap** (Decrease-Key), **faul** (`heapq`), **Dial-Eimer**, **Fibonacci-Heap**. Jeder Knoten wird einmal eingefügt, etwa jeder zweite bekommt einen Decrease-Key. Fibonacci braucht rund 17 % weniger Schlüsselvergleiche als der Binärheap, ist aber in dieser Python-Umsetzung nicht spürbar schneller (Laufzeiten sind Messwerte, nicht getestet); das Feld wächst hier wie n hoch 1.5 (Knoten mal Front), die Heaps kaum stärker als linear |
| Rohdaten-Falle | ⚠️ ein Netz mit Parallelkanten: Mehrfachkanten-Graph und bereinigte Daten liefern 3; ein Wörterbuch je Knotenpaar 3 oder **7**, je nach Reihenfolge der Rohdaten |

Die Breitensuche als Vergleich ist die aus der Breitensuche-Demo; die Bellman-Ford-Referenz für negative Kanten ist nur ein Messwerkzeug (das Verfahren selbst ist ein späteres Stück der Linie).

## Was die Demo zeigt

1. **Dijkstra in Aktion** (Schritt-Regler + Abspielen): festgelegte Knoten nach Kosten gefärbt, die Front (Warteschlange) mit orangem Ring; bei kleinen Netzen die **Tabelle** mit Entfernung, Vorgänger und Status je Knoten (Änderungen gelb hervorgehoben) und der **Alarm** bei negativen Kanten; bei großen Netzen die Größe der Warteschlange je Schritt. Zuschaltbare Vergleichsrouten: Breitensuche, "immer die billigste Kante", wahres Minimum (Bellman-Ford).
2. **Kosten korrekt – und was kostet das?** – Kennzahlen des gezeigten Paars (Kosten, Kanten, festgelegte Knoten, größte Front) mit Urteil, dazu die **Verteilung über 200 zufällige Paare** (festgelegter Anteil, Umweg der Breitensuche).
3. **Vergleich** (Expander) der Verfahren und der Zähler der Warteschlange; **Experimente auf Knopfdruck**: Suchfläche gegen Radius, fünf Warteschlangen im Vergleich, negative Kanten in Zufallsnetzen, Rohdaten-Falle.
4. **Wo die Annahmen enden** (Tabelle mit den Ansatzpunkten der nächsten Stücke) und **Mathematische Formulierung** (Invariante, Beweisidee, Aufwand je Warteschlange, gleiche Kosten = Breitensuche, negative Kanten).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz und Regler wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz nicht gehören, sind ausgeblendet.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `dj_graph.py` | gerichteter Graph in CSR-Form; bereinigt (billigste Parallelkante, keine Schleifen) oder roh |
| `dj_queues.py` | fünf Prioritätswarteschlangen mit denselben Zählern (Einfügungen, Decrease-Keys, Entnahmen, Vergleiche) |
| `dj_algorithm.py` | Dijkstra (austauschbare Warteschlange, Protokoll je Schritt, Alarm), Bellman-Ford-Referenz, Breitensuche, Greedy-Gegenprobe, naive Wörterbuch-Variante |
| `dj_scenario.py` | Netze: Lieferwege, Tausch, Stadtnetz, Labyrinth, Toronto, Rohdaten-Beispiel |
| `dj_evaluation.py` | Kennzahlen, Verteilung über Paare, Experimente, Tabellenzustand |
| `dj_visualization.py`, `dj_presets.py`, `dj_constants.py` | Abbildungen, Presets und Permalink, Konstanten |
| `data/toronto_campus.json` | der OSM-Ausschnitt (5 072 Knoten, 14 503 gerichtete Kanten), Kosten in der App auf ganze Meter gerundet |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe von Dijkstra läuft gegen networkx (alle Warteschlangen, auch mit negativen Kanten gegen Bellman-Ford).
