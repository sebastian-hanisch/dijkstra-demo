"""Dijkstra - Kosten korrekt, aber blind in alle Richtungen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Dijkstra-Algorithmus - und lässt stattdessen das Beispiel wachsen.
Zweites Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Fortsetzung der Breitensuche-Demo: sie zählte Kanten, Dijkstra zählt Kosten - und hat seine eigenen Schwächen (alle Richtungen gleich, keine negativen Kosten).
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import dj_constants as C
import dj_evaluation as ev
from dj_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from dj_queues import QUEUE_LABELS
from dj_scenario import RAW_TRAP_ARCS, make_network
from dj_visualization import ROUTE_NAMES, build_front, build_negative, build_network, build_queues, build_radius, build_share_hist

st.set_page_config(page_title="Dijkstra – Sebastian Hanisch", layout="wide")

FIXED_NETS = ("delivery", "exchange", "toronto")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


def _cost(net, x):
    return f"{x:,.0f} {net.unit}".replace(",", ".")


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    return ev.analyse(make_network(*params))


@st.cache_data(show_spinner=False)
def _pair_stats(params, pairs):
    return ev.pair_stats(make_network(*params), pairs, params[-1])


@st.cache_data(show_spinner=False)
def _radius(reach):
    return ev.radius_growth(reach=reach)


@st.cache_data(show_spinner=False)
def _queues():
    return ev.queue_table()


@st.cache_data(show_spinner=False)
def _negative():
    return ev.negative_share()


@st.cache_data(show_spinner=False)
def _raw_trap():
    return ev.raw_trap()


st.title("🧭 Dijkstra – Kosten korrekt, aber blind in alle Richtungen")
st.markdown(
    """
Die Breitensuche zählt Kanten - **Dijkstra** zählt Kosten. Er legt Knoten nacheinander **endgültig fest**, immer den mit den kleinsten bisher bekannten Kosten vom Start, und verbessert dabei die Kosten seiner Nachbarn.
Weil er immer den billigsten Knoten nimmt, steht jeder festgelegte Knoten schon am Ende seiner kürzesten Route - **solange keine Kante negative Kosten hat**. Der Preis: Dijkstra kennt die Richtung des Ziels nicht und legt alles fest,
was näher am Start liegt als das Ziel. Diese Demo zeigt die Korrektheit und ihren Preis - und zwei Fallen: **negative Kanten** und **Rohdaten** mit Parallelkanten.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Fortsetzung der Breitensuche-Demo - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwächen von Dijkstra sind die Ansatzpunkte der nächsten Stücke: die Suche in alle Richtungen (**Bidirektionale Suche**, **A\\***), die negativen Kanten (**Bellman-Ford**), jede Anfrage von vorn (**Contraction Hierarchies**). "
    "Die kleinen Netze sind eigene Graphen in der Art der Beispiele aus *Grokking Algorithms* (A. Bhargava, Kap. 9); die Rohdaten-Falle und das Toronto-Szenario folgen *Optimization Algorithms* (A. Khamis, Kap. 3)."
)

with st.expander("So funktioniert Dijkstra", expanded=True):
    st.markdown(
        """
1. **Start:** der Start hat die vorläufige Entfernung 0, alle anderen ∞. Ein Knoten ist zunächst *unbekannt*.
2. **Festlegen:** unter den Knoten in der **Warteschlange** (vorläufige Entfernung bekannt, noch nicht festgelegt) kommt der mit der kleinsten Entfernung dran. Seine Entfernung ist jetzt **endgültig**: jede andere Route zu ihm müsste über einen Knoten
   führen, der noch weiter weg ist - und mit nichtnegativen Kosten kann eine Route nicht billiger werden, wenn man Kanten anhängt.
3. **Verbessern:** für jeden Nachbarn wird Entfernung + Kantenkosten berechnet; ist das kleiner als seine bisherige Entfernung, merkt er sich die neue Entfernung und den Vorgänger.
4. **Ziel:** sobald das Ziel festgelegt ist, endet die Suche; die Route ergibt sich rückwärts über die Vorgänger.
5. **Was nicht geht:** eine **negative Kante** macht Schritt 2 falsch - ein festgelegter Knoten könnte über sie noch billiger werden. Dijkstra merkt das nicht von selbst (hier zählt ein **Alarm** mit), sein Ergebnis stimmt dann nicht mehr.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (Lieferwege, Tausch mit Rückzahlung: eigene Graphen mit Tabelle), erzeugt (Stadtnetz, Labyrinth) oder echte OpenStreetMap-Daten (Toronto Campus: 5 072 Kreuzungen und Wegpunkte, 14 503 gerichtete Kanten). "
             "Alle Kosten sind ganze Zahlen (Meter, Minuten, Euro).",
    )
    if net_key in C.GRID_NETS:
        side = st.slider(
            "Kreuzungen je Seite" if net_key == "city" else "Zellen je Seite", *bounds("side_slider"), key="side_slider",
            help="Größe des Netzes (Seite × Seite). Beim Stadtnetz (Reichweite 2.3, Streuung 1.0) ist der Median-Umweg der Breitensuche-Route bei 6 / 10 / 20 / 30 / 40 Kreuzungen je Seite 10 % / 22 % / 28 % / 28 % / 30 %; "
                 "der Anteil des Netzes, den Dijkstra bis zum Ziel festlegt, bleibt im Median bei rund der Hälfte (52 % / 48 % / 50 % / 46 % / 46 %).",
        )
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "city":
        reach = st.slider(
            "Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1,
            help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster, größer = auch längere Verbindungen). Median-Umweg der Breitensuche-Route bei 1.0 / 1.5 / 2.3 / 3.2: 7 % / 18 % / 28 % / 34 %; "
                 "Dijkstra ist davon unberührt (Umweg 0), sein festgelegter Anteil bleibt bei rund der Hälfte.",
        )
        st.session_state[KEPT["reach_slider"]] = reach
        spread = st.slider(
            "Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25,
            help="Kosten einer Straße = Länge × (1 + Streuung × Zufall), gerundet auf ganze Meter: Ampeln, Steigung, Belag. Median-Umweg der Breitensuche-Route bei 0 / 0.5 / 1 / 2 / 3: 11 % / 18 % / 28 % / 47 % / 62 %.",
        )
        st.session_state[KEPT["spread_slider"]] = spread
        blocked = st.slider(
            "Gesperrte Straßen [%]", *bounds("blocked_slider"), key="blocked_slider",
            help="Anteil der Straßen, die gesperrt sind (das Netz bleibt zusammenhängend). Median-Umweg der Breitensuche-Route bei 0 / 20 / 40 / 60 %: 30 % / 28 % / 22 % / 15 %.",
        )
        st.session_state[KEPT["blocked_slider"]] = blocked
    else:
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
        blocked = int(st.session_state.get(KEPT["blocked_slider"], C.DEFAULT_BLOCKED))
    if net_key == "maze":
        walls = st.slider(
            "Wände [%]", *bounds("walls_slider"), key="walls_slider",
            help="Anteil der Zellen, die Wand sind (bei zu vielen Wänden werden einzelne wieder geöffnet, damit Start und Ziel verbunden bleiben). Jeder Schritt kostet dasselbe: Dijkstra und Breitensuche finden dieselbe Route.",
        )
        st.session_state[KEPT["walls_slider"]] = walls
    else:
        walls = int(st.session_state.get(KEPT["walls_slider"], C.DEFAULT_WALLS))
    if net_key in C.GRID_NETS:
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Größe, Reichweite und Streuung gehören zum Stadtnetz.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2), "blocked_slider": int(blocked),
                   "walls_slider": int(walls), "seed_input": int(seed)})

# feste Netze ignorieren die Regler des Rasters, das Stadtnetz das Labyrinth und umgekehrt: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
if net_key in FIXED_NETS:
    params = (net_key, C.DEFAULT_SIDE, C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, C.DEFAULT_WALLS, C.DEFAULT_SEED)
elif net_key == "city":
    params = (net_key, int(side), round(float(reach), 1), round(float(spread), 2), int(blocked), C.DEFAULT_WALLS, int(seed))
else:
    params = (net_key, int(side), C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, int(walls), int(seed))
with st.spinner("Rechne..."):
    a = _analysis(params)
net, res, m = a.net, a.result, a.metrics
g = net.graph
small = bool(g.names)
route_options = [k for k in ("bfs", "greedy", "bellman_ford") if a.routes.get(k)]

# --- Dijkstra in Aktion ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Dijkstra in Aktion")
last_step = len(res.order)
if st.session_state.get("dj_step_owner") != params:
    st.session_state["dj_step"] = last_step
    st.session_state["routes_select"] = [k for k in ("bfs", "bellman_ford") if k in route_options]
    st.session_state["dj_step_owner"] = params
st.session_state["routes_select"] = [k for k in st.session_state.get("routes_select", []) if k in route_options]
step_col, play_col = st.columns([5, 2])
with step_col:
    if last_step > 1:
        step = st.slider("Festgelegte Knoten", 0, last_step, key="dj_step",
                         help="Wie viele Knoten Dijkstra schon endgültig festgelegt hat: 0 = nur der Start ist bekannt, ganz rechts = das Ziel ist festgelegt und die Routen erscheinen.")
    else:
        step = last_step
        st.caption("Start und Ziel sind derselbe Knoten - es gibt nur einen Schritt.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
routes_shown = st.multiselect("Vergleichsrouten einblenden", route_options, key="routes_select", format_func=lambda k: ROUTE_NAMES[k],
                              help="Rot durchgezogen ist immer die Route von Dijkstra. Blau gestrichelt: die Route der Breitensuche (wenigste Kanten). Grau gepunktet (nur kleine Netze): immer die billigste noch unbesuchte Kante nehmen. "
                                   "Grün (nur beim Tauschnetz): das wahre Minimum nach Bellman-Ford.")
view_slot = st.empty()


def _render(current):
    with view_slot.container():
        if small:
            st.plotly_chart(build_network(net, a, current, ("dijkstra",) + tuple(routes_shown), height=400), width="stretch", key="net_chart")
            state = ev.table_state(a, current)
            df = pd.DataFrame(state["rows"])
            changed = df.pop("geändert")

            def highlight(row):
                return ["background-color: #ffe08a" if (changed.loc[row.name] and col in ("Entfernung", "Vorgänger", "Status")) else "" for col in row.index]

            t1, t2 = st.columns([3, 2])
            t1.markdown("**Entfernungen und Vorgänger** (gelb: in diesem Schritt geändert)")
            t1.dataframe(df.style.apply(highlight, axis=1).format({"Entfernung": lambda x: "∞" if x is None or pd.isna(x) else f"{x:g}"}), hide_index=True, width="stretch")
            t2.markdown("**Warteschlange:** " + (", ".join(f"{name} ({d:g})" for d, name in state["queue"]) if state["queue"] else "leer"))
            for name, old, new in state["alarms"]:
                t2.markdown(f"🚨 **Alarm:** {name} war schon mit {old:g} festgelegt, über eine negative Kante wäre {new:g} möglich gewesen.")
        else:
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_network(net, a, current, ("dijkstra",) + tuple(routes_shown)), width="stretch", key="net_chart")
            c2.markdown("**Knoten in der Warteschlange nach jeder Festlegung** (die Front)")
            c2.plotly_chart(build_front(res.front, current), width="stretch", key="front_chart")
            c2.caption(f"Nach {current} von {last_step} Festlegungen; größte Front: {m['front_max']} Knoten.")


if auto_play:
    n_frames = min(last_step, 60)
    for k in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames + 1)}):
        _render(k)
        time.sleep(min(0.6, 6.0 / n_frames))
    step = last_step
else:
    _render(step)

st.caption(net.note + (" Karte: © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Daten unter der [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/)." if net.key == "toronto" else ""))

st.markdown("---")

# --- Kosten korrekt - und was kostet das? -----------------------------------------------------------------------------------------------

st.markdown("## 🎯 Kosten korrekt – und was kostet das?")
st.caption(
    "**Umweg** = Kosten der Breitensuche-Route geteilt durch die von Dijkstra, minus 1 - die Schwäche der Wurzel dieser Linie, hier aufgelöst. "
    "**Festgelegt** = Knoten, deren Kosten schon endgültig feststehen, wenn das Ziel gefunden ist: alles, was näher am Start liegt als das Ziel."
)
if not m["reachable"]:
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar - Dijkstra meldet das ausdrücklich, statt eine Route zu erfinden.")
else:
    m1, m2, m3, m4 = st.columns(4)
    if net.negative:
        m1_delta = f"{_cost(net, m['cost_dijkstra'] - m['cost_bellman_ford'])} mehr als das wahre Minimum" if m["cost_dijkstra"] > m["cost_bellman_ford"] + 1e-9 else "gleich dem wahren Minimum"
    else:
        m1_delta = f"{_cost(net, m['cost_bfs'] - m['cost_dijkstra'])} weniger als die Breitensuche" if m["cost_bfs"] > m["cost_dijkstra"] + 1e-9 else "gleich der Breitensuche"
    m1.metric("Kosten der Dijkstra-Route", _cost(net, m["cost_dijkstra"]), delta=m1_delta, delta_color="off",
              help=f"Kosten der Route der Breitensuche (wenigste Kanten): {_cost(net, m['cost_bfs'])}." + (f" Wahres Minimum nach Bellman-Ford: {_cost(net, m['cost_bellman_ford'])}." if net.negative else ""))
    m2.metric("Kanten der Dijkstra-Route", f"{m['hops_dijkstra']}", delta=f"{m['hops_dijkstra'] - m['hops_bfs']:+d} gegenüber der Breitensuche", delta_color="off",
              help="Die billigste Route hat oft mehr Kanten als die kantenkürzeste.")
    m3.metric("Festgelegte Knoten", f"{m['settled']}", delta=f"{_pct(m['settled_share'])} von {m['n']}", delta_color="off", help="Alle Knoten, die näher am Start liegen als das Ziel - Dijkstra kennt die Richtung des Ziels nicht.")
    m4.metric("Größte Front", f"{m['front_max']}", delta=f"Breitensuche entdeckte {m['discovered_bfs']}", delta_color="off", help="Größte Warteschlange = Speicherbedarf; im Delta die Zahl der von der Breitensuche entdeckten Knoten.")
    code = ev.verdict(a)
    if code == "negative_wrong":
        st.warning(f"⚠️ Negative Kante: Dijkstra findet {_cost(net, m['cost_dijkstra'])}, die beste Route kostet {_cost(net, m['cost_bellman_ford'])} - **{_pct(m['overpay'])} zu viel**. "
                   f"Der Alarm zeigt, wo es schiefging: ein Knoten war schon festgelegt und hätte über die negative Kante noch billiger erreicht werden können; die Festlegung war nicht endgültig. Die Referenz ist Bellman-Ford (nächstes Stück der Linie).")
    elif code == "negative_ok":
        st.info("ℹ️ Das Netz hat negative Kanten, Dijkstra liegt hier aber zufällig richtig - keine Garantie.")
    elif code == "unweighted":
        st.success(f"✅ Alle Kanten kosten dasselbe: Dijkstra legt genau die Schichten der Breitensuche fest (gleiche Route mit {m['hops_dijkstra']} Kanten, {m['settled']} von {m['n']} Knoten festgelegt). "
                   "Die Warteschlange ist dann eine gewöhnliche Schlange - bei gleichen Kosten ist Dijkstra die Breitensuche.")
    elif code == "same_route":
        st.info(f"Zufällig gleich: die Route der Breitensuche ist auch die kostenoptimale ({_cost(net, m['cost_dijkstra'])}).")
    else:
        extra = ""
        if m["greedy_detour"] == m["greedy_detour"]:
            extra = f" Immer die billigste Kante zu nehmen, endet bei {_cost(net, m['cost_greedy'])} ({_pct(m['greedy_detour'])} mehr)."
        st.success(f"✅ Dijkstra findet die kostenoptimale Route: {_cost(net, m['cost_dijkstra'])} mit {m['hops_dijkstra']} Kanten. Die Breitensuche (wenigste Kanten: {m['hops_bfs']}) käme auf {_cost(net, m['cost_bfs'])} - **{_pct(m['detour_bfs'])} mehr**.{extra} "
                   f"Der Preis: bevor das Ziel feststeht, sind {m['settled']} von {m['n']} Knoten ({_pct(m['settled_share'])}) endgültig festgelegt.")

    if not net.negative:
        st.markdown("**Nicht nur dieses eine Paar**")
        ps = _pair_stats(params, C.PAIRS)
        if ps["n_pairs"]:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Festgelegt (Median)", _pct(ps["share_median"]), help=f"Über {ps['n_pairs']} zufällige erreichbare Start-Ziel-Paare: die Hälfte der Paare legt höchstens diesen Anteil des Netzes fest.")
            p2.metric("Festgelegt (90 %-Quantil)", _pct(ps["share_p90"]), help="Bei einem Zehntel der Paare wird mehr festgelegt.")
            p3.metric("Umweg der Breitensuche (Median)", _pct(ps["detour_median"]), delta=f"90 %-Quantil {_pct(ps['detour_p90'])}", delta_color="off", help="Wie viel teurer die Route der Breitensuche im Median ist; Dijkstra hat immer 0.")
            p4.metric("Paare ohne Umweg", _pct(ps["detour_zero"]), help="Anteil der Paare, bei denen die Breitensuche zufällig die kostenoptimale Route findet.")
            st.plotly_chart(build_share_hist(ps["share"]), width="stretch", key="share_hist")
            st.caption(f"{ps['n_pairs']} zufällige Start-Ziel-Paare im gewählten Netz. Der festgelegte Anteil ist bei zufälligen Paaren ungefähr gleichverteilt (Dijkstra legt alle Knoten fest, die näher liegen als das Ziel - die Rangzahl eines zufälligen Ziels ist "
                       "gleichverteilt), im Mittel also rund die Hälfte des Netzes, egal wie das Netz aussieht. Dijkstra liefert dafür immer die kostenoptimale Route.")

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Dijkstra im Vergleich"):
    st.markdown("**Was jedes Verfahren für Start und Ziel oben findet**")
    if m["reachable"]:
        rows = [("Dijkstra (kostenoptimal)", "dijkstra"), ("Breitensuche (wenigste Kanten)", "bfs"), ("immer die billigste Kante", "greedy"), ("Bellman-Ford (wahres Minimum)", "bellman_ford")]
        rows = [(n, k) for n, k in rows if a.routes[k]]
        cost_of = {"dijkstra": m["cost_dijkstra"], "bfs": m["cost_bfs"], "greedy": m["cost_greedy"], "bellman_ford": m["cost_bellman_ford"]}
        hops_of = {"dijkstra": m["hops_dijkstra"], "bfs": m["hops_bfs"], "greedy": m["hops_greedy"], "bellman_ford": len(a.routes["bellman_ford"]) - 1}
        st.table({"Verfahren": [n for n, _ in rows], "Kanten": [hops_of[k] for _, k in rows], f"Kosten [{net.unit}]": [f"{cost_of[k]:,.0f}".replace(",", ".") for _, k in rows]})
    c = m["counters"]
    st.markdown("**Was die Warteschlange (Binärheap mit Decrease-Key) dabei geleistet hat**")
    st.table({"Zähler": ["Knoten eingefügt", "Schlüssel gesenkt (Decrease-Key)", "Knoten entnommen", "Schlüsselvergleiche", "geprüfte Kanten"],
              "Anzahl": [c["pushes"], c["decrease_keys"], c["pops"], c["work"], c["relaxations"]]})
    st.caption("Jeder eingefügte Knoten wird genau einmal entnommen; ein Decrease-Key ist eine Verbesserung der vorläufigen Entfernung eines Knotens, der schon in der Warteschlange wartet. "
               "Die fünf Warteschlangen im direkten Vergleich stehen unten bei den Experimenten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Blind in alle Richtungen: wie viel legt Dijkstra fest?")
radius_reach = float(reach) if net_key == "city" else C.DEFAULT_REACH
if st.button("Von der Mitte eines Stadtnetzes aus alle Radien durchgehen (dauert wenige Sekunden)", key="radius_start"):
    st.session_state["radius_on"] = True
if st.session_state.get("radius_on"):
    with st.spinner("Lege ein Netz mit 3 600 Kreuzungen fest..."):
        rg = _radius(round(radius_reach, 1))
    st.plotly_chart(build_radius(rg), width="stretch", key="radius_chart")
    st.caption(f"60 × 60 Kreuzungen (Reichweite {radius_reach:.1f}), Start in der Mitte, alle Knoten bis zum Kostenradius L festgelegt. Die Zahl wächst mit **L hoch {rg['slope']:.2f}** (Steigung im Log-Log, gemessen im Inneren des Netzes) - "
               "wie die Fläche eines Kreises: doppelte Entfernung, vierfach so viele festgelegte Knoten, und das in alle Richtungen, auch vom Ziel weg.")

st.markdown("---")

st.subheader("🔬 Die Warteschlange: fünf Umsetzungen, dasselbe Ergebnis")
if st.button("Feld, Binärheap, faulen Heap, Dial-Eimer und Fibonacci-Heap vergleichen (dauert wenige Sekunden)", key="queue_start"):
    st.session_state["queue_on"] = True
if st.session_state.get("queue_on"):
    with st.spinner("Lasse fünf Warteschlangen auf vier Netzen laufen..."):
        qrows = _queues()
    biggest = [r for r in qrows if r["n"] == max(x["n"] for x in qrows)]
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_queues(qrows), width="stretch", key="queue_chart")
    c2.markdown(f"**Größtes Netz: {biggest[0]['n']:,} Kreuzungen, {biggest[0]['m']:,} Kanten**".replace(",", "."))
    c2.table({"Warteschlange": [QUEUE_LABELS[r["queue"]] for r in biggest], "Vergleiche bzw. Eimerschritte": [f"{r['work']:,}".replace(",", ".") if r["queue"] != "lazy" else "n/a (in C)" for r in biggest],
              "Decrease-Keys": [r["decrease_keys"] for r in biggest], "Laufzeit [ms]": [f"{r['ms']:.0f}" for r in biggest]})
    binary = next(r for r in biggest if r["queue"] == "binary")["work"]
    fib = next(r for r in biggest if r["queue"] == "fibonacci")["work"]

    def _slope(queue):
        sel = [r for r in qrows if r["queue"] == queue]
        return float(np.polyfit(np.log([r["n"] for r in sel]), np.log([r["work"] for r in sel]), 1)[0])
    st.caption(f"Alle fünf liefern dieselben Entfernungen (im Lauf geprüft); es unterscheidet sich nur der Aufwand. Das **Feld** vergleicht bei jeder Entnahme alle wartenden Knoten: sein Aufwand wächst hier wie n hoch {_slope('array'):.1f} (Knoten mal Front; im schlimmsten Fall n²), der des Binärheaps nur wie n hoch {_slope('binary'):.2f}. "
               f"Der **Fibonacci-Heap** braucht {1 - fib / binary:.0%} weniger Schlüsselvergleiche als der Binärheap, ist in dieser Python-Umsetzung aber nicht spürbar schneller (Laufzeiten sind Messwerte und schwanken) - die theoretische Schranke O(E + V log V) zahlt sich erst bei sehr dichten Netzen aus. "
               "Der **faule Heap** (`heapq`, in C) hat keinen Decrease-Key: er legt statt dessen einen zweiten Eintrag an, der alte wird beim Entnehmen übersprungen. Die Laufzeiten sind Messwerte dieses Laufs und schwanken; die Zähler sind fest.")

st.markdown("---")

st.subheader("🔬 Negative Kanten: wie oft geht Dijkstra schief?")
if st.button("Zufallsnetze mit negativen Kanten prüfen (dauert wenige Sekunden)", key="negative_start"):
    st.session_state["negative_on"] = True
if st.session_state.get("negative_on"):
    with st.spinner("Prüfe 1 000 Zufallsnetze gegen Bellman-Ford..."):
        nrows = _negative()
    st.plotly_chart(build_negative(nrows), width="stretch", key="negative_chart")
    r2 = next(r for r in nrows if abs(r["fraction"] - 0.02) < 1e-9)
    st.caption(f"Je Anteil 200 zufällige gerichtete Netze (40 Knoten, 120 Kanten; 5 feste Sweep-Datensätze). Ohne negative Kanten ist Dijkstra immer richtig. Schon bei **2 % negativer Kanten** liefert er in {r2['wrong'] / r2['total']:.0%} der Netze ({r2['wrong']} von {r2['total']}) falsche Kosten "
               f"(in {r2['cycle'] / r2['total']:.0%} gibt es sogar einen negativen Zyklus - dann existiert keine kürzeste Route). Mit vielen negativen Kanten überwiegen die Zyklen. Die Referenz ist Bellman-Ford, das nächste Stück der Linie.")

st.markdown("---")

st.subheader("🔬 Rohdaten-Falle: Parallelkanten und Schleifen")
if st.button("Ein Netz mit Parallelkanten auf vier Arten lösen", key="raw_start"):
    st.session_state["raw_on"] = True
if st.session_state.get("raw_on"):
    rt = _raw_trap()
    st.markdown("**Rohdaten** (von → nach: Kosten): " + ", ".join(f"{'ABC'[u]} → {'ABC'[v]}: {c:g}" for u, v, c in RAW_TRAP_ARCS) + ". A → B gibt es doppelt (5 und 1), B → B ist eine Schleife.")
    st.table({"Umsetzung": ["Mehrfachkanten-Graph (alle Kanten bleiben)", "bereinigt (billigste Parallelkante, Schleife weg)", "Wörterbuch, letzte Kante gewinnt", "Wörterbuch, erste Kante gewinnt"],
              "Kosten A nach C": [f"{rt['multigraph']:g}", f"{rt['cleaned']:g}", f"{rt['dict_last_wins']:g}", f"{rt['dict_first_wins']:g}"]})
    st.caption("Karten-Rohdaten (OpenStreetMap) enthalten Parallelkanten und Schleifen. Ein Graph, der alle Kanten kennt, oder ein bereinigter liefert 3. Eine Umsetzung mit einem Wörterbuch je Knotenpaar kennt nur **eine** der Kanten - "
               "welche, hängt von der Reihenfolge der Rohdaten ab, und das Ergebnis schwankt zwischen 3 und 7. Deshalb bereinigt diese Demo die Toronto-Daten vorab (billigste Parallelkante, keine Schleifen).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Keine negativen Kosten** | Tauschnetz: Dijkstra findet 16 Euro, die beste Route kostet 13 (**23 % zu viel**), der Alarm zeigt den Bruch der Festlegung. In Zufallsnetzen mit 2 % negativer Kanten liefert er in etwa jedem fünften Netz falsche Kosten. | **Bellman-Ford** (nächstes Stück der Linie) |
| **Die Suche darf in alle Richtungen gleich weit laufen** | Bei zufälligen Paaren legt Dijkstra im Median rund **die Hälfte** des Netzes fest (Stadtnetz 51 %, Toronto 51 %, bei einem Zehntel der Paare über 85 %); die Zahl der festgelegten Knoten wächst mit der Entfernung im Quadrat (Steigung 2.08). | **Bidirektionale Suche** (von beiden Enden), **A\\*** (Baumsuche-Linie) |
| **Jede Anfrage beginnt von vorn** | Toronto: für **ein** Paar (Route 891 m) werden 2 513 von 5 072 Knoten festgelegt; die nächste Anfrage fängt wieder bei null an. | **Contraction Hierarchies**: erst vorrechnen, dann blitzschnell fragen |
| **Ein Start genügt** | Dijkstra liefert die kürzesten Wege von **einem** Start; für alle Paare läuft er n-mal. | **Floyd-Warshall**, **Johnson** (Konvergenz mit Bellman-Ford) |
"""
)
st.caption("Die Nachbarn der Kürzeste-Wege-Linie (noch nicht gebaut): Bidirektionale Suche, Contraction Hierarchies, Bellman-Ford, Floyd-Warshall, Johnson und Mehrkriterien-Routing. Bereits gebaut: die Wurzel, die Breitensuche-Demo. A\\* steht in der Baumsuche-Linie.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \ge 0$ (ganzzahlig in dieser Demo); Start $s$; $\delta(v)$ = Kosten einer billigsten Route von $s$ nach $v$.

**Dijkstra.** Vorläufige Entfernungen $d(v)$, zu Beginn $d(s)=0$ und sonst $\infty$; Menge $S$ der festgelegten Knoten, zu Beginn leer. Wiederhole: nimm $u \notin S$ mit kleinstem $d(u)$, setze $S \leftarrow S \cup \{u\}$ und verbessere
alle Nachbarn $v$: $d(v) \leftarrow \min\big(d(v),\ d(u)+c_{uv}\big)$ (**Relaxation**).
**Invariante:** für jeden Knoten $u \in S$ gilt $d(u)=\delta(u)$, und für jeden anderen Knoten ist $d(v)$ die Länge einer billigsten Route, die außer dem Endknoten nur Knoten in $S$ benutzt.
**Beweisidee:** angenommen, $d(u) > \delta(u)$ beim Festlegen von $u$. Auf einer billigsten Route zu $u$ gibt es den ersten Knoten $y \notin S$; sein Vorgänger liegt in $S$, also $d(y)=\delta(y)$. Wegen $c_e \ge 0$ ist $\delta(y) \le \delta(u) < d(u)$ -
dann hätte $y$ vor $u$ gewählt werden müssen. **Bei einer negativen Kante bricht genau der Schritt $\delta(y) \le \delta(u)$**: eine Route kann billiger werden, wenn man Kanten anhängt.

**Aufwand.** Mit einer Warteschlange, die Einfügen, Verkleinern und Entnehmen kann: $|V|$ Einfügungen und Entnahmen, höchstens $|E|$ Verkleinerungen. Feld: $O(|V|^2)$ im schlimmsten Fall (in ebenen Netzen mit einer Front von etwa $\sqrt{|V|}$ eher $|V|^{1,5}$); Binärheap: $O((|V|+|E|)\log|V|)$; Fibonacci-Heap: $O(|E|+|V|\log|V|)$; Dial-Eimer bei ganzzahligen Kosten
bis $C$: $O(|E|+C|V|)$. Der faule Heap (`heapq`) hat kein Verkleinern, legt dafür bis zu $|E|$ Einträge an: $O(|E|\log|E|)$.

**Festgelegter Anteil.** Beim Abbruch am Ziel $t$ sind genau die Knoten mit $\delta(v) < \delta(t)$ festgelegt (plus einige mit Gleichstand). Für ein zufälliges Ziel ist die Rangzahl von $\delta(t)$ gleichverteilt, der erwartete Anteil also etwa $\tfrac12$.
In der Ebene enthält der Radius $L$ etwa $\pi L^2$ Fläche: die Zahl festgelegter Knoten wächst wie $L^2$.

**Gleiche Kosten.** Ist $c_e \equiv 1$, sind die Warteschlange und die Schichten der Breitensuche dasselbe: Dijkstra legt die Knoten in der Reihenfolge der Breitensuche fest.

**Negative Kanten.** Ein festgelegter Knoten $v$ mit einer Verbesserung $d(u)+c_{uv}<d(v)$ ist der Alarm dieser Demo; Dijkstra bessert nicht nach (so wird es gelehrt), das Ergebnis ist dann falsch. Negative Kanten ohne negativen Zyklus löst Bellman-Ford
mit $O(|V||E|)$; mit negativem Zyklus gibt es keine kürzeste Route.

Implementiert in `dj_graph.py` (CSR-Graph, bereinigte und rohe Kanten), `dj_queues.py` (fünf Warteschlangen mit Zählern), `dj_algorithm.py` (Dijkstra, Bellman-Ford, Breitensuche, Greedy), `dj_scenario.py` (Netze), `dj_evaluation.py` (Kennzahlen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
