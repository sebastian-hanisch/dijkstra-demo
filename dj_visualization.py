"""Plotly-Abbildungen: Netz mit festgelegten Knoten und Front, Routen, Warteschlangengröße, Verteilungen, Experimente. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import dj_constants as C

ROUTE_NAMES = {"dijkstra": "Dijkstra (kostenoptimal)", "bfs": "Breitensuche (wenigste Kanten)", "greedy": "immer die billigste Kante", "bellman_ford": "wahres Minimum (Bellman-Ford)"}
QUEUE_COLORS = {"array": "#7f7f7f", "binary": "#1f77b4", "lazy": "#17becf", "dial": "#2ca02c", "fibonacci": "#d62728"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _edge_segments(g):
    """Alle Kanten als eine Linienspur (None trennt die Segmente); Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = np.zeros(len(src), dtype=bool)
    _, first = np.unique(lo * g.n + hi, return_index=True)
    keep[first] = True
    u, v = lo[keep], hi[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _fmt(x):
    return f"{x:g}"


def build_network(net, analysis, k, show_routes=("dijkstra", "bfs"), height=520):
    """Das Netz nach `k` Festlegungen: festgelegte Knoten nach Kosten gefärbt, die Front (in der Warteschlange) mit orangem Ring; die Routen erscheinen, sobald das Ziel festgelegt ist."""
    g, res = net.graph, analysis.result
    fig = go.Figure()
    small = bool(g.names)                                                        # kleine, benannte Netze: Beschriftungen, Kosten an den Kanten, Tabelle
    ex, ey = _edge_segments(g)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.45)", width=1), hoverinfo="skip", showlegend=False))
    settled_nodes = np.array(res.order[:k], dtype=int)
    settled = np.zeros(g.n, dtype=bool)
    settled[settled_nodes] = True
    frontier = np.where((res.disc_step >= 0) & (res.disc_step <= k) & ~settled)[0]
    if small:
        rest = np.where(~settled & ~np.isin(np.arange(g.n), frontier))[0]
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers+text", showlegend=False, text=[g.names[i] for i in rest], textposition="top center",
                                     marker=dict(size=12, color="white", line=dict(color="gray", width=1.5)), hoverinfo="skip"))
        if g.directed:
            src = np.repeat(np.arange(g.n), g.degree())
            for u, v in zip(src, g.indices):
                fig.add_annotation(x=g.xy[v, 0], y=g.xy[v, 1], ax=g.xy[u, 0], ay=g.xy[u, 1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                                   arrowwidth=1.2, arrowcolor="rgba(120,120,120,0.8)", standoff=9, startstandoff=9)
        # Kosten an den Kanten
        src = np.repeat(np.arange(g.n), g.degree())
        done = set()
        for u, v, w in zip(src.tolist(), g.indices.tolist(), g.weight.tolist()):
            if not g.directed and (v, u) in done:
                continue
            done.add((u, v))
            mid = (g.xy[u] + g.xy[v]) / 2
            fig.add_annotation(x=mid[0], y=mid[1], text=_fmt(w), showarrow=False, font=dict(size=12, color="#b2182b" if w < 0 else "#555"), bgcolor="rgba(255,255,255,0.75)")
    if len(frontier):
        fig.add_trace(go.Scatter(x=g.xy[frontier, 0], y=g.xy[frontier, 1], mode="markers+text" if small else "markers", name="Front (in der Warteschlange)", showlegend=True,
                                 text=[g.names[i] for i in frontier] if small else None, textposition="top center",
                                 marker=dict(size=16 if small else (7 if g.n > 1500 else 9), color="rgba(255,255,255,0.9)", line=dict(color=C.COLORS["front"], width=2.5)), hoverinfo="skip"))
    if len(settled_nodes):
        cmax = max(float(np.nanmax(res.dist[np.isfinite(res.dist)])), 1.0) if np.isfinite(res.dist).any() else 1.0
        fig.add_trace(go.Scatter(
            x=g.xy[settled_nodes, 0], y=g.xy[settled_nodes, 1], mode="markers+text" if small else "markers", showlegend=False, text=[g.names[i] for i in settled_nodes] if small else None,
            textposition="top center", customdata=res.dist[settled_nodes], hovertemplate=f"Kosten vom Start: %{{customdata:g}} {net.unit}<extra></extra>",
            marker=dict(size=14 if small else (3 if g.n > 1500 else 6), opacity=1.0 if small else 0.9, color=res.dist[settled_nodes], colorscale="Viridis", cmin=min(0.0, float(res.dist[settled_nodes].min())),
                        cmax=cmax, colorbar=dict(title=f"Kosten<br>[{net.unit}]", thickness=12, len=0.6))))
    found = res.found >= 0 and k >= len(res.order)
    if found:
        styles = {"greedy": dict(color="#7f7f7f", width=3, dash="dot"), "bellman_ford": dict(color="#2ca02c", width=4, dash="dashdot"), "bfs": dict(color=C.COLORS["bfs"], width=3, dash="dash"),
                  "dijkstra": dict(color=C.COLORS["dijkstra"], width=5)}
        for key in ("greedy", "bellman_ford", "bfs", "dijkstra"):
            route = analysis.routes.get(key)
            if key in show_routes and route:
                pts = g.xy[route]
                fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=styles[key], name=ROUTE_NAMES[key], hoverinfo="skip"))
    for nodes, name, label, color in ((net.sources, "Start", net.start_label, C.COLORS["start"]), (net.targets, "Ziel", net.goal_label, C.COLORS["goal"])):
        pts = g.xy[list(nodes)]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{name}: {label}", hoverinfo="skip",
                                 marker=dict(size=16, color=color, symbol="star" if name == "Ziel" else "diamond", line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.14 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]])
        fig.update_yaxes(range=[lo[1] - 0.15 * (hi[1] - lo[1]), hi[1] + 0.15 * (hi[1] - lo[1])])
    return _base(fig, height)


def build_front(front, k=None, height=260):
    """Größe der Warteschlange nach jeder Festlegung (die Front): der Speicherbedarf der Suche."""
    x = np.arange(1, len(front) + 1)
    colors = ["#1f77b4" if k is None or i <= k else "#c8d6e5" for i in x]
    fig = go.Figure(go.Bar(x=x, y=front, marker_color=colors, hovertemplate="nach %{x} Festlegungen: %{y} in der Warteschlange<extra></extra>"))
    fig.update_layout(xaxis_title="festgelegte Knoten", yaxis_title="Knoten in der Warteschlange")
    return _base(fig, height)


def build_share_hist(share, height=300):
    fig = go.Figure(go.Histogram(x=np.asarray(share) * 100, marker_color=C.COLORS["dijkstra"], opacity=0.8, xbins=dict(start=0, end=100, size=5)))
    fig.add_vline(x=float(np.median(share)) * 100, line=dict(color="black", dash="dash"), annotation_text="Median", annotation_position="top")
    fig.update_layout(xaxis_title="festgelegter Anteil des Netzes, bevor das Ziel gefunden ist [%]", yaxis_title="Start-Ziel-Paare")
    return _base(fig, height)


def build_radius(rg, height=320):
    r, c = rg["radii"], rg["counts"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=c, mode="lines+markers", name="festgelegte Knoten", line=dict(color=C.COLORS["dijkstra"])))
    ref = c[3] * (r / r[3]) ** 2
    fig.add_trace(go.Scatter(x=r, y=ref, mode="lines", name="Steigung 2 (Kreisfläche)", line=dict(color="rgba(120,120,120,0.7)", dash="dash")))
    fig.update_layout(xaxis=dict(title="Kostenradius L (Meter)", type="log"), yaxis=dict(title="bis L festgelegte Knoten", type="log"))
    return _base(fig, height)


def build_queues(rows, height=340):
    fig = go.Figure()
    for q in ("array", "binary", "fibonacci", "dial"):
        sel = [r for r in rows if r["queue"] == q]
        fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[r["work"] for r in sel], mode="lines+markers", name=q, line=dict(color=QUEUE_COLORS[q])))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Schlüsselvergleiche (Dial: Eimerschritte)", type="log"))
    return _base(fig, height)


def build_negative(rows, height=320):
    x = [f"{r['fraction']:.0%}" for r in rows]
    fig = go.Figure()
    for key, name, color in (("right", "Dijkstra stimmt (zufällig)", "#2ca02c"), ("wrong", "Dijkstra liefert falsche Kosten", "#d62728"), ("cycle", "negativer Zyklus: keine kürzeste Route", "#7f7f7f")):
        fig.add_trace(go.Bar(x=x, y=[r[key] / r["total"] * 100 for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="stack", xaxis_title="Anteil negativer Kanten", yaxis_title="Zufallsnetze [%]", yaxis=dict(range=[0, 100]))
    return _base(fig, height)
