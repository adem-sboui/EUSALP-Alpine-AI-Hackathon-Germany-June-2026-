"""Cooperative graph + strategy suggestions based on station similarity."""
import math

FEATURES = [
    ("altitude_m", 3000),
    ("snow_trend_days_per_decade", 15),
    ("energy_cost_index", 100),
    ("closure_days_last_season", 30),
]


def _vec(row):
    return [abs(row[c]) / scale for c, scale in FEATURES]


def similarity(a, b):
    d = math.dist(_vec(a), _vec(b))
    return round(1 / (1 + d), 3)


def similar_stations(df, name, k=3):
    target = df[df.station == name].iloc[0]
    rows = []
    for _, r in df[df.station != name].iterrows():
        rows.append((r["station"], similarity(target, r)))
    return sorted(rows, key=lambda x: -x[1])[:k]


def edges(df, threshold=0.8, cross_region_only=True):
    out = []
    rows = [r for _, r in df.iterrows()]
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if cross_region_only and a["region"] == b["region"]:
                continue
            s = similarity(a, b)
            if s >= threshold:
                out.append((a["station"], b["station"], s))
    return out


def to_dot(df, threshold=0.8):
    lines = ["graph G {", "node [shape=ellipse, style=filled, fillcolor=lightblue];"]
    for _, r in df.iterrows():
        lines.append(f'"{r["station"]}" [label="{r["station"]}\\n{r["region"]}"];')
    for a, b, s in edges(df, threshold):
        lines.append(f'"{a}" -- "{b}" [label="{s}"];')
    lines.append("}")
    return "\n".join(lines)


def strategy_suggestions(df, name, k=3):
    """Suggest diversification measures that worked for similar stations."""
    out = []
    for other, sim in similar_stations(df, name, k=len(df)):
        r = df[df.station == other].iloc[0]
        if r["outcome"] == "positive" and r["diversification"] != "none yet":
            out.append({
                "measure": r["diversification"],
                "from_station": other,
                "region": r["region"],
                "similarity": sim,
            })
        if len(out) == k:
            break
    return out
