"""Demo risk index (RI) for a ski station.

IMPORTANT: this is a transparent weighted formula on synthetic data.
It is NOT a trained model and not validated against real closures.
"""

WEIGHTS = {"snow": 0.35, "energy": 0.25, "closures": 0.25, "altitude": 0.15}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def components(snow_trend, energy_cost, closure_days, altitude_m):
    """Each component is scaled to 0 (low risk) .. 1 (high risk)."""
    return {
        "snow": _clamp(-snow_trend / 15),
        "energy": _clamp(energy_cost / 100),
        "closures": _clamp(closure_days / 30),
        "altitude": _clamp((2400 - altitude_m) / 1400),
    }


def ri_score(snow_trend, energy_cost, closure_days, altitude_m):
    comps = components(snow_trend, energy_cost, closure_days, altitude_m)
    return round(100 * sum(WEIGHTS[k] * v for k, v in comps.items()), 1)


def score_row(row):
    return ri_score(
        row["snow_trend_days_per_decade"],
        row["energy_cost_index"],
        row["closure_days_last_season"],
        row["altitude_m"],
    )
