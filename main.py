"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Reads one case (/input/<id>.in) and writes /output/<id>.out: for each of
hours 2..24 of the UTC release day, five quantiles (5/25/50/75/95%) of the
Twitch "Grand Theft Auto VI" category's concurrent viewers.

Method (fitted at run time from the case's own comparables, so the final
case's extra comparables are used automatically):

  A launch day's shape depends strongly on how awake its category already
  is in hour 1.  Games whose hour-1 print is tiny (a category asleep before
  a mid-day unlock) grow 10-200x by the evening; games whose hour 1 is
  already huge (US prime-time hype + a rolling-midnight unlock) dip through
  the small hours and rise only ~1.5-2x to the evening peak.  So, per hour
  h, we regress the comparables' log ratio ln(y_h / y_1) on ln(y_1)
  (ordinary least squares, one line per hour) and take the fitted value at
  this case's ln(p1) as the median log-ratio.  The spread is honest too:
  the residuals of the K=12 comparables nearest in ln(y_1) supply the
  empirical quantiles, so a huge hour 1 gets a tighter band than a small
  one.  Quantiles: q_tau(h) = p1 * exp(median_h + resid_tau(h)).

  Fallbacks (never an invalid case): too few comparables for a fit -> the
  pooled empirical log-ratio quantiles; no comparables at all -> the house
  persistence baseline (flat at p1, spread 0.15*sqrt(h-1)).

Validation (local, on the four practice launches' real SullyGnome curves,
training only on the 37 comparables the practice cases carry):
  house baseline pinball loss 0.387, this model 0.213 (k=12).
  Leave-one-out over the 37 comparables: baseline 0.930, this model 0.405.

Python 3.13, standard library only, no network, deterministic: the
comparables are sorted by name before any sum, so two runs on the same
case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The spread of an hour's move, in natural-log units per square-root hour
# (baseline fallback only, same as the house starter).
SIGMA = 0.15
# Comparables nearest in log hour-1 whose residuals set the spread.
K_NEIGHBORS = 12
# Minimum points to fit the per-hour log-log line; below that, fall back
# to the pooled empirical quantiles, and below 3 points, to the baseline.
MIN_FIT_POINTS = 8
MIN_POOLED_POINTS = 3


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(xs, tau):
    """Linear-interpolation empirical quantile of a non-empty list."""
    xs = sorted(xs)
    n = len(xs)
    if n == 1:
        return xs[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def linreg(xs, ys):
    """Ordinary least squares y ~ a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2022-02-25T00:00:00Z."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def release_day_viewers(comp):
    """A comparable's release-day hourly viewers, hours 1..24 (None if missing).

    comp has `day_start` (00:00 UTC of its release date) and `hourly`, a
    list of {from, viewers, channels} covering 14 days before the release
    to 3 days after.  A point at `from` covers [from, from + 1h).
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and isinstance(viewers, (int, float)):
            by_hour[h] = viewers
    ys = [by_hour.get(h) for h in range(1, 25)]
    if not ys[0] or ys[0] <= 0:
        return None
    return ys


def hour_quantiles(p1, h, levels, curves):
    """The five quantiles of hour h, from the comparables' curves.

    curves: release-day viewer lists (hour 1..24), sorted by game name so
    every sum and tie-break is deterministic.
    """
    log_p1 = math.log(p1)
    pts = []  # (log y1, log ratio y_h / y1) for comparables with hour h
    for ys in curves:
        y1 = ys[0]
        yh = ys[h - 1]
        if yh is not None and yh > 0:
            pts.append((math.log(y1), math.log(yh / y1)))
    if len(pts) >= MIN_FIT_POINTS:
        a, b = linreg([p[0] for p in pts], [p[1] for p in pts])
        median = a + b * log_p1
        near = sorted(pts, key=lambda p: abs(p[0] - log_p1))[:K_NEIGHBORS]
        resid = [p[1] - (a + b * p[0]) for p in near]
        return [p1 * math.exp(median + quantile(resid, tau)) for tau in levels]
    if len(pts) >= MIN_POOLED_POINTS:
        ratios = [p[1] for p in pts]
        return [p1 * math.exp(quantile(ratios, tau)) for tau in levels]
    # No usable comparables: the house persistence baseline.
    return [p1 * math.exp(norm_ppf(tau) * SIGMA * math.sqrt(h - 1)) for tau in levels]


def hour_one_print(case):
    """Hour 1's viewer count, falling back to the last history point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    history = case.get("history") or []
    for row in reversed(history):
        v = row.get("viewers") if isinstance(row, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 1.0


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    p1 = max(1.0, hour_one_print(case))
    curves = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", ""))):
        ys = release_day_viewers(comp)
        if ys is not None:
            curves.append(ys)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in rows:
        h = int(row["hour"])
        if not 2 <= h <= 24:
            continue
        qs = hour_quantiles(p1, h, levels, curves)
        # Valid output: finite, above 0, at most 1e9, non-decreasing.
        clean = []
        for q in qs:
            if not math.isfinite(q) or q <= 0:
                q = p1
            q = min(q, 1e9)
            clean.append(round2(q))
        clean.sort()
        hours.append({"hour": h, "q": clean})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
