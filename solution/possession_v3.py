"""CEEA v3 -- v2 with the team-shape cue shifted to undo its delay.

    python solution/possession_v3.py                  # writes evaluation/results/CEEA_v3.csv

v2 showed (solution/check.py against the automatic Smart Tagging) that team shape was the only
new cue with a real effect: much better in settled phases, worse right after a turnover. The
reason is physical: a team needs a few seconds to open up after winning the ball and to close
down after losing it, so shape(t) describes the possession of a few seconds earlier.

The match is decoded offline, so v3 simply reads the shape `lag` frames AHEAD: the evidence for
frame t comes from the shape at t + lag. The lag is not chosen by hand and not fitted to any
label: it is the shift that best lines up the shape signal with v1's own control evidence (who
is next to a slow ball), measured over the whole match. That is self-supervised -- no human
label is involved -- which RULES.md explicitly allows.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import possession_v2 as v2  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
v1 = v2.v1
P = v2.P
P.update(
    lag_max=250,        # search shifts up to 10 s either way
    lag_step=5,         # in steps of 0.2 s
    ctrl_min=0.5,       # frames whose control evidence is at least this clear are the reference
)


def spread_diff(players, team, game=GAME):
    """Spread of Barcelona minus spread of Athletic, in metres (NaN where unknown)."""
    length, width = GAMES[game]["pitch"]
    px, py = players[..., 0] * length, players[..., 1] * width
    spread = []
    with np.errstate(all="ignore"):
        for side in (v1.HOME, v1.AWAY):
            x, y = px[:, team == side], py[:, team == side]
            cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
            spread.append(np.sqrt(np.nanmean((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2,
                                             axis=1)))
    return spread[0] - spread[1]


def estimate_lag(diff, control, game=GAME):
    """Shift (frames) that best aligns shape(t + lag) with the control evidence at t.

    Correlation between the sign of clear control evidence and the standardised spread
    difference, computed inside each half only. Returns (best lag, its correlation, correlation
    at lag 0) so the effect can be reported.
    """
    graded = np.zeros(len(diff), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    c = np.where(np.abs(control) >= P["ctrl_min"], np.sign(control), 0.0)
    s = diff - np.nanmean(diff[graded])
    s = s / np.nanstd(s[graded])
    s = np.where(graded, s, np.nan)

    def corr(k):
        if k >= 0:
            a, b = c[:len(c) - k], s[k:]
        else:
            a, b = c[-k:], s[:len(s) + k]
        ok = (a != 0) & ~np.isnan(b)
        return float(np.mean(a[ok] * b[ok])) if ok.any() else -np.inf

    lags = range(-P["lag_max"], P["lag_max"] + 1, P["lag_step"])
    scores = {k: corr(k) for k in lags}
    best = max(scores, key=lambda k: (scores[k], -abs(k)))
    return best, scores[best], scores[0]


def shift(x, k):
    """y[t] = x[t + k], NaN where out of range."""
    y = np.full(len(x), np.nan)
    if k >= 0:
        y[:len(x) - k] = x[k:]
    else:
        y[-k:] = x[:len(x) + k]
    return y


def predict(game=GAME, verbose=True):
    players, ball_raw, team = v1.load_atd(game)
    ball, filled = v2.interpolate_ball(ball_raw)
    f = v1.measurements(players, ball, team, game)

    control = v1.team_evidence(f) * np.where(filled, P["w_interp"], 1.0)
    rec_ev, n_rec = v2.reception_evidence(f)
    t_ev = control + rec_ev

    diff = spread_diff(players, team, game)
    lag, c_best, c_zero = estimate_lag(diff, control, game)
    shape = P["w_shape"] * np.tanh(np.nan_to_num(shift(diff, lag), nan=0.0) / P["shape_scale"])
    t_ev = np.where(t_ev == 0, shape, t_ev)

    f["out"] = f["out"] | v2.exit_out(ball_raw, game)
    brake_ev, n_brake = v2.brake_evidence(f)
    d_ev = v1.dead_evidence(f) + brake_ev

    if verbose:
        print("  shape lag: {:+d} frames ({:+.1f} s); alignment with control {:.3f} "
              "(was {:.3f} at lag 0)".format(lag, lag / 25.0, c_best, c_zero))
        print("  cues: {} receptions, {} braking events".format(n_rec, n_brake))

    labels = np.full(len(t_ev), "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        forced = np.zeros(last - first + 1, dtype=bool)
        if half == 2:
            forced[:max(0, v1.SECOND_HALF_KICKOFF - first)] = True
        sl = slice(first, last + 1)
        labels[sl] = v1.STATES[v1.viterbi(t_ev[sl], d_ev[sl], forced)]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="CEEA")
    parser.add_argument("--out", help="output path (default evaluation/results/<team>_v3.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}_v3.csv".format(args.team))

    labels = predict()
    v1.write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = v1.summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
