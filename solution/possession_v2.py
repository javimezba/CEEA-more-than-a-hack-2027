"""CEEA v2 -- v1 plus five physically motivated cues, still with no training data.

    python solution/possession_v2.py                  # writes evaluation/results/CEEA_v2.csv

Everything from solution/possession.py (v1) is reused. v2 adds:

  possession changes
    1. short ball gaps (<= 1 s) are filled by linear interpolation, at half weight
    2. receptions: the ball arrives fast and stops dead next to a player -> that team takes control
    3. team shape: the team in possession spreads out, the other one closes up. A weak cue, used
       only on frames where nobody is controlling the ball (ball unseen or in flight)
  stoppages
    4. ball lost near a line while travelling towards it -> it went out
    5. collective braking: when the referee whistles, all players slow down together

Every new parameter is set a priori and explained next to it; none is tuned on the public window.
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
import possession as v1  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Halftime has no tracked players, so means and medians over empty slices warn there. Harmless.
warnings.filterwarnings("ignore", category=RuntimeWarning)

P = v1.P  # shared: v1's functions read the same dictionary
P.update(
    # 1. interpolation
    interp_max_gap=25,   # frames (1 s): a ball cannot do much unseen in one second
    w_interp=0.5,        # interpolated positions are guesses: half weight

    # 2. receptions
    rec_v_in=6.0,        # m/s: the ball arrives faster than a dribble ...
    rec_v_out=3.0,       # ... and leaves slower than walking pace: it has been stopped
    rec_win=6,           # frames (~0.25 s) before / after used to measure that
    rec_r=2.0,           # m: and a player is right there
    rec_hold=15,         # frames (0.6 s) of extra evidence for that player's team ...
    w_rec=2.0,           # ... 2 per frame = 30 in total, below 2 * cost_team (36): a reception
                         # alone cannot cause a brief switch, it only fixes the timing of a real one

    # 3. team shape
    shape_scale=3.0,     # m of difference in spread for a full-strength cue
    w_shape=0.2,         # weak: 3 s of shape alone (75 * 0.2 = 15) cannot beat cost_team (18)

    # 4. ball leaving the pitch
    exit_near=2.0,       # m from a line when last seen ...
    exit_v=2.0,          # ... moving towards that line faster than 2 m/s

    # 5. collective braking
    brake_win=25,        # frames (1 s) before vs after
    brake_drop=1.5,      # m/s drop in the players' median speed within ~1 s
    brake_hold=50,       # frames (2 s) of extra dead evidence after it
    w_brake=1.0,
)


def interpolate_ball(ball):
    """Fill gaps of at most interp_max_gap frames linearly. Returns (ball, filled_mask)."""
    n = len(ball)
    x = ball[:, 0].astype(np.float64)
    y = ball[:, 1].astype(np.float64)
    ok = ~np.isnan(x)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(ok, idx, -1))
    nxt = np.minimum.accumulate(np.where(ok, idx, n)[::-1])[::-1]
    fill = ~ok & (prev >= 0) & (nxt < n) & ((nxt - prev - 1) <= P["interp_max_gap"])
    out = ball.copy()
    out[fill, 0] = np.interp(idx[fill], idx[ok], x[ok])
    out[fill, 1] = np.interp(idx[fill], idx[ok], y[ok])
    return out, fill


def trailing_mean(x, w):
    """Mean of the non-NaN values in frames [i - w, i - 1]."""
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    n = c[i] - c[lo]
    out = np.full(len(x), np.nan)
    out[n > 0] = (s[i] - s[lo])[n > 0] / n[n > 0]
    return out


def leading_mean(x, w):
    """Mean of the non-NaN values in frames [i + 1, i + w]."""
    return trailing_mean(x[::-1], w)[::-1]


def hold(events, frames):
    """True on each event frame and the `frames - 1` frames after it."""
    c = np.concatenate(([0], np.cumsum(events)))
    i = np.arange(len(events))
    return (c[i + 1] - c[np.maximum(0, i + 1 - frames)]) > 0


def reception_evidence(f):
    v = f["v_ball"]
    before, after = trailing_mean(v, P["rec_win"]), leading_mean(v, P["rec_win"])
    d_near = np.minimum(f["d_home"], f["d_away"])
    rec = (np.nan_to_num(before, nan=0.0) > P["rec_v_in"]) & \
          (np.nan_to_num(after, nan=99.0) < P["rec_v_out"]) & (d_near <= P["rec_r"])
    rec[1:] &= ~rec[:-1]                                   # one event per reception
    side = np.where(f["d_home"] < f["d_away"], 1.0, -1.0)
    ev = np.zeros(len(v))
    for i in np.flatnonzero(rec):
        ev[i:i + P["rec_hold"]] += side[i] * P["w_rec"]
    return ev, int(rec.sum())


def shape_evidence(players, team, game=GAME):
    """> 0 when Barcelona is more spread out than Athletic (the team in possession opens up)."""
    length, width = GAMES[game]["pitch"]
    px, py = players[..., 0] * length, players[..., 1] * width
    spread = []
    with np.errstate(all="ignore"):
        for side in (v1.HOME, v1.AWAY):
            x, y = px[:, team == side], py[:, team == side]
            cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
            spread.append(np.sqrt(np.nanmean((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2,
                                             axis=1)))
    diff = np.nan_to_num(spread[0] - spread[1], nan=0.0)
    return P["w_shape"] * np.tanh(diff / P["shape_scale"])


def exit_out(ball, game=GAME):
    """Frames where the ball was last seen near a line, moving towards it, and is now unseen."""
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    x = ball[:, 0].astype(np.float64) * length
    y = ball[:, 1].astype(np.float64) * width
    ok = ~np.isnan(x)
    n = len(x)
    out = np.zeros(n, dtype=bool)
    exits = np.flatnonzero(ok[:-1] & ~ok[1:])
    near, vmin = P["exit_near"], P["exit_v"]
    for i in exits:
        j = i - 3
        if j < 0 or not ok[j]:
            continue
        vx, vy = (x[i] - x[j]) * fps / 3.0, (y[i] - y[j]) * fps / 3.0
        leaving = ((x[i] < near and vx < -vmin) or (length - x[i] < near and vx > vmin) or
                   (y[i] < near and vy < -vmin) or (width - y[i] < near and vy > vmin))
        if leaving:
            k = i + 1
            stop = min(n, i + 1 + P["out_carry"])
            while k < stop and not ok[k]:
                out[k] = True
                k += 1
    return out


def brake_evidence(f):
    v = v1.centred_nanmean(f["v_players"], 5)
    drop = trailing_mean(v, P["brake_win"]) - leading_mean(v, P["brake_win"])
    events = np.nan_to_num(drop, nan=0.0) > P["brake_drop"]
    events[1:] &= ~events[:-1]
    return P["w_brake"] * hold(events, P["brake_hold"]), int(events.sum())


def predict(game=GAME, verbose=True):
    players, ball_raw, team = v1.load_atd(game)
    ball, filled = interpolate_ball(ball_raw)
    f = v1.measurements(players, ball, team, game)

    t_ev = v1.team_evidence(f) * np.where(filled, P["w_interp"], 1.0)
    rec_ev, n_rec = reception_evidence(f)
    t_ev = t_ev + rec_ev
    t_ev = np.where(t_ev == 0, shape_evidence(players, team, game), t_ev)

    f["out"] = f["out"] | exit_out(ball_raw, game)
    d_ev = v1.dead_evidence(f)
    brake_ev, n_brake = brake_evidence(f)
    d_ev = d_ev + brake_ev

    if verbose:
        print("  cues: {} ball frames interpolated, {} receptions, {} braking events, "
              "{} frames of ball-exit".format(int(filled.sum()), n_rec, n_brake,
                                               int(exit_out(ball_raw, game).sum())))

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
    parser.add_argument("--out", help="output path (default evaluation/results/<team>_v2.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}_v2.csv".format(args.team))

    labels = predict()
    v1.write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = v1.summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
