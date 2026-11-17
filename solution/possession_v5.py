"""CEEA v5 -- v2 plus restart detection and a cleaner ball speed.

    python solution/possession_v5.py                  # writes evaluation/results/CEEA_v5.csv

Two additions to v2, both aimed at the weakest part of the model:

  1. Cleaner ball speed. The ball speed in v1/v2 is a difference of raw positions, so one jump of
     the tracker looks like a 20 m/s ball. v5 median-filters the ball position over 5 frames
     before differentiating. Every cue that uses ball speed benefits: control (slow ball), ball
     still, fast ball, receptions.

  2. Restarts. A stoppage in football ends with the ball placed and still at a place the Laws of
     the Game dictate. v5 looks for the ball standing still for >= 1 s at one of them:
       throw-in     on a touchline
       corner       in a corner
       goal kick    in a goal area
       kick-off     on the centre spot, every player in his own half
     and, for all of them, the players walking (a team does not walk while the ball is in play).
     When it finds one, the whole still period and the 2 s before it get strong dead-ball
     evidence. A missed stoppage merges two possessions into one, and spellF1 then loses both.

All thresholds come from the pitch markings (IFAB Laws of the Game) or from physics; none is fitted
to any label.
"""

import argparse
import os
import sys
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import possession_v2 as v2  # noqa: E402
from games import GAME, GAMES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
v1 = v2.v1
P = v2.P
P.update(
    # 1. ball speed
    ball_median=5,       # frames in the median filter of the ball position
    ball_mean=2,         # then +/- 2 frames of averaging of the speed

    # 2. restarts (distances in metres, from the pitch markings)
    rs_still_v=0.5,      # m/s: ball standing still ...
    rs_still_win=25,     # ... for at least 1 s
    rs_line=1.5,         # within 1.5 m of a touchline: throw-in position
    rs_corner=2.0,       # within 2 m of both lines: corner (the arc has a 1 m radius)
    rs_goal_area_x=6.5,  # goal area is 5.5 m deep ...
    rs_goal_area_y=10.2, # ... and 18.32 m wide (9.16 m each side of centre), +1 m of tolerance
    rs_centre=2.0,       # within 2 m of the centre spot
    rs_own_half=0.9,     # share of each team's players in its own half at a kick-off
    rs_walk=1.4,         # m/s: players walking -- the same threshold v1 uses for a stoppage
    rs_before=50,        # frames (2 s) before the ball is placed are dead too
    w_restart=2.5,
)


def ball_speed(ball, game=GAME):
    """Speed (m/s) from the median-filtered ball position."""
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    k = P["ball_median"]
    pos = []
    for col, scale in ((0, length), (1, width)):
        x = ball[:, col].astype(np.float64) * scale
        pad = np.concatenate((np.full(k // 2, np.nan), x, np.full(k // 2, np.nan)))
        xm = np.nanmedian(sliding_window_view(pad, k), axis=1)
        pos.append(np.where(np.isnan(x), np.nan, xm))
    x, y = pos
    v = np.full(len(x), np.nan)
    v[1:-1] = np.sqrt((x[2:] - x[:-2]) ** 2 + (y[2:] - y[:-2]) ** 2) * fps / 2.0
    return v1.centred_nanmean(v, P["ball_mean"])


def restart_evidence(ball, players, team, v_ball, v_players, game=GAME):
    length, width = GAMES[game]["pitch"]
    x = ball[:, 0].astype(np.float64) * length
    y = ball[:, 1].astype(np.float64) * width
    n = len(x)

    still = np.nan_to_num(v_ball, nan=99.0) < P["rs_still_v"]
    run = v1.run_length(still)
    placed = run >= P["rs_still_win"]

    on_touchline = (y < P["rs_line"]) | (y > width - P["rs_line"])
    corner = (((x < P["rs_corner"]) | (x > length - P["rs_corner"])) &
              ((y < P["rs_corner"]) | (y > width - P["rs_corner"])))
    goal_area = (((x < P["rs_goal_area_x"]) | (x > length - P["rs_goal_area_x"])) &
                 (np.abs(y - width / 2.0) < P["rs_goal_area_y"]))
    centre = np.hypot(x - length / 2.0, y - width / 2.0) < P["rs_centre"]

    walking = np.nan_to_num(v1.centred_nanmean(v_players, 25), nan=99.0) < P["rs_walk"]
    px = players[..., 0] * length
    left_home = np.nanmean(np.where(np.isnan(px[:, team == v1.HOME]), np.nan,
                                    px[:, team == v1.HOME] < length / 2.0), axis=1)
    left_away = np.nanmean(np.where(np.isnan(px[:, team == v1.AWAY]), np.nan,
                                    px[:, team == v1.AWAY] < length / 2.0), axis=1)
    q = P["rs_own_half"]
    split = (((left_home >= q) & (left_away <= 1 - q)) | ((left_home <= 1 - q) & (left_away >= q)))
    split = np.nan_to_num(split, nan=0).astype(bool)

    spot = (on_touchline | corner | goal_area | (centre & split)) & walking
    event = placed & spot & ~np.isnan(x)

    dead = np.zeros(n, dtype=bool)
    kinds = dict(throw=0, corner=0, goal=0, kickoff=0)
    last_start = -1
    for i in np.flatnonzero(event):
        start = i - run[i] + 1
        if start != last_start:
            kinds["corner" if corner[i] else "throw" if on_touchline[i] else
                  "kickoff" if centre[i] else "goal"] += 1
            last_start = start
        dead[max(0, start - P["rs_before"]):i + 1] = True
    return P["w_restart"] * dead, kinds


def predict(game=GAME, verbose=True):
    players, ball_raw, team = v1.load_atd(game)
    ball, filled = v2.interpolate_ball(ball_raw)
    f = v1.measurements(players, ball, team, game)
    f["v_ball"] = ball_speed(ball, game)                    # 1. cleaner speed everywhere

    t_ev = v1.team_evidence(f) * np.where(filled, P["w_interp"], 1.0)
    rec_ev, n_rec = v2.reception_evidence(f)
    t_ev = t_ev + rec_ev
    t_ev = np.where(t_ev == 0, v2.shape_evidence(players, team, game), t_ev)

    f["out"] = f["out"] | v2.exit_out(ball_raw, game)
    brake_ev, n_brake = v2.brake_evidence(f)
    rs_ev, kinds = restart_evidence(ball, players, team, f["v_ball"], f["v_players"], game)
    d_ev = v1.dead_evidence(f) + brake_ev + rs_ev           # 2. restarts

    if verbose:
        print("  cues: {} receptions, {} braking events".format(n_rec, n_brake))
        print("  restarts found: {throw} throw-ins, {corner} corners, {goal} goal kicks, "
              "{kickoff} kick-offs".format(**kinds))

    labels = np.full(len(t_ev), "X", dtype="<U1")
    for half, first, last in v2.half_video_blocks(game):
        forced = np.zeros(last - first + 1, dtype=bool)
        if half == 2:
            forced[:max(0, v1.SECOND_HALF_KICKOFF - first)] = True
        sl = slice(first, last + 1)
        labels[sl] = v1.STATES[v1.viterbi(t_ev[sl], d_ev[sl], forced)]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="CEEA")
    parser.add_argument("--out", help="output path (default evaluation/results/<team>_v5.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}_v5.csv".format(args.team))

    labels = predict()
    v1.write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = v1.summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
