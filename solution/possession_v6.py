"""CEEA v6 -- v2 plus throw-in and goal-kick detection from the Laws of the Game.

    python solution/possession_v6.py                  # writes evaluation/results/CEEA_v6.csv

Where v2 misses a stoppage (measured against the automatic Smart Tagging, aggregates only), it is
almost always a throw-in or a goal kick: the ball IS visible and the players are NOT walking --
they are moving into position for the restart. So v6 looks at the ball, not at the players:

  throw-in    the ball stands still (< 0.5 m/s for >= 1 s) on a touchline. In play a ball does
              not stay still on the line.
  goal kick   the ball stands still inside a goal area AND no opponent is inside that penalty
              area. Law 16 (since 2019) keeps opponents out of the penalty area until the ball
              is in play, which is also what tells a goal kick from a goalkeeper holding a ball
              he has just caught (attackers are usually still in the box then).

The defending team at each end is the team of the player nearest that goal line (its keeper).
When a restart is found, the still period and the 2 s before it get strong dead-ball evidence.
Distances are pitch markings; nothing is fitted to any label.
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
from games import GAME, GAMES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
v1 = v2.v1
P = v2.P
P.update(
    rs_still_v=0.5,       # m/s: ball standing still ...
    rs_still_win=25,      # ... for at least 1 s
    rs_line=1.5,          # m from a touchline: throw-in position (ball held over the line)
    goal_area_x=6.5,      # goal area: 5.5 m deep ...
    goal_area_y=10.2,     # ... 9.16 m each side of the centre, +1 m tolerance
    pen_area_x=16.5,      # penalty area: 16.5 m deep ...
    pen_area_y=20.16,     # ... 20.16 m each side of the centre
    rs_before=50,         # frames (2 s) before the ball is placed are dead too
    w_restart=2.5,
)


def opponents_in_box(players, team, game=GAME):
    """Per frame and per end (0 = x small, 1 = x large): is any opponent of the defending team
    inside that penalty area? The defending team at an end owns the player nearest its goal line."""
    length, width = GAMES[game]["pitch"]
    px = players[..., 0].astype(np.float64) * length
    py = players[..., 1].astype(np.float64) * width
    out = []
    for end in (0, 1):
        depth = px if end == 0 else length - px
        depth = np.where(np.isnan(depth), np.inf, depth)
        keeper_team = team[np.argmin(depth, axis=1)]
        in_box = (depth < P["pen_area_x"]) & (np.abs(py - width / 2.0) < P["pen_area_y"])
        opponent = team[None, :] != keeper_team[:, None]
        out.append(np.any(in_box & opponent, axis=1))
    return out


def restart_evidence(ball, players, team, v_ball, game=GAME):
    length, width = GAMES[game]["pitch"]
    x = ball[:, 0].astype(np.float64) * length
    y = ball[:, 1].astype(np.float64) * width
    n = len(x)

    still = np.nan_to_num(v_ball, nan=99.0) < P["rs_still_v"]
    run = v1.run_length(still)
    placed = (run >= P["rs_still_win"]) & ~np.isnan(x)

    throw_in = (y < P["rs_line"]) | (y > width - P["rs_line"])
    near_centre = np.abs(y - width / 2.0) < P["goal_area_y"]
    opp0, opp1 = opponents_in_box(players, team, game)
    goal_kick = ((x < P["goal_area_x"]) & near_centre & ~opp0) | \
                ((x > length - P["goal_area_x"]) & near_centre & ~opp1)

    dead = np.zeros(n, dtype=bool)
    counts = dict(throw=0, goal=0)
    last_start = -1
    for i in np.flatnonzero(placed & (throw_in | goal_kick)):
        start = i - run[i] + 1
        if start != last_start:
            counts["throw" if throw_in[i] else "goal"] += 1
            last_start = start
        dead[max(0, start - P["rs_before"]):i + 1] = True
    return P["w_restart"] * dead, counts


def predict(game=GAME, verbose=True):
    players, ball_raw, team = v1.load_atd(game)
    ball, filled = v2.interpolate_ball(ball_raw)
    f = v1.measurements(players, ball, team, game)

    t_ev = v1.team_evidence(f) * np.where(filled, P["w_interp"], 1.0)
    rec_ev, _ = v2.reception_evidence(f)
    t_ev = t_ev + rec_ev
    t_ev = np.where(t_ev == 0, v2.shape_evidence(players, team, game), t_ev)

    f["out"] = f["out"] | v2.exit_out(ball_raw, game)
    brake_ev, _ = v2.brake_evidence(f)
    rs_ev, counts = restart_evidence(ball, players, team, f["v_ball"], game)
    d_ev = v1.dead_evidence(f) + brake_ev + rs_ev

    if verbose:
        print("  restarts found: {throw} throw-ins, {goal} goal kicks".format(**counts))

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
    parser.add_argument("--out", help="output path (default evaluation/results/<team>_v6.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}_v6.csv".format(args.team))

    labels = predict()
    v1.write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = v1.summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
