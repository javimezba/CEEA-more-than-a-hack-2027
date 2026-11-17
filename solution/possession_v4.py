"""CEEA v4 -- v2, with the team-shape cue switched off during transition phases.

    python solution/possession_v4.py                  # writes evaluation/results/CEEA_v4.csv

v2 showed that team shape helps a lot inside settled possessions but hurts right after a
turnover: a team needs a few seconds to open up after winning the ball and to close down after
losing it. v3 tried to undo that delay by shifting the signal; on the real match the measured
shift (9 s) was not physically credible and half of v2's gain was lost.

v4 keeps the shape exactly as in v2 but ignores it where it is known to be stale:

  pass 1   decode without the shape cue (control + receptions only) to find the changes of state
  gate     from 1 s before to 5 s after every change into H or A, the shape cue is switched off.
           5 s is the usual length of a transition phase in tactical analysis (the "five-second
           rule" of counter-pressing) -- a football convention, not a number fitted to this game
  pass 2   decode again with the shape cue everywhere else

No labels of any kind are used.
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
from games import GAME, half_video_blocks  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
v1 = v2.v1
P = v2.P
P.update(
    gate_before=25,     # frames (1 s) before a change: the change itself is only known to ~1 s
    gate_after=125,     # frames (5 s) after it: the transition phase
)


def decode(t_ev, d_ev, game=GAME):
    labels = np.full(len(t_ev), "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        forced = np.zeros(last - first + 1, dtype=bool)
        if half == 2:
            forced[:max(0, v1.SECOND_HALF_KICKOFF - first)] = True
        sl = slice(first, last + 1)
        labels[sl] = v1.STATES[v1.viterbi(t_ev[sl], d_ev[sl], forced)]
    return labels


def transition_gate(labels):
    """True from gate_before frames before to gate_after frames after each change into H or A."""
    n = len(labels)
    change = np.zeros(n, dtype=bool)
    change[1:] = (labels[1:] != labels[:-1]) & np.isin(labels[1:], ["H", "A"])
    gate = np.zeros(n, dtype=bool)
    for s in np.flatnonzero(change):
        gate[max(0, s - P["gate_before"]):min(n, s + P["gate_after"])] = True
    return gate, int(change.sum())


def predict(game=GAME, verbose=True):
    players, ball_raw, team = v1.load_atd(game)
    ball, filled = v2.interpolate_ball(ball_raw)
    f = v1.measurements(players, ball, team, game)

    control = v1.team_evidence(f) * np.where(filled, P["w_interp"], 1.0)
    rec_ev, n_rec = v2.reception_evidence(f)
    base = control + rec_ev

    f["out"] = f["out"] | v2.exit_out(ball_raw, game)
    brake_ev, n_brake = v2.brake_evidence(f)
    d_ev = v1.dead_evidence(f) + brake_ev

    first_pass = decode(base, d_ev, game)
    gate, n_changes = transition_gate(first_pass)

    shape = v2.shape_evidence(players, team, game)
    t_ev = np.where(base == 0, np.where(gate, 0.0, shape), base)
    labels = decode(t_ev, d_ev, game)

    if verbose:
        graded = labels != "X"
        print("  pass 1: {} changes into H/A; shape switched off on {:.0%} of graded frames".format(
            n_changes, float(np.mean(gate[graded]))))
        print("  cues: {} receptions, {} braking events".format(n_rec, n_brake))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="CEEA")
    parser.add_argument("--out", help="output path (default evaluation/results/<team>_v4.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}_v4.csv".format(args.team))

    labels = predict()
    v1.write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = v1.summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
