"""CEEA -- ball possession from automatic tracking, with no training data.

    python solution/possession.py                     # writes evaluation/results/CEEA.csv
    python solution/possession.py --team MYTEAM       # writes evaluation/results/MYTEAM.csv

The method in four steps
------------------------
1. Read the tactical ATD (players + ball, 25 fps) and turn it into a few physical measurements per
   frame: distance from the ball to the nearest player of each team, the ball's speed, whether the
   ball is outside the pitch, and how fast the players are moving as a group.
2. Turn those into two per-frame "evidence" scores:
     team evidence  > 0 means Barcelona (H) is controlling the ball, < 0 means Athletic (A).
                    It is only non-zero when a player is close to a ball that is moving slowly
                    enough to be under control. A ball in flight, or one that just bounces off
                    someone, gives no evidence -- so possession stays with the last team that had it.
     dead evidence  > 0 means the ball looks out of play: ball outside the lines, ball standing
                    still, players walking, ball unseen for a long time.
3. Find the single most likely sequence of states H / A / D for each half with the Viterbi
   algorithm. Every change of state has a fixed cost, so a state only changes when the evidence
   for the new one lasts long enough to pay for it. That is what removes the frame-to-frame
   flicker that destroys spellF1, without blurring the transitions the way a moving vote does.
4. Write one label per video frame; X outside the two halves (never graded).

Nothing is learned and nothing is fitted to the public ground truth. Every parameter in P below is
set from football and physics reasoning, written next to it. The public 5 minutes are only used by
evaluation/run_evaluation.py to check that the file is aligned and well formed.

numpy + pandas only. Deterministic: the same data always produces the same file.
"""

import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Parameters -- set a priori, not tuned on the public window (RULES.md forbids that).
# ---------------------------------------------------------------------------------------------
P = dict(
    # --- control: who has the ball -----------------------------------------------------------
    r_full=1.0,         # m. A player within 1 m of the ball is in contact range.
    r_ctrl=2.5,         # m. Beyond 2.5 m nobody is controlling it (a stride is ~2 m).
    team_scale=1.0,     # m of distance advantage over the nearest opponent per unit of evidence
    team_clip=3.0,      # max evidence one frame can give, so no single frame decides anything
    v_ctrl_ok=6.0,      # m/s. A dribbled ball moves at running speed (< ~6 m/s) ...
    v_ctrl_none=12.0,   # m/s. ... a ball faster than 12 m/s is a pass or a shot: nobody controls it

    # --- dead ball ---------------------------------------------------------------------------
    dead_bias=-0.5,     # by default the ball is in play
    speed_win=25,       # frames (+/- 1 s) to smooth the players' median speed
    v_dead=1.4,         # m/s. Walking pace: in play the median player jogs, at a stoppage he walks
    v_scale=0.5,        # m/s per unit of evidence
    w_speed=1.5,
    out_margin=0.5,     # m past the line before the ball counts as out (tracking noise)
    out_carry=250,      # frames (10 s). An out ball is usually lost by the tracker; keep "out"
    w_out=4.0,          # strongest single cue: ball beyond the lines is dead by the Laws
    still_speed=0.4,    # m/s. Ball standing still ...
    still_win=25,       # ... for at least 1 s: placed for a restart
    w_still=1.0,
    fast_ball=10.0,     # m/s. A ball this fast has just been kicked: in play
    w_fast=2.0,
    miss_long=75,       # frames (3 s) without seeing the ball before that itself hints at a stoppage
    w_miss=0.25,        # weak: the ball is unseen on 41% of graded frames, mostly during play

    # --- switch costs (Viterbi) --------------------------------------------------------------
    # The shortest change a cost allows is ~ 2 * cost / evidence-per-frame. With full evidence
    # (3 per frame):
    cost_team=18.0,       # H <-> A: 2*18/3 = 12 frames ~ 0.5 s of clean control by the other team
    cost_to_dead=40.0,    # alive -> D: ~1 s of clear stoppage evidence; stoppages last seconds
    cost_from_dead=40.0,  # D -> alive
)

FIELD_SEP, GROUP_SEP = ":", ";"
HOME, AWAY = 0, 1
STATES = np.array(["H", "A", "D"])

# Published by the organisers (FAQ.md, "Which frames are actually graded?"): the second half is
# graded from 110,808 but kicks off at 113,742, and every frame before the kickoff is D.
SECOND_HALF_KICKOFF = 113742


# ---------------------------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------------------------
def team_of_each_track(meta_path):
    """0 for each home track slot, 1 for each away slot. The first team in the XML is home."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def load_atd(game=GAME):
    """players (N, slots, 3) = x, y, speed and ball (N, 2) = x, y; 0-1 coordinates, NaN = missing.

    Parsed once with the pandas C reader and cached as .npz in the DATA folder -- never inside the
    repository, because it is derived match data.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    cache = os.path.join(DATA_DIR, "cache", "ceea_atd_tactical.npz")
    if os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["players"], z["ball"], z["team"]

    team = team_of_each_track(meta_path)
    n_slots = len(team)
    print("parsing {} (first run only, then cached) ...".format(os.path.basename(raw_path)))
    with open(raw_path) as handle:
        text = handle.read()
    n_ball = len(text[:text.index("\n")].split(FIELD_SEP)[2].split(","))
    text = text.replace(FIELD_SEP, ",").replace(GROUP_SEP, ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN", "nan", ""]).to_numpy()
    del text

    if table.shape[1] != 1 + 3 * n_slots + n_ball:
        raise ValueError("unexpected ATD layout: {} columns".format(table.shape[1]))
    frame_id = table[:, 0].astype(np.int64)
    n_frames = source_frames("tactical", game)
    if len(frame_id) != n_frames or frame_id[0] != 1 or np.any(np.diff(frame_id) != 1):
        raise ValueError("ATD is not one record per video frame numbered from 1")

    players = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3).astype(np.float32)
    ball = table[:, 1 + 3 * n_slots:3 + 3 * n_slots].astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, players=players, ball=ball, team=team)
    return players, ball, team


# ---------------------------------------------------------------------------------------------
# 2. Measurements and evidence
# ---------------------------------------------------------------------------------------------
def centred_nanmean(x, w):
    """Mean of the non-NaN values in a centred window of +/- w frames (NaN if there are none)."""
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    n = c[hi] - c[lo]
    out = np.full(len(x), np.nan)
    out[n > 0] = (s[hi] - s[lo])[n > 0] / n[n > 0]
    return out


def run_length(mask):
    """For each frame, how many consecutive True frames end there (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int64)
    run = 0
    for i, m in enumerate(mask.tolist()):
        run = run + 1 if m else 0
        out[i] = run
    return out


def measurements(players, ball, team, game=GAME):
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    px, py, ps = players[..., 0] * length, players[..., 1] * width, players[..., 2]
    bx = ball[:, 0].astype(np.float64) * length
    by = ball[:, 1].astype(np.float64) * width
    ball_ok = ~np.isnan(bx)

    # nearest home / away player to the ball, in metres (inf when unknown)
    d = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    d = np.where(np.isnan(d), np.inf, d)
    d_home = np.where(team[None, :] == HOME, d, np.inf).min(axis=1)
    d_away = np.where(team[None, :] == AWAY, d, np.inf).min(axis=1)

    # how fast the players move as a group (median over tracked players)
    with np.errstate(all="ignore"):
        v_players = np.nanmedian(ps, axis=1)   # all-NaN at halftime -> NaN, handled below

    # ball speed in m/s between consecutive frames, smoothed over +/- 3 frames
    v_ball = np.full(len(bx), np.nan)
    v_ball[1:] = np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2) * fps
    v_ball = centred_nanmean(v_ball, 3)

    # ball beyond the lines; kept "out" while the tracker has lost it, for up to out_carry frames
    m = P["out_margin"]
    out_now = ball_ok & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    out = np.zeros(len(bx), dtype=bool)
    since, state = 10 ** 9, False
    for i, (seen, o) in enumerate(zip(ball_ok.tolist(), out_now.tolist())):
        if seen:
            state, since = o, 0
        else:
            since += 1
        out[i] = state and since <= P["out_carry"]

    return dict(d_home=d_home, d_away=d_away, ball_ok=ball_ok, v_players=v_players,
                v_ball=v_ball, out=out)


def team_evidence(f):
    """> 0: Barcelona controls the ball; < 0: Athletic; 0: nobody (or no ball)."""
    d_near = np.minimum(f["d_home"], f["d_away"])
    near = np.clip((P["r_ctrl"] - d_near) / (P["r_ctrl"] - P["r_full"]), 0.0, 1.0)
    vb = np.nan_to_num(f["v_ball"], nan=0.0)
    slow = np.clip((P["v_ctrl_none"] - vb) / (P["v_ctrl_none"] - P["v_ctrl_ok"]), 0.0, 1.0)
    with np.errstate(invalid="ignore"):
        advantage = np.where(np.isfinite(f["d_home"]) & np.isfinite(f["d_away"]),
                             (f["d_away"] - f["d_home"]) / P["team_scale"],
                             np.where(np.isfinite(f["d_home"]), P["team_clip"], -P["team_clip"]))
    advantage = np.clip(np.nan_to_num(advantage), -P["team_clip"], P["team_clip"])
    return np.where(f["ball_ok"], advantage * near * slow, 0.0)


def dead_evidence(f):
    """> 0: the ball looks out of play."""
    v = centred_nanmean(f["v_players"], P["speed_win"])
    v = np.where(np.isnan(v), P["v_dead"], v)
    dead = P["dead_bias"] + P["w_speed"] * np.clip((P["v_dead"] - v) / P["v_scale"], -2, 2)
    dead = dead + P["w_out"] * f["out"]
    vb = f["v_ball"]
    still = run_length(np.nan_to_num(vb, nan=99.0) < P["still_speed"]) >= P["still_win"]
    dead = dead + P["w_still"] * still
    dead = dead - P["w_fast"] * (np.nan_to_num(vb, nan=0.0) > P["fast_ball"])
    dead = dead + P["w_miss"] * (run_length(~f["ball_ok"]) >= P["miss_long"])
    return dead


# ---------------------------------------------------------------------------------------------
# 3. Decoding
# ---------------------------------------------------------------------------------------------
def viterbi(team_ev, dead_ev, forced_dead):
    """Highest-scoring H/A/D path. Per-frame score: H +t/2, A -t/2, D dead; minus switch costs."""
    n = len(team_ev)
    big = 1e4
    eH = np.where(forced_dead, -big, 0.5 * team_ev).tolist()
    eA = np.where(forced_dead, -big, -0.5 * team_ev).tolist()
    eD = np.where(forced_dead, big, dead_ev).tolist()
    cT, cTD, cDT = P["cost_team"], P["cost_to_dead"], P["cost_from_dead"]

    back = np.zeros((n, 3), dtype=np.int8)
    sH, sA, sD = eH[0], eA[0], eD[0]
    for i in range(1, n):
        h = (sH, sA - cT, sD - cDT)
        a = (sH - cT, sA, sD - cDT)
        dd = (sH - cTD, sA - cTD, sD)
        bH = 0 if h[0] >= h[1] and h[0] >= h[2] else (1 if h[1] >= h[2] else 2)
        bA = 1 if a[1] >= a[0] and a[1] >= a[2] else (0 if a[0] >= a[2] else 2)
        bD = 2 if dd[2] >= dd[0] and dd[2] >= dd[1] else (0 if dd[0] >= dd[1] else 1)
        sH, sA, sD = h[bH] + eH[i], a[bA] + eA[i], dd[bD] + eD[i]
        back[i] = (bH, bA, bD)

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax((sH, sA, sD)))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def predict(game=GAME):
    players, ball, team = load_atd(game)
    f = measurements(players, ball, team, game)
    t_ev, d_ev = team_evidence(f), dead_evidence(f)

    labels = np.full(len(t_ev), "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        forced = np.zeros(last - first + 1, dtype=bool)
        if half == 2:
            forced[:max(0, SECOND_HALF_KICKOFF - first)] = True
        sl = slice(first, last + 1)
        labels[sl] = STATES[viterbi(t_ev[sl], d_ev[sl], forced)]
    return labels


# ---------------------------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------------------------
def write_submission(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summary(labels, fps=25):
    """Possession count and mean length over the whole match -- the file's own claims."""
    graded = labels != "X"
    alive = np.isin(labels, ["H", "A"])
    new = alive.copy()
    new[1:] &= ~((labels[1:] == labels[:-1]) & alive[:-1])
    n_poss = int(np.sum(new & graded))
    return n_poss, alive.sum() / float(fps) / max(1, n_poss), float(np.mean(labels[graded] == "D"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="CEEA", help="team name: the CSV is named after it")
    parser.add_argument("--out", help="output path (default evaluation/results/<team>.csv)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}.csv".format(args.team))

    labels = predict()
    write_submission(out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    n_poss, mean_s, d_share = summary(labels)
    print("wrote {}".format(out))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  full match: {} possessions, mean {:.1f} s, ball dead on {:.0%} of graded frames".format(
        n_poss, mean_s, d_share))
    print("\nnext:  python evaluation/validate_submission.py \"{}\"".format(out))
