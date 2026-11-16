"""Compare submissions WITHOUT any ground truth -- using only automatic, non-human signals.

    python solution/check.py evaluation/results/CEEA.csv evaluation/results/CEEA_v2.csv

RULES.md forbids choosing between versions on the public 5 minutes. This script compares them
over the WHOLE match against the Smart Tagging instead, which Metrica generates automatically
(no human labels, so it is allowed as a reference). It is noisy, and its phases are not
possessions, so read it as a consistency check, not as a score:

  structure      possessions, mean length, share of dead ball, share of very short possessions
  restarts       share of D in the 3 s before each SET PIECES tag starts (should be high)
  phases         for every team-tagged phase code: share of our alive frames inside the phase
                 that give the ball to the tagged team. Attacking phases should be high,
                 defending phases low. Higher contrast between the two = more consistent labels.

Prints aggregates only.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from starter.load_smart_tagging import load as load_tags  # noqa: E402

FPS = 25


def lower(s):
    return (s or "").strip().lower()


def side_of(team):
    t = lower(team)
    return "H" if "barcelona" in t else ("A" if "atl" in t else None)


def structure(lab):
    graded = lab != "X"
    alive = np.isin(lab, ["H", "A"])
    starts = np.flatnonzero(alive & np.concatenate(([True], lab[1:] != lab[:-1])) & graded)
    lengths = []
    for s in starts:
        e = s
        while e + 1 < len(lab) and lab[e + 1] == lab[s]:
            e += 1
        lengths.append((e - s + 1) / FPS)
    lengths = np.array(lengths)
    return dict(poss=len(lengths), mean=lengths.mean() if len(lengths) else 0.0,
                short=np.mean(lengths < 2.0) if len(lengths) else 0.0,
                dead=np.mean(lab[graded] == "D"))


def main(paths):
    tags = load_tags()
    set_pieces = [t for t in tags if lower(t["code"]) == "set pieces"]
    phase_codes = sorted({t["code"] for t in tags if side_of(t["tags"].get("Team"))
                          and lower(t["code"]) not in ("set pieces", "goals")})
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    labs = [pd.read_csv(p)["label"].astype(str).str.strip().str.upper().to_numpy() for p in paths]

    w = max(12, max(len(n) for n in names) + 2)
    head = "{:<34}".format("") + "".join("{:>{w}}".format(n, w=w) for n in names)
    print(head)
    print("-" * len(head))

    st = [structure(l) for l in labs]
    for key, fmt, title in (("poss", "{:d}", "possessions (full match)"),
                            ("mean", "{:.1f} s", "mean possession"),
                            ("short", "{:.0%}", "possessions shorter than 2 s"),
                            ("dead", "{:.0%}", "dead ball (graded frames)")):
        print("{:<34}".format(title) + "".join(
            "{:>{w}}".format(fmt.format(s[key]), w=w) for s in st))

    row = []
    for l in labs:
        vals = []
        for t in set_pieces:
            seg = l[max(0, t["start_frame"] - 75):max(0, t["start_frame"])]
            seg = seg[seg != "X"]
            if len(seg):
                vals.append(np.mean(seg == "D"))
        row.append(np.mean(vals) if vals else np.nan)
    print("{:<34}".format("D in 3 s before a set piece") + "".join(
        "{:>{w}}".format("{:.0%}".format(v), w=w) for v in row))

    print("\nphase agreement: share of alive frames given to the TAGGED team")
    for code in phase_codes:
        inst = [t for t in tags if t["code"] == code and side_of(t["tags"].get("Team"))]
        cells = []
        for l in labs:
            hit = tot = 0
            for t in inst:
                seg = l[max(0, t["start_frame"]):max(0, t["end_frame"])]
                alive = np.isin(seg, ["H", "A"])
                hit += int(np.sum(seg[alive] == side_of(t["tags"].get("Team"))))
                tot += int(alive.sum())
            cells.append("{:.0%}".format(hit / tot) if tot else "n/a")
        print("{:<34}".format("  {} ({})".format(code.lower()[:24], len(inst))) +
              "".join("{:>{w}}".format(c, w=w) for c in cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
