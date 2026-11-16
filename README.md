# CEEA — More than a Hack 2027, Phase 1: Ball Possession

Team **CEEA**'s submission to the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
initial challenge: label every frame of FC Barcelona vs Athletic Club with `H` / `A` / `D` / `X`.

The model uses **no training data and no hand labels**. It turns the automatic tracking into
physical evidence (who is next to a slow ball, is the ball out, are the players walking, which team
is spread out) and decodes the most consistent sequence of states with the Viterbi algorithm.
Details, experiments and limitations are in the technical document.

## Reproduce the submission

Requirements: Python 3.12 (3.9+ should work), the match data bundle (not included — it is
licensed for the competition only).

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r solution/requirements.txt

# point to the unpacked data bundle (folder containing tactical/ and broadcast/)
# Windows: $env:MTAH_DATA_DIR = "C:\mtah_data"     Linux/macOS: export MTAH_DATA_DIR=/path/to/data

python solution/possession_v2.py --out evaluation/results/CEEA.csv
python evaluation/validate_submission.py evaluation/results/CEEA.csv
```

The first run parses the tracking and caches it as a `.npz` inside the data folder (never inside
the repository). The output is deterministic: the same data always produces the same file.

## What is in `solution/`

| file | role |
|---|---|
| `possession.py` | core (v1): loading, physical measurements, control and dead-ball evidence, Viterbi decoder |
| `possession_v2.py` | **final model**: v1 + team-shape cue + ball interpolation, receptions, ball-exit and braking cues |
| `check.py` | ground-truth-free consistency check of any submissions against the automatic Smart Tagging |
| `requirements.txt` | pinned dependencies |

## The method in one paragraph

Each frame gets two scores. **Team evidence** is non-zero only when a player is within 2.5 m of a
ball moving slower than 12 m/s, so a ball in flight or a deflection never changes possession; on
frames with no such evidence, the team that is more spread out on the pitch gets a weak push (the
team in possession opens up). **Dead-ball evidence** adds up the ball being outside the lines,
the players walking, the ball standing still, and a long absence of the ball. A three-state
Viterbi decoder (H / A / D) charges a fixed cost for every change of state, which removes the
frame-to-frame flicker that destroys `spellF1` without blurring the transitions. Every parameter
is set from football or physics reasoning and documented next to it in the code.

## Compliance with the rules

- **No hand labels and no training** on the provided game. Nothing is fitted to any label.
- The **public 5 minutes** were used only to check that files were aligned and well formed, never
  to choose parameters or versions. Versions were compared with `solution/check.py`, which uses
  only the automatic Smart Tagging.
- **Assets used:** tactical tracking (ATD) and its metadata, and the Smart Tagging (only for the
  consistency check). Homography, videos and the broadcast feed were not used.
- **AI assistance:** the code was developed with the help of Claude (Anthropic), as declared in
  the technical document. The match data was never shared with any hosted model.
- The match data is **not** in this repository (see `.gitignore`).

## Challenge documentation

The organisers' original documents are kept unchanged: [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md),
[RULES.md](RULES.md), [FAQ.md](FAQ.md). The evaluation code in `evaluation/` is theirs (MIT).
