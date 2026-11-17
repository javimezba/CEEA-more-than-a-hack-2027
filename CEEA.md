# CEEA — Technical Document

More than a Hack 2027 · Phase 1: Ball Possession · Team CEEA · November 2026

## Summary

Our system labels every frame of FC Barcelona vs Athletic Club as `H`, `A`, `D` or `X` from the automatic tactical tracking alone, with no training data and no hand labels. It turns the tracking into physical evidence (who is next to a slow ball, whether the ball is out, whether the players are walking, which team is spread out) and decodes the most consistent sequence of states with a three-state Viterbi decoder. The submitted file is produced by `solution/possession_v2.py`.

| Output of the final model (v2) | Value |
| --- | --- |
| Possessions over the full match | 265 |
| Mean possession length | 14.9 s |
| Dead ball, share of graded frames | 38% |
| Phase types of the automatic Smart Tagging where v2 agrees better than our first model | 10 of 12 |

The public five minutes were used only as a sanity check of the first model: it found 17 possessions where the ground truth has 17 (spellF1 0.41, macroF1 0.75, against 0.33 and 0.62 for the reference baseline). Versions were compared with a ground-truth-free check against the automatic Smart Tagging, never on the public window.

## Approach

The model is a hand-built evidence model decoded with Viterbi: every frame gets two scores from the tracking, and the decoder picks the state sequence that agrees best with them while paying a fixed cost for each change of state.

1. **Measurements.** From the tactical tracking (25 fps) we compute, per frame and in metres: the distance from the ball to the nearest home and away player, the ball speed, whether the ball is beyond a line, the players' median speed, and each team's spread (mean distance of its players to the team centroid).
2. **Team evidence** (positive = Barcelona, negative = Athletic). It is non-zero only when a player is within 2.5 m of a ball moving slower than 12 m/s, and it grows with the distance advantage over the nearest opponent, capped at 3 per frame. On frames with no such evidence, the more spread-out team gets a weak push of at most 0.2 per frame, because the team in possession opens up.
3. **Dead-ball evidence** (positive = out of play). A sum of cues: ball beyond a line, players walking, ball still for 1 s, ball unseen for 3 s, minus a push when the ball moves faster than 10 m/s.
4. **Decoding.** A three-state Viterbi (H, A, D) per half. Switching team costs 18 and entering or leaving D costs 40, so a change needs about 0.5 s of clean control by the other team, or about 1 s of clear stoppage.

```
Tactical tracking ──> Measurements ──┬──> Team evidence ──────┬──> Viterbi decoder ──> H / A / D per frame
(players, ball,       (distances,     │   (player near slow    │   (switch cost 18 / 40)
 25 fps, team of      speeds, ball    │    ball + team spread) │
 each slot)           out, spread)    └──> Dead-ball evidence ─┘
                                          (ball out, walking,
                                           ball still or unseen)
```

Every frame gets a team score and a dead-ball score; the decoder trades them against the cost of changing state over a whole half.

We chose this over a learned model for three reasons. The rules forbid hand labels on this game and using the public five minutes as training data, so a supervised model would need labelled data from other games, which we did not have. The ranking metric rewards whole possessions, and a decoder with switch costs removes the frame-to-frame flicker that a per-frame classifier or a majority vote produces, without blurring the transitions. And every parameter can be justified from football or physics and audited in the code.

## Assets used and dropped

The submission is computed from the tactical tracking and its metadata only; the Smart Tagging is used to compare versions, not to produce labels.

| Asset | Used | How, or why not |
| --- | --- | --- |
| Tactical ATD (players and ball) | Yes | Every measurement in the model |
| ATD metadata XML | Yes | Team of each track slot; the first team listed is home |
| Smart Tagging | For checking only | Ground-truth-free comparison of versions (`solution/check.py`). An exploratory version used set-piece tags as priors; with the tag windows placed correctly its output barely changed (271 vs 270 possessions), so it was dropped from the model |
| Public 5-minute ground truth | Sanity check only | Alignment and file format of the first model; never used to choose parameters or versions |
| Tactical homography | No | The tracking is already in pitch coordinates |
| Tactical video | No | Useful for ball detection and ball height; out of reach in the time available |
| Broadcast video, ATD and homography | No | Not synchronised with the tactical timeline; mapping it was out of scope |

## Training data

None. No labelled dataset of any game was used, and no person labelled or corrected anything on the provided game.

- **Parameters** were set before running, from pitch markings, the Laws of the Game or physics, and each is explained next to its value in the code.
- **Choosing between versions** used `solution/check.py`, which compares submissions over the whole match against the Smart Tagging. The tagging is generated automatically by Metrica, so no human label enters the comparison. For each candidate, the decision rule was written down before its results were seen.
- **Synthetic matches.** To debug the code we generated simulated matches with known states (players moving continuously, passes, interceptions, balls going out). They come from no real game and were used to check that each cue does what it should, not to fit any parameter.
- **Self-supervised experiment.** v3 estimated a time lag by aligning two of the model's own signals; no label was involved. It was dropped (see Experiments).

## Possession rules implemented

Possession changes only when the other team shows sustained control; everything else keeps the last owner.

| Situation | What the model does |
| --- | --- |
| Controlled ball | A player within 2.5 m (full weight within 1 m) of a ball slower than 12 m/s (full weight below 6 m/s, dribbling pace) gives evidence to his team, scaled by his distance advantage over the nearest opponent |
| Loose ball or ball in flight | No player close enough, or ball too fast: no evidence, so possession stays with the last team that had it |
| Simple touch, deflection, aerial duel | Too brief to pay the switch cost of 18: about 0.5 s of clean control is needed with full evidence, longer with partial evidence |
| Challenge (both players close) | Distance advantage near zero: almost no evidence either way |
| Ball not seen | The more spread-out team gets at most 0.2 per frame; 3 s of this alone (15) cannot flip possession (18) |
| Stoppage | Dead-ball evidence: ball beyond a line by 0.5 m (kept for up to 10 s if the tracker then loses it), players' median speed under 1.4 m/s, ball still for 1 s, ball unseen for 3 s. Entering or leaving D costs 40, about 1 s of clear stoppage |
| Restart | No rule assigns the restart to a team: the first team to control the ball after D gets it |
| Outside the halves | `X` before kick-off, at half-time and after the final whistle (never graded) |
| Before the second-half kick-off | Forced to D: frames 110,808 to 113,741, as published in the challenge FAQ |

## Experiments

Of five improvements tried on top of the first model, only the team-shape cue changed the output measurably, and it is the one kept. Each version was compared with `solution/check.py` over the whole match: possession count and length, share of D in the 3 s before each Smart Tagging set piece, and, for each team-tagged phase type, the share of our in-play frames that give the ball to the tagged team (high for attacking phases, low for defending ones).

| Version | Change | What we measured | Decision |
| --- | --- | --- | --- |
| v1 | Core: control evidence, dead-ball cues, Viterbi | 241 possessions, 16.3 s mean, 38% D; D before 79% of set pieces | Base |
| v2 | + team shape; + ball interpolation over gaps up to 1 s, receptions, ball lost while leaving the pitch, collective braking | 265 possessions, 14.9 s. Better agreement in 10 of 12 phase types, for example creating chances 57% to 77% and defending in own third 43% to 23%. Worse in both transition types (attacking 73% to 66%, defensive 27% to 34%); possessions under 2 s 4% to 10%. Ablations: without shape the output is almost identical to v1; without receptions, almost identical to v2 | **Kept (final)** |
| v3 | Shape read τ seconds ahead, τ estimated by aligning shape with control evidence | τ = 9.2 s, not physically credible. Transitions back to 72% and 28%, but half of v2's gain in settled phases lost | Dropped |
| v4 | Shape switched off from 1 s before to 5 s after each change of possession (two passes) | Transitions 68% and 32%; settled phases 3 to 6 points worse than v2; possessions under 2 s 6% | Dropped |
| v5 | Restart detection (ball still at a restart spot, players walking) + median-filtered ball speed | D before set pieces 80%; every other figure within 1 point of v2 | Dropped |
| v6 | Throw-in and goal-kick detection from the Laws (ball still on a touchline; ball still in the goal area with no opponent in the penalty area) | Found 6 throw-ins and 1 goal kick; output identical to v2 | Dropped |

The stoppages v2 misses are mostly goal kicks (6 of 10) and throw-ins (7 of 26). In the 3 s before those tags the ball is visible 76% to 83% of the time, moves at a median 3.5 to 4.5 m/s and is never still, and the players move at about 1.9 m/s. A placed, still ball does not appear in the automatic tracking, which is why v5 and v6 found almost nothing.

On simulated matches v2 beat v1 clearly (spellF1 0.96 vs 0.88), and v3, v4 and v6 did not beat v2. These simulations are our own and only show that the code behaves as intended.

## Limitations and future work

The model's main weakness is the timing of changes of possession, and its second is the stoppages it cannot see.

| Limitation | Evidence | What we would try with more time |
| --- | --- | --- |
| Late switches after a turnover | Team shape lags possession by a few seconds: transition phases agree less with the tagging in v2 than in v1 | A cue on the *change* of spread (a team opening up quickly is gaining the ball) instead of its level |
| Short spurious possessions | 10% of v2's possessions last under 2 s | A minimum-duration constraint in the decoder (a semi-Markov model) |
| Missed stoppages | About 21% of tagged set pieces have no D in the 3 s before; mostly goal kicks and throw-ins | Detect the ball, and whether it is carried in the hands, from the video |
| Ball unseen on 41% of graded frames | Gaps are long: only 185 frames fall in gaps of up to 1 s, and just 213 receptions are detected in the match | Ball detection from the video, or synchronising the broadcast feed to fill gaps |
| Weak validation | No full-match ground truth; the Smart Tagging is automatic and its phases are not possessions | Hand-labelled possessions from other games, which the rules allow |
| One match-specific constant | The second-half kick-off frame comes from the challenge FAQ | Detect kick-offs from the formation (every player in his own half, ball on the centre spot) |

## Reproduction

One command regenerates the submitted file byte for byte from the data bundle. Repository: [github.com/javimezba/CEEA-more-than-a-hack-2027](https://github.com/javimezba/CEEA-more-than-a-hack-2027).

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r solution/requirements.txt   # numpy 2.5.4, pandas 3.0.6

# folder that contains tactical/ and broadcast/ from the data bundle
# Windows: $env:MTAH_DATA_DIR = "C:\mtah_data"     Linux/macOS: export MTAH_DATA_DIR=/path/to/data

python solution/possession_v2.py --out evaluation/results/CEEA.csv
python evaluation/validate_submission.py evaluation/results/CEEA.csv
```

- Tested with Python 3.12 on Windows. Only numpy and pandas are needed.
- The first run parses the tracking and caches it as `.npz` inside the data folder, never inside the repository; later runs take seconds.
- The pipeline has no randomness: the same data always gives the same file. We checked that the command above reproduces the submitted file exactly.
- The match data is not in the repository.

## AI assistance

An AI assistant (Claude, by Anthropic) was used for strategy and code. The team ran every script locally and decided which version to keep.

The assistant was not given the restricted data: no tracking, video, Smart Tagging or homography, nor any extract of them. It only received public documentation, our code, and aggregated outputs of the scorer and of our own scripts (counts, percentages, medians). The public five-minute ground truth was used only to check that the pipeline works, never to choose parameters or versions.
