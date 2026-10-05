# Playing games with the late-interaction decision model (design note, 2026-10-05)

Owner, 2026-10-05: "Jev showed that they could play DOOM using their model as well as pokemon. I don't know how they encoded it, but it
makes me wonder if we could use our late interaction stuff to quickly do a multiple choice experiment like that." Research by a subagent
(local Jev notes, the Open-Jev clone, web); licences checked via PyPI JSON and the GitHub API.

## 1. What Jev did
- No demo used pixels: code reads the game's internal state into fields, Jev picks from a short action list, code presses buttons.
- DOOM (TypeSafe's launch demo): state as "a data structure with text, not on images (yet...)", ~10 queries/s (~$7/h), 0.114 s per call
  (advertised 70-500 ms); TypeSafe concedes "a non-AI doom bot could play better". No action schema, training details or score published.
- An unofficial playground schema: 15 fields (enemy angle/distance, health %, ammo %, walls left/right/ahead, door ahead, items, last
  action), 10 actions (move, strafe, turn, shoot, open door, use item, idle).
- Community DOOM (lukaske/jev-doom-agent): Chocolate Doom in WebAssembly feeds health, armour, ammo, position, visible monsters and pickups
  into one choice among tactical macros; a controller turns macros into keys (GPL-2.0+ engine, BSD Freedoom content).
- Pokemon (valentynkit/jev-plays-pokemon-red, MIT code, ROM supplied by the user): PyBoy reads memory into a typed snapshot; code decides
  legal actions and handles routine play; Jev is asked only at branch points (goal sentence, ~725 input tokens, a "faints this turn"
  forecast per candidate); median 621 ms (n=6); won the first rival battle; one turn of lookahead at most.
- Open-Jev's ViZDoom basic scenario: text state, three questions per step (strafe left/right/hold; attack yes/no; a 5-level alignment
  score), behaviour cloning on 6,354 rows from a tracking heuristic.
- Unknown: how Jev was trained on games, win rates, whether demos were tuned.

## 2. Mapping onto li_decider
- State -> query text, e.g. `goal: pick up the red key | ahead: wall 2 | left: red key 1 | carrying: nothing | last: turn left, forward`.
- Each legal action -> an option document with its recent outcomes (as category documents carry past filings): `go forward | last 3
  times: moved, moved, blocked`. Softmax over legal options, CE + Brier.
- Fits: large option sets that change every step (TextWorld's allowed commands), cached option encodings, ~0.16 ms per decision, calibrated
  fallback to an expert. Does not fit: visual state (render to text first), tiny fixed action sets (plain MaxSim cannot relate state and
  action: needs row 211's cross-attention or listwise layer), long horizons (a one-step reader: memory goes into the query or options).

## 3. Environments (open-licence rule)
| Environment | Licence | Verdict |
|---|---|---|
| Minigrid / BabyAI | Apache-2.0 (GitHub), MIT (PyPI) | best first target: CPU, scripted expert (BabyAIBot), symbolic grid easy to render as text |
| TextWorld | MIT; generates its own games | best fit for option documents: allowed commands and expert walkthroughs |
| BALROG | MIT | text renderings of BabyAI, TextWorld, Crafter plus an LLM-baseline harness |
| Crafter, Craftax | MIT | longer horizon, text via BALROG |
| ScienceWorld | Apache-2.0 | needs a JVM |
| ViZDoom + Freedoom | ViZDoom code MIT, Freedoom BSD-3, ZDoom engine mixed incl. GPL | fails the strict rule (as does Chocolate Doom, GPL) |
| NLE / NetHack | NetHack GPL | fails |
| Jericho | GPL-2.0; most story files proprietary | fails |
| PyBoy + Pokemon Red | PyBoy LGPL-3.0; the ROM is proprietary (Nintendo) | fails |
| Tuxemon | GPL-3.0 | fails |
| pokemon-showdown / poke-env | code MIT; Pokemon data is Nintendo IP | owner's call |

## 4. Training signals
Behaviour cloning from BabyAIBot or TextWorld walkthroughs (~1M steps in minutes on CPU); DAgger (the bot relabels states the model
visits); reward-weighted CE or outcome labels (did the episode succeed after this choice); a small-LLM teacher (Qwen3.5-2B/4B letter
logits as soft targets).

## 5. A 1-2 day experiment
- Day 1, BabyAI (GoToLocal, PickupLoc, then a held-out level): text renderer for the egocentric 7x7 view + last 3 actions; 6 action
  documents with their last 3 outcomes; clone the bot on ~200k steps; li_decider a0 (MaxSim), then INTERACT=x.
- Day 2, TextWorld cooking games: options = allowed commands, query = room + inventory + goal; one round of DAgger.
- Measure on 500 held-out seeds/games: success rate and episode length vs random-among-legal, the scripted expert (ceiling), and
  Qwen3.5-2B zero-shot on the same text (letter logits); per-decision accuracy vs the expert; ECE of the chosen option.
- Compute: rollouts on CPU in minutes; training ~7 min on the 3090; the Qwen read under 1 h; $0 Modal.
- What it teaches about the categoriser: whether MaxSim reads short structured state or needs cross-attention with tiny fixed options;
  option-set effects (a variable count of allowed commands); whether outcome history in an option document helps (as past filings did);
  calibration under shift (a held-out level is the analogue of an unseen payee).

## 6. Risks
Compounding errors and missing memory (cap episodes, report loops); text rendering does much of the playing (credit the model only above
random and rendering baselines); BabyAI's 6 actions may reduce to a plain classifier (TextWorld is the stronger test); Minigrid's
GitHub/PyPI licences disagree (both pass); Jev's own game results are unpublished.

Sources: references/jev_credibility_and_unknowns.md, references/openjev_zefan_analysis.md; the Open-Jev clone (~/projects/Taytay/Open-Jev:
docs/games.md, docs/doom-case.md, docs/community.md, docs/public-capabilities.md, jev/case_doom.py, jev/doom_control.py);
https://typesafe.ai/blog/introducing-system-one-models-and-jev; https://assets.theregister.com/2026/09/16/2026/;
https://typesafe-ai-playground.vercel.app/simulations/doom; https://github.com/lukaske/jev-doom-agent;
https://github.com/valentynkit/jev-plays-pokemon-red.
