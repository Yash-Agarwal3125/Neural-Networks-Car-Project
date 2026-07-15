# Manual — Setup, Usage, and Internals

This is the detailed reference. `README.md` is the pitch; this file is the
"how do I actually run/extend this" document, written for a collaborator
opening the repo for the first time.

## 1. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

That's it — one virtual environment. (You may notice `.venv-tf/` in the
folder; it was an old separate environment from debugging a TensorFlow
install issue and is not needed — don't recreate it, and it's excluded from
git.)

Python 3.10+ is required (TensorFlow/Keras + pygame combo was built/tested
on 3.10).

Notebooks open in Jupyter/VS Code as usual. Pygame opens a real window, so
run notebooks locally (not in a headless/remote environment) when you want
to *see* the car — set `os.environ["SDL_VIDEODRIVER"]` to `"windows"` (or
just leave it unset) for that. Training notebooks set it to `"dummy"` for
headless, faster training — leave that alone unless you specifically want to
watch training live (much slower).

## 2. Repo map — what every file is for

| File | Status | Purpose |
|---|---|---|
| `Core_Game_Parts.py` | **Core, shared** | `Car` class, ray-casting sensors, all track/physics constants, checkpoint data. Imported by the manual-play notebook. |
| `User_Mode.ipynb` | **Canonical** | Drive the car yourself with arrow keys. Also your track-testing tool. |
| `Notebook.ipynb` | Scratch / dev notebook | Contains the **track drawing tool** (`draw_track()`) and an early plain-DQN prototype. Kept because the drawing tool lives here — see §4. Everything else in it is superseded. |
| `Advanced_D3QN_Trainer.ipynb` | **Canonical trainer** | Trains the current agent: Dueling Double DQN (D3QN) with Prioritized Experience Replay. |
| `old time code.py` | **Most advanced version (despite the name)** | A `.py` script, newer than the notebook above, adding lap timing, a 5-phase training curriculum, and top-3 lap-time checkpoint saving. This is the reference implementation — treat it as "D3QN v2". Should eventually be merged back into `Advanced_D3QN_Trainer.ipynb`. |
| `Run_D3QN_model.ipynb` | **Canonical playback** | Load a D3QN weights file and watch it drive. |
| `Run_Model.ipynb` | Legacy playback | Only works with old plain-DQN weights (`finetuned_weights_episode_*.h5`), not D3QN checkpoints. |
| `Train_Agent.ipynb` | Legacy trainer | Plain DQN, superseded by the D3QN trainer. Kept for reference/comparison. |
| `Tuning_Weignts.ipynb` | Legacy | Hyperparameter/fine-tuning experiments for the *old* plain-DQN/Double-DQN architecture, not D3QN. |
| `User_Training_Weights.ipynb` | Legacy, currently broken | Imitation-learning (behavioral cloning) pipeline. See §6 — not compatible with the current D3QN agent as-is. |
| `Track_images/` | Assets | `car.png`, `track1.png` (the only track actually wired up and used), plus `silverstone.png`, `spa.png`, `monza_draw.png` — drawn/collected but **not currently referenced by any code**. Aspirational multi-track assets. |
| `Weights/` + loose `*.weights.h5` in the project root | Trained checkpoints | Two separate batches of saved weights from different training runs (see §5 and §8 for which is which). |
| `reward_components_log.csv` | Training log | Per-step reward log written by `old time code.py`'s trainer. Only `total_reward` and `checkpoints` columns currently contain real data (see §8, item 1). |
| `requirements.txt` | Dependencies | `pygame`, `numpy`, `tensorflow`, `keras`, `matplotlib`. |

## 3. Quick start — watch the current best agent drive

1. Open `Run_D3QN_model.ipynb`.
2. In the weights-loading cell, set:
   ```python
   WEIGHTS_FILENAME = "Weights/best_d3qn_19checkpoints.weights.h5"  # or any file in Weights/
   ```
   (The notebook's default, `best_lap_1.weights.h5`, doesn't exist — you must
   pick a real file from `Weights/` or the project root. Files are named by
   how many of the 12 checkpoints the agent cleared before crashing/timing
   out in that run — bigger number is generally better, but compare a few.)
3. Run all cells. A pygame window opens and the car drives itself on
   `Track_images/track1.png`.

## 4. Creating a new track / map

There's no visual map editor UI — tracks are hand-drawn pixel art plus a
hand-entered checkpoint list. Workflow:

1. **Draw the track.** In `Notebook.ipynb`, run `draw_track()`. A blank
   pygame canvas opens:
   - Hold **left mouse button** and drag to paint track walls (grey).
   - Hold **right mouse button** and drag to erase.
   - Press **S** to save — this writes to `monza_draw.png` in the project
     root (the constant is `TRACK_SAVE_PATH` in `Core_Game_Parts.py`).
2. **Move it into place.** Move/rename the saved PNG into `Track_images/`,
   e.g. `Track_images/my_track.png`.
3. **Point the code at it.** Edit `TRACK_IMAGE_PATH` in
   `Core_Game_Parts.py`:
   ```python
   TRACK_IMAGE_PATH = r"Track_images\my_track.png"
   ```
4. **Set the spawn point.** Edit `DEFAULT_START_X`, `DEFAULT_START_Y`,
   `DEFAULT_START_ANGLE` to a clear spot on your new track.
5. **Find checkpoint coordinates.** Run `User_Mode.ipynb` (manual drive) and
   add a temporary print of the mouse position on `MOUSEMOTION` events, or
   watch `car.x, car.y` while you drive along your intended racing line —
   there's no built-in coordinate picker, so this is manual trial and error.
6. **Enter checkpoints by hand.** Update `checkpoint_data` in
   `Core_Game_Parts.py` — a list of `(x, y, width, height, angle)` rectangles
   the car must pass through, in order, plus `finish_line_rect` for the
   start/finish gate.
7. If you're training D3QN, also check `START_LINE_RECT` inside
   `Run_D3QN_model.ipynb` / `old time code.py` — it's a **second, separate**
   finish-line box that isn't derived from `finish_line_rect` and must be
   updated to match your new track too (see known issue in §8).

This whole process is the most tedious part of the project and is a good
first thing to automate — see the improvement suggestions in §9.

## 5. Training an agent

### The current approach: D3QN (recommended)

Run `Advanced_D3QN_Trainer.ipynb` (or, for the more advanced version, `old
time code.py` directly with `python "old time code.py"`).

- **State (5 values):** left/front/right ray distances, current speed,
  curvature (`|left - right| / (left + right)`, a proxy for "how sharp is
  the turn ahead").
- **Actions (4):** turn left, accelerate, turn right, brake.
- **Network:** Dueling architecture — shared `Dense(128)→Dense(128)` trunk,
  split into a value stream (`Dense(64)→Dense(1)`) and an advantage stream
  (`Dense(64)→Dense(action_size)`), recombined as `Q = V + (A - mean(A))`.
- **Algorithm:** Double DQN target computation + Prioritized Experience
  Replay (`alpha=0.5`, `beta=0.4`) + soft (Polyak, `tau=0.005`) target
  updates.
- **Key hyperparameters:** `gamma=0.98`, `batch_size=128`, replay capacity
  `20000`, `epsilon` decays `1.0 → 0.1` at `0.995`/step (`old time code.py`
  fine-tune runs use a slower `0.9995` decay and a lower learning rate
  `1e-4` when resuming).
- **Curriculum (`old time code.py` only):** training starts by only
  rewarding progress toward the first few checkpoints, unlocking more of the
  track as the agent proves it can clear them, then switches into a
  lap-time-optimization phase once lap times stabilize.

Checkpoints save periodically to `d3qn.weights.h5` and, in `old time
code.py`, to a running top-3 "best lap time" leaderboard
(`best_1/2/3.weights.h5`) plus `old_{episode}.weights.h5` snapshots every 200
episodes.

### Resuming / fine-tuning

Load an existing `.weights.h5` file into the same network architecture,
lower the learning rate and epsilon, and continue calling the training loop.
In `Advanced_D3QN_Trainer.ipynb` this is currently a markdown cell near the
bottom (turn it back into a code cell to use it) — `old time code.py` has a
working version at the bottom of the file that loads
`best_d3qn_16checkpoints.weights.h5` at low epsilon (`0.03`) as a template.

### The old approach: plain DQN (legacy, kept for comparison)

`Train_Agent.ipynb` / `Notebook.ipynb` — a 4-input (3 sensors + speed),
3-action (left/right/brake — no explicit accelerate; the car auto-throttles)
single-stream DQN with a hard-copied target network. This is what the
original README described. It's simpler and a reasonable baseline to cite
in a paper alongside D3QN, but it's not being actively developed.

## 6. Imitation learning / pretraining pipeline (currently broken)

`User_Training_Weights.ipynb` lets you drive manually to record
`(state, action)` pairs (`collect_expert_data()`), then trains a small
classifier on them (`pretrain_agent()`). Two things to know before using it:

1. The recorded data file (`pretrain_data.npy`) doesn't exist yet —
   `collect_expert_data()` is commented out in the notebook's last cell.
   Uncomment and run it first, or the pretraining cells will crash trying to
   load a missing file.
2. **This pipeline targets the old 4-input/3-action architecture, not
   current D3QN (5 inputs, 4 actions).** Weights produced here cannot be
   loaded straight into the D3QN network. If you want imitation learning for
   D3QN, the state vector and action set need to be updated to match (see
   §9 — this is one of the "must-have" items if you want to use imitation
   learning going forward).

## 7. What each weights file is

Weight files are checkpoints from different training runs and are **not all
compatible with the same network shape** — legacy DQN weights (4→3) will
error or silently mismatch if loaded into the D3QN network (5→4), and
vice versa.

- `Weights/best_d3qn_*checkpoints.weights.h5`, `Weights/d3qn*.h5` — D3QN
  checkpoints, named by how many of the 12 checkpoints were cleared.
- `Weights/root_run_d3qn/` — a **separate** D3QN training run that was
  originally saved loose in the project root. Despite matching filenames
  with files directly in `Weights/` (e.g. `best_d3qn_15checkpoints.weights.h5`
  exists in both places), these are **not the same weights** — different
  byte content, verified by hash — kept in their own subfolder specifically
  so they don't collide/overwrite the other run.
- `Weights/legacy_dqn/` — legacy plain-DQN checkpoints
  (`finetuned_weights_episode_350/550.weights.h5`, `old_200.weights.h5`),
  compatible only with `Run_Model.ipynb`, not `Run_D3QN_model.ipynb`.

If you only remember one file to try: `Weights/best_d3qn_19checkpoints.weights.h5`
or `Weights/best_d3qn_36checkpoints.weights.h5` are the highest checkpoint
counts currently saved.

## 8. Known issues (found during this audit)

1. **Reward-component logging is dead.** `reward_components_log.csv` has
   `r_center`, `r_speed`, `r_progress`, `r_step` columns that are always 0 —
   the environment's `step()` never populates the `info['reward_breakdown']`
   dict the logger reads from. Only `total_reward` and `checkpoints` are
   real. Fix before using this log for any paper plots.
2. `Run_D3QN_model.ipynb` defaults to a weights filename
   (`best_lap_1.weights.h5`) that doesn't exist on disk — must be changed
   before running (see §3).
3. `Run_D3QN_model.ipynb` defines `build_dueling_dqn` twice in one cell
   (once for the old 4/3 shape, once for 5/4) — the second silently wins.
   Harmless but confusing; delete the first definition.
4. `silverstone.png`, `spa.png`, `monza_draw.png` in `Track_images/` are not
   referenced by any code — the project is effectively single-track
   (`track1.png`) despite the assets suggesting otherwise.
5. Two independent finish-line definitions that must be kept in sync by
   hand: `finish_line_rect` (`Core_Game_Parts.py`) and `START_LINE_RECT`
   (hardcoded inside `Run_D3QN_model.ipynb`).
6. `old time code.py` and `Advanced_D3QN_Trainer.ipynb` use the same network
   shape but different reward constants and physics limits (e.g. `max_steps`
   1000 vs 2000, checkpoint bonus 20 vs 50) — weights trained under one are
   not guaranteed to behave the same evaluated under the other.
7. Root-level `track.png` is an old/unused track image (different from
   `Track_images/track1.png`) — safe to delete once you confirm nothing
   references it locally.

## 9. Suggested improvements

**Must-have (for research-paper credibility / a stable shared repo):**

- Fix the reward-breakdown logging bug (#1 above) — you need real
  component-wise reward curves to explain *why* the agent behaves as it
  does in a paper.
- Add a headless evaluation script: run N episodes with `epsilon=0`, report
  checkpoint-clearance rate, mean/best lap time, crash rate. Right now
  "how good is this checkpoint" is judged by eyeballing a pygame window —
  a paper needs numbers.
- Fix random seeding (`random`, `numpy`, `tf`) so training runs are
  reproducible enough to report.
- Pick one canonical D3QN implementation — merge `old time code.py`'s
  curriculum/lap-timing improvements into `Advanced_D3QN_Trainer.ipynb` (or
  vice versa) so there's a single source of truth instead of two scripts
  with diverging reward constants.
- Unify the two finish-line definitions (#5) into one.

**Nice-to-have (future work):**

- A coordinate-picker helper (click on the track image to print
  x/y) to make checkpoint placement on new tracks far less tedious than
  the current trial-and-error.
- Actually wire up `silverstone.png`/`spa.png` as second/third tracks (or
  delete them) — real multi-track generalization would strengthen a paper's
  claims about the agent, not just a single memorized circuit.
- Frame-stacking or a small recurrent layer, since the agent currently has
  no memory between steps (pure reactive control from a single frame's
  sensor readings).
- More/finer ray-casting sensors (5–7 rays instead of 3) for sharper
  cornering awareness.
- A config file (single `track_config.py`/JSON per track: image path, start
  pose, checkpoints, finish line) instead of hand-editing constants in
  `Core_Game_Parts.py` for every track switch.
- Bring the imitation-learning pipeline (§6) up to date with the D3QN
  state/action shapes if you want to use it again.
