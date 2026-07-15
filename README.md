# AI Self-Driving Car — D3QN

A Pygame simulation where a car learns to drive a racetrack using
reinforcement learning (TensorFlow/Keras). The car senses its surroundings
with three ray-cast distance sensors and picks one of four actions each
step: turn left, turn right, accelerate, or brake.

The current agent is a **Dueling Double DQN (D3QN)** with prioritized
experience replay and a training curriculum that unlocks more of the track
as the agent improves. An earlier, simpler plain-DQN version is kept in the
repo for comparison.

**Full setup, usage, track-creation, and architecture details:
see [`MANUAL.md`](MANUAL.md).** This README is just the pitch.

## Status

The agent reliably clears multiple checkpoints per run — weight filenames
encode how many checkpoints were cleared before the run ended (higher is
better; the track has 12 checkpoints per lap, so counts above 12 mean
multiple laps). It is not yet reliably completing full clean laps at speed.
Training is ongoing; see `MANUAL.md` §9 for what's next.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Then open `Run_D3QN_model.ipynb`, point `WEIGHTS_FILENAME` at a file in
`Weights/`, and run all cells to watch the trained agent drive.

To drive manually instead, open `User_Mode.ipynb`.

Details on training your own agent, creating new tracks, and everything
else live in [`MANUAL.md`](MANUAL.md).

## Repo layout (short version)

| Path | What it is |
|---|---|
| `Core_Game_Parts.py` | Shared physics, sensors, track/checkpoint constants |
| `Advanced_D3QN_Trainer.ipynb`, `old time code.py` | Train the current D3QN agent |
| `Run_D3QN_model.ipynb` | Watch a trained agent drive |
| `User_Mode.ipynb` | Drive manually |
| `Notebook.ipynb` | Track-drawing tool + early prototype |
| `Weights/` | Saved model checkpoints |
| `Track_images/` | Track and car sprites |

Full map of every file, what's canonical vs. legacy, and why: `MANUAL.md`.

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/thing`)
3. Commit your changes
4. Open a pull request
