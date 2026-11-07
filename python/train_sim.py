"""Trains in the Rust simulator: the same rounds, rules, inputs and model as train_auto.py, about a hundred times faster.

Hundreds of rounds run side by side. The model picks every round's next input in one go on the graphics card, the
simulator plays them on all cores, and learning works through big stretches of play. The model it saves plays the
real game unchanged: live_auto.py shows it on a real server you can join.

    python train_sim.py --name auto6 --resume ../runs/models/auto5/latest.zip
"""
import argparse
import json
import os
import signal
from pathlib import Path

import torch
from torch.distributions import Distribution

import train_auto as ta
from curriculum import Curriculum
from sim_env import SimVecEnv
from spatial import MemoryPolicy

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="auto6")
    p.add_argument("--envs", type=int, default=512, help="rounds played side by side")
    p.add_argument("--steps-per-env", type=int, default=64, help="ticks of each round per stretch of play before learning")
    p.add_argument("--batch", type=int, default=1024,
                   help="ticks per learning step. Bigger batches fill the card's memory, and then its library takes a far slower way: 2048 learned 3x slower than 1024 with the viewer's game open")
    p.add_argument("--epochs", type=int, default=4, help="passes over each stretch; with cheap play, more play beats more passes")
    p.add_argument("--steps", type=int, default=2_000_000_000)
    p.add_argument("--resume", help="model to continue from (its layout may be older: every weight that fits is kept)")
    p.add_argument("--levels", help="levels.json to start from, when the run has none yet")
    p.add_argument("--recordings", nargs="+", default=[str(ROOT / "runs" / "demos"), str(ROOT / "runs" / "demos_teacher")])
    p.add_argument("--start-from-recordings", type=int, default=0)
    p.add_argument("--sim-threads", type=int, default=6, help="cores for the simulator")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", str(a.sim_threads))
    Distribution.set_default_validate_args(False)
    # Far more decisions per hour than the game gives, so the schedules are counted further out.
    ta.EXAM_EVERY = 2_000_000
    ta.SAVE_EVERY = 500_000  # about every two minutes
    ta.CALM_FROM, ta.CALM_OVER = 20_000_000, 400_000_000
    out = ROOT / "runs" / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "levels.json"
    start_levels = json.loads(saved.read_text()) if saved.exists() else json.loads(Path(a.levels).read_text()) if a.levels else None
    curriculum = Curriculum(start_levels)
    latest = out / "latest.json"
    saved_state = json.loads(latest.read_text()) if latest.exists() else {}
    curriculum.load_state(saved_state.get("curriculum", {}))
    # A run that already has its own model always continues from it: --resume only starts a new run.
    continuing = (out / "latest.zip").exists()
    if continuing and a.resume:
        print(f"{out / 'latest.zip'} exists: continuing this run, not {a.resume}", flush=True)
        a.resume = None
    vec = SimVecEnv(a.envs, curriculum.levels, seed=a.seed)
    kwargs = dict(n_steps=a.steps_per_env, batch_size=a.batch, n_epochs=a.epochs, gamma=ta.GAMMA, gae_lambda=ta.GAE_LAMBDA, learning_rate=ta.LEARNING_RATE,
                  ent_coef=ta.ENTROPY, target_kl=ta.TARGET_KL, device="cuda", policy_kwargs=ta.POLICY, seed=a.seed)
    model = ta.SplitPPO(MemoryPolicy, vec, **kwargs)
    ta.fast_convolutions()
    resumed_steps = 0
    if continuing or a.resume:
        # The eyes and hands may have grown since that model was saved: what was there is kept exactly.
        grew, steps = ta.load_weights(model, out / "latest.zip" if continuing else a.resume)
        print(f"continuing from {out / 'latest.zip' if continuing else a.resume}: grown {grew}", flush=True)
        if not continuing:
            # A model that has learned for a while goes on at its calmer learning rate and exploration: from step 0
            # the schedule's first ones undid much of what it could (bridging fell from 0.82 to 0.66 in 500k).
            resumed_steps = model.num_timesteps = steps
        torch.cuda.empty_cache()
    else:
        ta.init_action_bias(model.policy)
        if a.start_from_recordings:
            ta.start_from_recordings(model, a.start_from_recordings, a.recordings)
            model.save(out / "copied_recordings")
    model.play_on_card = True
    if continuing:
        # The steps saved with these weights; older runs only have status.json, which can be a little ahead of them.
        resumed_steps = saved_state.get("steps") or (json.loads((out / "status.json").read_text()).get("steps", 0) if (out / "status.json").exists() else 0)
        model.num_timesteps = resumed_steps
        print(f"continuing this run from {resumed_steps} decisions", flush=True)
    progress = ta.Progress(vec, out, curriculum, 2)
    if continuing:
        progress.resume_last_exam = saved_state.get("last_exam")

    def stop(*_):
        raise KeyboardInterrupt  # a stop (start.sh, a redeploy) ends learning where it is, and the run is saved below

    signal.signal(signal.SIGTERM, stop)
    try:
        model.learn(a.steps, callback=progress, reset_num_timesteps=resumed_steps == 0)
    except KeyboardInterrupt:
        progress.save_latest()  # nothing since the last save is lost
        print(f"stopped at {model.num_timesteps} decisions: saved", flush=True)


if __name__ == "__main__":
    main()
