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
from opponents import Opponents
from pool import Pool, SelfPlay
from rating import Rating
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
    # Network options (docs/training-design.md#natet), kept in the run's net.json so a continued run builds the same.
    p.add_argument("--critic-memory", action="store_true", help="the value gets its own memory (LSTM)")
    p.add_argument("--normalize-inputs", action="store_true", help="the body part's inputs standardized, measured once")
    p.add_argument("--entities", action="store_true", help="opponents and hotbar slots through their own shared networks")
    p.add_argument("--hl-gauss", action="store_true", help="the value as a distribution over bins (HL-Gauss)")
    p.add_argument("--exploit", help="train only to beat this frozen model in duels, then measure how often it does (exploit.json)")
    p.add_argument("--no-evals", action="store_true", help="no exams, ratings or benchmarks during the run (experiments: benchmark after)")
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
    net_file = out / "net.json"
    net = json.loads(net_file.read_text()) if net_file.exists() else dict(critic_memory=a.critic_memory, normalize_inputs=a.normalize_inputs, entities=a.entities,
                                                                           hl_gauss=a.hl_gauss)
    net.setdefault("hl_gauss", False)
    net_file.write_text(json.dumps(net))
    print(f"network: {net}", flush=True)
    kwargs = dict(n_steps=a.steps_per_env, batch_size=a.batch, n_epochs=a.epochs, gamma=ta.GAMMA, gae_lambda=ta.GAE_LAMBDA, learning_rate=ta.LEARNING_RATE,
                  ent_coef=ta.ENTROPY, target_kl=ta.TARGET_KL, device="cuda", policy_kwargs=ta.policy_kwargs(net["critic_memory"], net["entities"], hl_gauss=net["hl_gauss"]), seed=a.seed)
    model = ta.SplitPPO(MemoryPolicy, vec, **kwargs)
    ta.fast_convolutions()
    resumed_steps = 0
    warmup = saved_state.get("value_warmup", 0)
    if continuing or a.resume:
        # The eyes and hands may have grown since that model was saved: what was there is kept exactly.
        grew, steps = ta.load_weights(model, out / "latest.zip" if continuing else a.resume)
        print(f"continuing from {out / 'latest.zip' if continuing else a.resume}: grown {grew}", flush=True)
        if not continuing:
            # A model that has learned for a while goes on at its calmer learning rate and exploration: from step 0
            # the schedule's first ones undid much of what it could (bridging fell from 0.82 to 0.66 in 500k).
            resumed_steps = model.num_timesteps = steps
            if net["critic_memory"] and not any(k.startswith("lstm_critic") for k in ta.saved_keys(a.resume)):
                # The value's new memory starts as a copy of the policy's, and only the value learns for 1M decisions.
                model.policy.lstm_critic.load_state_dict(model.policy.lstm_actor.state_dict())
                warmup = steps + 1_000_000
                print(f"critic memory: a copy of the policy's; only the value learns until {warmup}", flush=True)
            if net["hl_gauss"] and not any(k.startswith("value_net.linear") for k in ta.saved_keys(a.resume)):
                warmup = steps + 1_000_000  # the new value head learns alone first
                print(f"HL-Gauss value: a new head; only the value learns until {warmup}", flush=True)
            if net["normalize_inputs"] and bool((model.policy.features_extractor.body_std == 1).all()):
                mean, std = ta.measure_inputs(model, vec)
                model.policy.features_extractor.fold_normalization(mean.to(model.device), std.to(model.device))
                print(f"inputs standardized from {200 * a.envs} ticks; the model answers as before", flush=True)
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
    progress.value_warmup = warmup
    progress.rating = None if a.no_evals else Rating(out)  # Elo against frozen references after each exam (docs/training-design.md#referenser)
    progress.bench_every = None if a.no_evals else 10_000_000  # the fixed benchmark (benchmark.py): a few minutes, about every hour
    if a.no_evals:
        ta.EXAM_EVERY = 10 ** 12
    progress.rate_every = 4_000_000  # a rating plays 3 kinds of fight x 64 rounds per reference: every other exam
    if continuing:
        progress.resume_last_exam = saved_state.get("last_exam")
        progress.last_bench = saved_state.get("last_bench")
        progress.last_rated = saved_state.get("last_rated")
        progress.bonus_weight = saved_state.get("bonus_weight", 1.0)  # it only falls, across restarts too
        progress.snapshot_scores = saved_state.get("snapshot_scores", [])
    if a.exploit:
        # An exploiter (docs/training-design.md#mata, FightLadder arXiv 2406.02081): a copy trained only to beat the
        # frozen model, in the rating's kinds of fight. How often it then wins says how exploitable that model is,
        # which a score against fixed references cannot: they never adapt to it.
        from rating import FIGHTS

        levels = dict(curriculum.levels)
        levels["duel"] = dict(levels["duel"], bot=0, arena=2, distance=2, yaw=1, gear=2, kb=1, count=0)  # draws up to these
        vec.env_method("set_levels", levels)
        vec.env_method("set_exam", "duel", False)  # only duels, drawn below those limits; no dial moves
        vec.opponents = Opponents(vec.num_envs)
        vec.opponents.policies["target"] = ta.load_policy(a.exploit, model.device)
        vec.opponents.choose = lambda meta: "target"
    else:
        # Self-play (docs/training-design.md#self-play): some duels against the latest model or an older version of it.
        progress.pool = Pool(out, seed=a.seed)
        vec.opponents = Opponents(vec.num_envs)
        vec.opponents.choose = SelfPlay(vec.opponents, progress.pool, model.policy, lambda: ta.finished(curriculum.levels, "duel"), seed=a.seed)

    def stop(*_):
        # A stop (start.sh, a redeploy) ends learning where it is, and the run is saved below. A second one (uv passes
        # the signal on, and a script may signal both) must not cut that save short.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        model.learn(a.steps, callback=progress, reset_num_timesteps=resumed_steps == 0)
    except KeyboardInterrupt:
        progress.save_latest()  # nothing since the last save is lost
        print(f"stopped at {model.num_timesteps} decisions: saved", flush=True)
    if a.exploit:
        exploit_score(model, a.exploit, out)


def exploit_score(model, target, out, rounds=64):
    """How often the exploiter beats the frozen target, per kind of fight (wins + half the draws), with intervals."""
    from benchmark import first_rounds, wilson
    from rating import FIGHTS

    model.save(out / "latest")
    result = {}
    for fight, dials in FIGHTS.items():
        levels = Curriculum().levels
        levels["duel"].update(dials, bot=0)
        vec = SimVecEnv(rounds, levels, seed=515151)
        vec.env_method("set_exam", "duel", True)
        opp = Opponents(rounds)
        opp.policies["target"] = ta.load_policy(target, model.device)
        opp.choose = lambda meta: "target"
        infos = first_rounds(vec, model.policy, model.device, deterministic=False, opponents=opp)
        points = sum(1.0 if i["success"] else 0.0 if i["deaths"] > 0 else 0.5 for i in infos)
        lo, hi = wilson(points, rounds)
        result[fight] = dict(score=round(points / rounds, 3), low=round(lo, 3), high=round(hi, 3))
    (out / "exploit.json").write_text(json.dumps(dict(target=str(target), steps=model.num_timesteps, result=result), indent=1))
    print("exploiter against its target: " + ", ".join(f"{f} {r['score']:.2f} [{r['low']:.2f}-{r['high']:.2f}]" for f, r in result.items()), flush=True)


if __name__ == "__main__":
    main()
