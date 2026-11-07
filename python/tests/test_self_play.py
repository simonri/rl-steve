"""Self-play in the simulator: an opponent played by a model acts on what it sees (it moves, where its bot, a dummy,
stands still), rounds without one are as before, and its memory starts again with each round.

    python test_self_play.py
"""
import _path  # noqa: F401  (python/ on the import path)
import numpy as np
import torch

import auto_env
import curriculum
import mcsim
import train_auto as ta
from opponents import BOT, Opponents
from sim_env import SimVecEnv
from spatial import MemoryPolicy

auto_env.DUEL_TICKS = 100  # short fights: what is checked here happens in the first ticks or at a round's end
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the model is slow on the processor a tick at a time


def where_the_opponent_is(obs):
    """The first opponent slot's position relative to the model (duel.rs features 1..3)."""
    return obs[:, mcsim.MOVE_OBS + 1 : mcsim.MOVE_OBS + 4].copy()


def run(played_by, ticks=60):
    levels = curriculum.Curriculum().levels
    levels["duel"].update(bot=0, distance=1, arena=0)  # a dummy 4 blocks off, which never moves by itself
    vec = SimVecEnv(6, levels, seed=4)
    vec.env_method("set_exam", "duel", True)
    if played_by is not BOT:
        opp = Opponents(vec.num_envs)
        opp.policies["test"] = played_by
        opp.choose = lambda meta: "test"
        vec.opponents = opp
    obs = vec.reset()
    start = where_the_opponent_is(obs)
    idle = np.tile(np.array([0, 0, 0, 0, 0, 0, 5, 5, 0, 0]), (vec.num_envs, 1))
    for _ in range(ticks):
        vec.step_async(idle)
        obs, _, dones, _ = vec.step_wait()
        assert not dones.any(), "rounds of 100 ticks do not end in 60"
    return np.abs(where_the_opponent_is(obs) - start).max(), vec


def main():
    moved_by_bot, _ = run(BOT)
    assert moved_by_bot < 1e-6, f"the dummy stands still: {moved_by_bot}"
    vec0 = SimVecEnv(1, curriculum.Curriculum().levels, seed=0)
    policy = ta.SplitPPO(MemoryPolicy, vec0, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=ta.POLICY, device=DEVICE, seed=0).policy
    ta.fast_convolutions()
    moved, vec = run(policy)
    assert moved > 0.05, f"played by a model, it acts: moved {moved}"
    assert all(p == "test" for p in vec.opponents.players)
    # a new round starts its memory again
    vec._new(0)
    assert vec.opponents.start[0]
    pool_and_mix(policy)
    print("self-play ok")


def pool_and_mix(policy):
    """The pool: PFSP favours the versions the model loses to, keeps a floor for the rest, remembers across restarts;
    the mix: bots, the latest model and the pool in the documented shares, and rounds say who played."""
    import tempfile
    from pathlib import Path

    import pool as P

    w = P.pfsp_weights([0.0, 0.5, 1.0])
    assert w[0] > w[1] > w[2] and abs(w.sum() - 1) < 1e-9 and w[2] >= P.UNIFORM_FLOOR / 3 - 1e-12, w

    class Saver:  # model.save as SB3 does it: the name plus .zip
        def __init__(self, policy):
            self.policy = policy

        def save(self, path):
            src = Path(path).with_suffix(".zip")
            ta.SplitPPO(MemoryPolicy, SimVecEnv(1, curriculum.Curriculum().levels, seed=0), n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs=ta.POLICY, device="cpu", seed=0).save(src)

    with tempfile.TemporaryDirectory() as d:
        pool = P.Pool(d)
        assert pool.due(0) and pool.pick() is None
        name = pool.add(Saver(policy), 10_000_000)
        assert not pool.due(12_000_000) and pool.due(15_000_000)
        for _ in range(20):
            pool.record(name, 1.0)
        assert pool.score(pool.versions[0]) > 0.9
        pool.save()
        again = P.Pool(d)
        assert again.versions == pool.versions and again.last == 10_000_000
        # Loading an opponent must not switch training's convolutions to cuDNN's slow deterministic mode (load_policy).
        import torch
        if torch.cuda.is_available():
            ta.fast_convolutions()
            ta.load_policy(pool.folder / pool.versions[0]["file"], "cuda")
            assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
            ta.SplitPPO.load(pool.folder / pool.versions[0]["file"], device="cuda")
            assert torch.backends.cudnn.deterministic, "the trap load_policy avoids is still there"

        levels = curriculum.Curriculum().levels
        levels["duel"].update(bot=0, distance=1, arena=0)
        vec = SimVecEnv(4, levels, seed=4)
        opp = Opponents(vec.num_envs)
        mix = P.SelfPlay(opp, pool, policy, lambda: False, seed=1)
        picks = [mix({"family": "duel"}) for _ in range(4000)]
        bots = picks.count(BOT) / len(picks)
        latest = picks.count("latest") / len(picks)
        assert abs(bots - P.BOT_SHARE_LEARNING) < 0.03 and abs(latest - (1 - bots) * P.LATEST_SHARE) < 0.03, (bots, latest)
        assert picks.count(name) > 0 and name in opp.policies
        mix.duel_finished = lambda: True
        assert abs([mix({"family": "duel"}) for _ in range(4000)].count(BOT) / 4000 - P.BOT_SHARE_DONE) < 0.02
        # Rounds played by a model say so in their info, so training keeps them out of the curriculum.
        vec.env_method("set_exam", "duel", True)
        opp.choose = lambda meta: "latest"
        vec.opponents = opp
        obs = vec.reset()
        idle = np.tile(np.array([0, 0, 0, 0, 0, 0, 5, 5, 0, 0]), (vec.num_envs, 1))
        seen = []
        for _ in range(150):
            vec.step_async(idle)
            obs, _, dones, infos = vec.step_wait()
            seen += [infos[i].get("opponent") for i in np.flatnonzero(dones)]
        assert seen and all(s == "latest" for s in seen), seen


if __name__ == "__main__":
    main()
