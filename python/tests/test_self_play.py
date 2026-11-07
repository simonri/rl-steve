"""Self-play in the simulator: an opponent played by a model acts on what it sees (it moves, where its bot, a dummy,
stands still), rounds without one are as before, and its memory starts again with each round.

    python test_self_play.py
"""
import _path  # noqa: F401  (python/ on the import path)
import numpy as np

import curriculum
import mcsim
import train_auto as ta
from opponents import BOT, Opponents
from sim_env import SimVecEnv
from spatial import MemoryPolicy


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
    idle = np.tile(np.array([0, 0, 0, 0, 0, 0, 5, 5, 0]), (vec.num_envs, 1))
    for _ in range(ticks):
        vec.step_async(idle)
        obs, _, dones, _ = vec.step_wait()
        assert not dones.any(), "rounds of 600 ticks do not end in 60"
    return np.abs(where_the_opponent_is(obs) - start).max(), vec


def main():
    moved_by_bot, _ = run(BOT)
    assert moved_by_bot < 1e-6, f"the dummy stands still: {moved_by_bot}"
    vec0 = SimVecEnv(1, curriculum.Curriculum().levels, seed=0)
    policy = ta.SplitPPO(MemoryPolicy, vec0, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=ta.POLICY, device="cpu", seed=0).policy
    moved, vec = run(policy)
    assert moved > 0.05, f"played by a model, it acts: moved {moved}"
    assert all(p == "test" for p in vec.opponents.players)
    # a new round starts its memory again
    vec._new(0)
    assert vec.opponents.start[0]
    print("self-play ok")


if __name__ == "__main__":
    main()
