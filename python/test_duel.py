"""Fights make sense: every bot beats a model that does nothing, walking up to the dummy and hitting it wins, a round
without opponents shows nothing in their slots, and the opponents' view of the model is from the right side.

    python test_duel.py
"""
import numpy as np

import combat18
import curriculum
import mcsim
from sim_env import SimVecEnv


def play(bot, policy, n=32, ticks=900, seed=0, arena=0):
    levels = curriculum.Curriculum().levels
    levels["duel"]["bot"] = bot
    levels["duel"]["arena"] = arena
    vec = SimVecEnv(n, levels, seed=seed)
    vec.env_method("set_exam", "duel", False)
    obs = vec.reset()
    won = lost = rounds = 0
    for t in range(ticks):
        vec.step_async(policy(obs, t))
        obs, _, dones, infos = vec.step_wait()
        for i in np.flatnonzero(dones):
            rounds += 1
            won += infos[i]["success"]
            lost += infos[i]["deaths"] > 0
    return won, lost, rounds


def idle(obs, t):
    a = np.zeros((len(obs), mcsim.ACTIONS), np.int64)
    a[:, 6] = a[:, 7] = 5
    return a


def main():
    # Doing nothing loses to every bot that fights.
    for bot in range(1, len(curriculum.BOTS)):
        won, lost, rounds = play(bot, idle)
        assert rounds > 0 and won == 0 and lost >= rounds * 0.7, (curriculum.BOTS[bot], won, lost, rounds)  # the rest run out of time
    # The dummy never wins, and a round with it ends by time when nothing happens.
    won, lost, rounds = play(0, idle, ticks=700)
    assert won == 0 and lost == 0, (won, lost)
    # Gear from the dial: swords and Bedwars armor as in 1.8.
    assert combat18.duel_gear("wood") == ((5.0, 7), (5.0, 7))
    assert combat18.duel_gear("theirs better")[1] == (7.0, 11)
    # A movement round: the opponent slots are empty, the own fight values are there.
    levels = curriculum.Curriculum().levels
    vec = SimVecEnv(1, levels, seed=1)
    vec.env_method("set_exam", "bridge")
    obs = vec.reset()
    tail = obs[0, mcsim.MOVE_OBS:]
    assert not tail[: 3 * 20].any(), "no opponents, nothing in their slots"
    assert tail[60] == 1.0 and tail[61] == 0.0, "blocks in hand"
    # A duel: the first slot holds the opponent, straight ahead in the round's frame.
    levels["duel"]["bot"] = 0
    vec = SimVecEnv(1, levels, seed=2)
    vec.env_method("set_exam", "duel", False)
    obs = vec.reset()
    tail = obs[0, mcsim.MOVE_OBS:]
    assert tail[0] == 1.0 and tail[3] > 0.3 and abs(tail[1]) < 0.1, tail[:20]
    assert tail[61] == 1.0 and tail[69] == 1.0, "sword in hand, and it is a fight"
    print("duel ok")


if __name__ == "__main__":
    main()
