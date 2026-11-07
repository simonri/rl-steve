"""Fights make sense: every bot beats a model that does nothing, walking up to the dummy and hitting it wins, a round
without opponents shows nothing in their slots, and the opponents' view of the model is from the right side.

    python test_duel.py
"""
import _path  # noqa: F401  (python/ on the import path)
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
    assert tail[0] == 1.0 and tail[3] > 0.2 and abs(tail[1]) < 0.1, tail[:20]  # 2 blocks ahead at the first distance: 2 / 8
    assert tail[61] == 1.0 and tail[68] == 1.0, "sword in hand, and it is a fight"
    # There is no hotbar input: the sword stays in hand.
    assert mcsim.ACTIONS == 9
    for _ in range(10):
        vec.step_async(idle(obs, 0))
        obs, _, _, _ = vec.step_wait()
    assert obs[0, mcsim.MOVE_OBS + 61] == 1.0, "the sword stays in hand"
    # Standing still facing the dummy 2 blocks off, the crosshair is often on it: that is worth a little every tick,
    # never more than 3 a round.
    vec = SimVecEnv(16, levels, seed=3)
    vec.env_method("set_exam", "duel", False)
    obs = vec.reset()
    aimed = []
    for t in range(700):
        vec.step_async(idle(obs, t))
        obs, _, dones, infos = vec.step_wait()
        aimed += [infos[i]["parts"].get("aimed", 0.0) for i in np.flatnonzero(dones)]
    assert aimed and max(aimed) > 0 and max(aimed) <= 3.0 * 0.1 + 1e-9, aimed  # parts are after the reward scale (0.1)
    # Swinging at a dummy 11 blocks off costs, every click; at 2 blocks it does not.
    def swing(obs, t):
        a = idle(obs, t)
        a[:, 8] = 1
        return a

    cost = {}
    for distance in (3, 0):  # 11 blocks, 2 blocks
        lv = curriculum.Curriculum().levels
        lv["duel"].update(bot=0, distance=distance, yaw=0)
        v = SimVecEnv(4, lv, seed=5)
        v.env_method("set_exam", "duel", True)  # the dials exactly as set
        o = v.reset()
        for t in range(20):
            v.step_async(swing(o, t))
            o, _, dones, infos = v.step_wait()
        cost[distance] = v.batch.report(0)["parts"].get("swing out of reach", 0.0)  # the round so far
    assert abs(cost[3] + 20 * 0.01 * 0.1) < 1e-9, cost  # every click at 11 blocks (parts are after the 0.1 scale)
    assert cost[0] > cost[3] / 2, cost  # at 2 blocks only the clicks after the dummy has been knocked away
    # A win is only a win alive and standing: a fighter that runs on after its hit, off the island, never wins while it
    # is in the air or falling, and falling off after the dummy is a loss.
    import skill_env
    from auto_env import AutoEnv
    from sim_env import FakeClient
    skill_env.McClient = FakeClient
    levels = curriculum.Curriculum().levels
    levels["duel"].update(bot=0, arena=1)
    results = []
    for seed in range(8):
        env = AutoEnv(0)
        env.set_levels(levels)
        env.set_exam("duel", False)
        env.reset(seed=seed)
        while True:
            o = env.obs
            p = o["players"][0]
            dx, dz = p["pos"][0] - o["pos"][0], p["pos"][2] - o["pos"][2]
            dyaw = (np.degrees(np.arctan2(-dx, dz)) - o["yaw"] + 540) % 360 - 180
            yaw = int(np.argmin([abs(s - dyaw) for s in (-60, -15, -5, -2, -0.5, 0, 0.5, 2, 5, 15, 60)]))
            _, _, term, trunc, info = env.step(np.array([1, 0, 0, 0, 0, 1, yaw, 5, env.t % 2]))  # sprint on, through it
            if term or trunc:
                if info["success"]:
                    assert env.obs["on_ground"] and env.obs["pos"][1] >= env._fall_line(), (seed, env.obs["pos"], env.obs["on_ground"])
                results.append((info["success"], info["deaths"]))
                break
    assert any(lost for _, lost in results), results  # running on, it does go over the edge
    print("duel ok")


if __name__ == "__main__":
    main()
