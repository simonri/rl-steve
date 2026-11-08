"""Fights make sense: every bot beats a model that does nothing, walking up to the dummy and hitting it wins, a round
without opponents shows nothing in their slots, and the opponents' view of the model is from the right side.

    python test_duel.py
"""
import _path  # noqa: F401  (python/ on the import path)
import numpy as np

import combat18
import items
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
    # The hotbar (items.py): the sword selected, and with no key pressed it stays in hand.
    hb = tail[-items.HOTBAR_F :]
    sel = int(np.argmax(hb[-items.SLOTS :]))
    assert items.is_sword(int(np.argmax(hb[sel * (len(items.KINDS) + 1) : sel * (len(items.KINDS) + 1) + len(items.KINDS)]))), "the selected slot holds a sword"
    assert mcsim.ACTIONS == 10
    for _ in range(10):
        vec.step_async(idle(obs, 0))
        obs, _, _, _ = vec.step_wait()
    assert obs[0, mcsim.MOVE_OBS + 61] == 1.0, "the sword stays in hand"
    # Pressing the blocks' slot puts them in hand: the sword is no longer held.
    blocks = next(i for i in range(items.SLOTS) if items.placeable(int(np.argmax(hb[i * (len(items.KINDS) + 1) : i * (len(items.KINDS) + 1) + len(items.KINDS)]))))
    a = idle(obs, 0)
    a[0, 9] = blocks + 1
    vec.step_async(a)
    obs, _, _, _ = vec.step_wait()
    assert obs[0, mcsim.MOVE_OBS + 61] == 0.0 and obs[0, mcsim.MOVE_OBS + 60] == 1.0, "blocks in hand"
    # Aim is a potential (shaping.rs): standing still a whole round, the aim points add up to about −Φ(start) plus the
    # discount's drift, far below what 600 ticks of a per-tick bonus paid. Bound: 0.1·π·(1 + 600·0.005), scaled by 0.1.
    vec = SimVecEnv(16, levels, seed=3)
    vec.env_method("set_exam", "duel", False)
    obs = vec.reset()
    aim = []
    for t in range(700):
        vec.step_async(idle(obs, t))
        obs, _, dones, infos = vec.step_wait()
        aim += [infos[i]["parts"].get("aim", 0.0) for i in np.flatnonzero(dones)]
    assert aim and max(abs(a) for a in aim) <= 0.1 * (0.1 * np.pi * (1 + 600 * 0.005)) + 1e-9, aim
    # Swinging at a dummy 11 blocks off costs, every click; at 2 blocks only the clicks that miss.
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
        cost[distance] = v.batch.report(0)["parts"].get("missed swing", 0.0)  # the round so far
    assert abs(cost[3] + 20 * 0.01 * 0.1) < 1e-9, cost  # every click at 11 blocks (parts are after the 0.1 scale)
    assert cost[0] > cost[3], cost  # at 2 blocks the clicks that land cost nothing
    # A fifth of training rounds practise the next opponent, without testing anything; exams never do.
    lv = curriculum.Curriculum().levels
    lv["duel"]["bot"] = 1
    rng = np.random.default_rng(0)
    draws = [curriculum.draw(lv, "duel", rng) for _ in range(2000)]
    nxt = [t for s, t in draws if s["bot"] == "medium walker"]
    assert 0.15 < len(nxt) / len(draws) < 0.25 and all(t is None for t in nxt), len(nxt)
    assert not any(s["bot"] == "medium walker" for s, _ in (curriculum.draw(lv, "duel", rng, practice_next=False) for _ in range(500)))
    # Hop practises its longer courses and bends the same way, even while those dials are down.
    hop = curriculum.FAMILIES["hop"]
    assert not any(s["longer"] != hop["longer"][0] for s, _ in (curriculum.draw(lv, "hop", rng) for _ in range(500))), "not before the rest of hop"
    lv["hop"] = {d: len(v) - 1 for d, v in hop.items()} | {"longer": 0, "bend": 0}
    draws = [curriculum.draw(lv, "hop", rng) for _ in range(2000)]
    ahead = [t for s, t in draws if s["longer"] == hop["longer"][1] or s["bend"] == hop["bend"][1]]
    assert 0.15 < len(ahead) / len(draws) < 0.25 and all(t is None for t in ahead), len(ahead)
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
    # "gap": two islands with only void between, as many blocks as the distance dial less two; it has blocks to build.
    from sim_env import CourseMaker
    lv = curriculum.Curriculum().levels
    lv["duel"].update(arena=curriculum.FAMILIES["duel"]["arena"].index("gap"), distance=3, bot=0)  # 11 blocks: 9 of void
    maker = CourseMaker(lv, seed=3)
    maker.set_exam("duel", True)
    spec, _ = maker.draw()
    cols = {(b[0], b[2]) for b in spec["boxes"]}
    gx, gz = spec["goal"]
    step = (int(np.sign(gx)), int(np.sign(gz)))
    line = [(step[0] * a, step[1] * a) for a in range(12)]
    assert [c in cols for c in line] == [True] * 2 + [False] * 9 + [True], [c in cols for c in line]
    assert spec["blocks"] >= 9
    # A course to build on starts with something else in hand now and then (RANDOM_HAND; never in an exam, and such a
    # round then tests nothing, so it cannot move a dial), and the first time it has its blocks in hand that pays once
    # (took blocks); a course that starts with them in hand never pays it.
    import auto_env
    K = len(items.KINDS) + 1

    def hands(obs):
        hb = obs[:, mcsim.MOVE_OBS + mcsim.EXTRA - items.HOTBAR_F :]
        sel = np.argmax(hb[:, items.SLOTS * K :], axis=1)
        kinds = [[int(np.argmax(hb[i, s * K : s * K + K - 1])) if hb[i, s * K + K - 1] > 0 else 0 for s in range(items.SLOTS)] for i in range(len(obs))]
        holds = np.array([items.placeable(kinds[i][s]) for i, s in enumerate(sel)])
        blocks = [next((s for s in range(items.SLOTS) if items.placeable(kinds[i][s])), None) for i in range(len(obs))]
        return holds, blocks

    lv = curriculum.Curriculum().levels
    v = SimVecEnv(64, lv, seed=8)
    v.env_method("set_exam", "bridge", False)
    holds, _ = hands(v.reset())
    assert holds.all(), "an exam starts with the blocks in hand"
    v = SimVecEnv(256, lv, seed=8)  # training rounds, of every open kind
    obs = v.reset()
    holds, blocks = hands(obs)
    building = np.array([b is not None for b in blocks]) & np.array([m["family"] != "duel" for m in v.meta])
    assert 0.1 < 1 - holds[building].mean() < 0.5, holds[building].mean()  # about RANDOM_HAND of every training course
    assert all(v.meta[i]["tested"] is None for i in np.flatnonzero(building & ~holds)), "a round with another hand tests nothing"
    a = idle(obs, 0)
    a[:, 9] = [b + 1 if b is not None else 0 for b in blocks]
    for t in range(3):  # pressing the blocks' key three times: it pays once, only where they were not in hand
        v.step_async(a)
        v.step_wait()
    paid = np.array([v.batch.report(i)["parts"].get("took blocks", 0.0) for i in range(256)])
    has = np.array([b is not None for b in blocks])
    course = np.array([m["family"] != "duel" for m in v.meta])
    assert np.allclose(paid[has & ~holds & course], auto_env.TOOK_BLOCKS_BONUS * 0.1) and np.all(paid[(holds | ~has) & course] == 0), (paid, holds)
    # The same in a fight that can only be won by building (gap arena, 5+ blocks of void): it starts with the sword,
    # and the first time it has blocks in hand pays once. A fight on a platform never pays it.
    for arena, distance, pays in (("gap", 2, True), ("gap", 1, False), ("platform", 2, False)):
        lv = curriculum.Curriculum().levels
        lv["duel"].update(arena=curriculum.FAMILIES["duel"]["arena"].index(arena), distance=distance, bot=0)
        v = SimVecEnv(8, lv, seed=9)
        v.env_method("set_exam", "duel", True)
        obs = v.reset()
        hb = obs[:, mcsim.MOVE_OBS + mcsim.EXTRA - items.HOTBAR_F :]
        a = idle(obs, 0)
        a[:, 9] = [next(s for s in range(items.SLOTS) if items.placeable(int(np.argmax(hb[i, s * K : s * K + K - 1])))) + 1 for i in range(8)]
        v.step_async(a)
        v.step_wait()
        paid = [v.batch.report(i)["parts"].get("took blocks", 0.0) for i in range(8)]
        assert all((p > 0) == pays for p in paid), (arena, distance, paid)
    # Only a hit with a sword pays in full: the same hits on the dummy with blocks in hand pay NON_SWORD of it.
    dealt = {}
    for hand in ("sword", "blocks"):
        lv = curriculum.Curriculum().levels
        lv["duel"].update(bot=0, distance=0, yaw=0, arena=0)
        v = SimVecEnv(4, lv, seed=5)
        v.env_method("set_exam", "duel", True)
        obs = v.reset()
        hb = obs[:, mcsim.MOVE_OBS + mcsim.EXTRA - items.HOTBAR_F :]
        for t in range(30):
            a = idle(obs, t)
            a[:, 8] = t % 3 == 0
            if t == 0 and hand == "blocks":
                a[:, 9] = [next(s for s in range(items.SLOTS) if items.placeable(int(np.argmax(hb[i, s * K : s * K + K - 1])))) + 1 for i in range(4)]
            v.step_async(a)
            obs, _, _, _ = v.step_wait()
        dealt[hand] = v.batch.report(0)["parts"].get("damage dealt", 0.0)
    # A sword does 5 a hit, blocks 1: per hit that lands, blocks pay 1/5 of the damage and NON_SWORD of the points for it.
    assert dealt["sword"] > 0 and 0 < dealt["blocks"] < dealt["sword"] * mcsim.NON_SWORD, dealt
    print("duel ok")


if __name__ == "__main__":
    main()
