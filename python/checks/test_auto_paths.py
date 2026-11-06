"""Real game check: scripts walk every new path shape and clear the harder parkour, so nothing asked for is impossible."""
import _path  # noqa: F401  (python/ on the import path)
import math
import sys

from auto_env import NO_TURN, TURN_STEPS, AutoEnv
from curriculum import FAMILIES, Curriculum
from skill_env import Y

env = AutoEnv(int(sys.argv[1]))
rounds = int(sys.argv[2])
turn = lambda e: min(range(len(TURN_STEPS)), key=lambda i: abs(e / 2 - TURN_STEPS[i]))


def walker():
    """Faces the next block, walks there, and jumps when the next block is higher or further than a step."""
    o = env.obs
    x, y, z = o["pos"]
    tx, ty, tz = env._next_target()
    d = math.hypot(tx - x, tz - z)
    ahead, side = env._way_in_view(o)
    y_turn = turn(math.degrees(math.atan2(side, ahead)))
    if ahead < 0.7:
        return [0, 0, 0, 0, 0, 0, y_turn, NO_TURN]
    higher = ty > y + 0.5
    gap = env._edge_left(o) * 2 < 0.45
    jump = int(o["on_ground"] and (gap or (higher and d < 1.2)))
    if jump and ty <= y + 0.5 and d < 1.0:
        jump = 0  # the next block is right here, no need to jump
    return [1, 0, 0, 0, jump, 1 if d > 1.2 else 0, y_turn, NO_TURN]


def set_levels(family, **chosen):
    levels = Curriculum().levels
    levels[family].update(chosen)
    env.levels = levels
    env.set_exam(family)


index = lambda family, dial, value: FAMILIES[family][dial].index(value)
for wind in FAMILIES["path"]["wind"]:
    for slope in FAMILIES["path"]["slope"]:
        set_levels("path", length=3, ticks=0, yaw=0, wind=index("path", "wind", wind), slope=index("path", "slope", slope))
        wins, shape = 0, 0
        for _ in range(rounds):
            env.reset()
            shape = max(shape, len({c[1] for c in env.course}))
            done, info = False, {}
            while not done:
                _, _, term, trunc, info = env.step(walker())
                done = term or trunc
            wins += bool(info["success"])
        print(f"path {wind:7s} {slope:17s}: {wins}/{rounds} reached the goal, heights used {shape}", flush=True)

# Harder parkour must also build in the real world.
set_levels("hop", **{d: len(v) - 1 for d, v in FAMILIES["hop"].items()})
for _ in range(10):
    obs, _ = env.reset()
    names = env.c.call(op="blocks", min=[env.course[-1][0], env.course[-1][1], env.course[-1][2]], max=[env.course[-1][0], env.course[-1][1], env.course[-1][2]])["names"]
    for _ in range(40):
        env.c.step()
    x, y, z = env.course[-1]
    assert env.c.call(op="blocks", min=[x, y, z], max=[x, y, z])["names"][0] == "minecraft:stone", env.course[-1]
    assert Y - 11 <= y <= Y + 11
print(f"harder parkour builds, {len(env.course)} blocks", flush=True)
env.c.close()
print("ok")
