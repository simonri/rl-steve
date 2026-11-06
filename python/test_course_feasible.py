"""Real game check: a scripted jumper with all human limits, on the climbing and single block courses."""
import sys

import numpy as np

from skill_env import WIDE_ZIGZAG, R, STAGES, YAW_STEPS, SkillEnv, Y

env = SkillEnv(int(sys.argv[1]))
EPISODES = int(sys.argv[2])


def turn_index(error):
    return min(range(len(YAW_STEPS)), key=lambda i: abs(error / 2 - YAW_STEPS[i]))


def jumper(margin, sprint):
    o = env.obs
    ahead, side = env._way_in_view(o)
    yaw_i = turn_index(float(np.degrees(np.arctan2(side, ahead))))
    edge = env._edge_left(o) * 2
    x, y, z = o["pos"]
    rx, rz = env._route(x, z)
    feet_x, feet_y, feet_z = (int(v) for v in o["feet"])
    size = 2 * R + 1
    # A block at foot height just ahead has to be jumped onto.
    nx, nz = int(np.floor(x + rx * 0.6)) - feet_x, int(np.floor(z + rz * 0.6)) - feet_z
    wall = abs(nx) <= R and abs(nz) <= R and o["solid"][(nx + R) * size * size + R * size + (nz + R)] == "1"
    jump = int(o["on_ground"] and (edge < margin or wall))
    # In the air, keep going forward only as far as it takes to reach the block.
    return [1 if ahead > 0.7 else 0, 0, 0, 0, jump, int(sprint), yaw_i, 3]


def corner_jump(inset, let_go, sprint):
    """Stands near the corner facing the next block, without running up, and jumps straight from there."""
    o = env.obs
    x, _, z = o["pos"]
    tx, tz = env._next_block(x, z)
    dist = float(np.hypot(tx - x, tz - z))
    ahead, side = env._way_in_view(o)
    yaw_i = turn_index(float(np.degrees(np.arctan2(side, ahead))))
    speed = float(np.hypot(o["vel"][0], o["vel"][2]))
    if not o["on_ground"]:
        return [1 if dist > let_go else 0, 0, 0, 0, 0, int(sprint and dist > let_go), yaw_i, 3]
    here = env._standing_on(o)
    if ahead < 0.985 or speed > 0.05 and here is not None and here not in env.stopped_on:
        return [0, 0, 1, 0, 0, 0, yaw_i, 3]
    # On the start platform, walk to its edge first.
    if here is None:
        if env._edge_left(o) * 2 > 0.3:
            return [1, 0, 1, 0, 0, 0, yaw_i, 3]
    else:
        cx, cz = here[0] + 0.5, here[2] + 0.5
        moved = (x - cx) * (tx - cx) + (z - cz) * (tz - cz)
        moved /= float(np.hypot(tx - cx, tz - cz)) or 1.0
        if moved < inset and env.sneak_held < 16:
            return [1, 0, 1, 0, 0, 0, yaw_i, 3]
    return [1, 0, 0, 0, 1, int(sprint), yaw_i, 3]


env.set_stage(next(i for i, t in enumerate(STAGES) if t[6] == WIDE_ZIGZAG))
env.delay_low, env.delay_high, env.late_chance, env.aim_noise = 1, 1, 0.3, 0.3
for inset, let_go in ((0.35, 0.6), (0.35, 0.3)):
    clean = died = 0
    for _ in range(EPISODES):
        env.reset()
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(corner_jump(inset, let_go, True))
            done = term or trunc
        clean += bool(info["success"] and not info["deaths"])
        died += info["deaths"] > 0
    print(f"careful script, 4 jumps with poles, let go {let_go}: {clean}/{EPISODES} clean, {died} died", flush=True)
clean = 0
for _ in range(EPISODES):
    env.reset()
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(jumper(0.5, True))
        done = term or trunc
    clean += bool(info["success"] and not info["deaths"])
print(f"straight sprinting script: {clean}/{EPISODES} clean", flush=True)
# The poles are really standing in the world.
env.reset()
for _ in range(30):
    env.c.step()
for x, z in env.poles:
    names = env.c.call(op="blocks", min=[x, Y + 1, z], max=[x, Y + 3, z])["names"]
    assert all(n == "minecraft:stone" for n in names), (x, z, names)
print("poles in the world:", len(env.poles), flush=True)
env.c.close()
print("ok")
