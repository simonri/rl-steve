"""The watchers' camera: still shots, few cuts, never inside a block, the player mostly in clear view.

Plays rounds on the simulator (AutoEnv on FakeClient) with inputs that wander, and checks the camera every tick
against every block of the course (pasted maps with their trees, platforms, and what the player placed).

    python test_camera.py [rounds]
"""
import math
import sys

import numpy as np

import curriculum
import skill_env
from auto_env import AutoEnv
from camera import FAR, HOLD, Shots
from sim_env import FakeClient


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    skill_env.McClient = FakeClient
    rng = np.random.default_rng(3)
    env = AutoEnv(0)
    cam = Shots()
    ticks = inside = hidden = far = cuts = quick = 0
    for n in range(rounds):
        family = ("treenan", "bedwars", "bridge", "hop")[n % 4]
        levels = curriculum.Curriculum().levels
        for d, v in curriculum.FAMILIES[family].items():
            levels[family][d] = int(rng.integers(len(v)))
        env.set_levels(levels)
        env.set_exam(family)
        env.reset(seed=500 + n)
        cam.new_round(env)
        world = set(cam.solid)  # everything the course has, before the player builds
        a = np.array([1, 0, 0, 0, 0, 1, 5, 5])
        since = 0
        for t in range(600):
            if rng.random() < 0.1:
                a = np.array([rng.integers(3), rng.integers(3), rng.random() < 0.3, rng.random() < 0.3, rng.random() < 0.2, rng.random() < 0.5, rng.integers(11), rng.integers(11)])
            _, _, term, trunc, _ = env.step(a)
            new = cam.update(env)
            if new is not None and t > 0:
                cuts += 1
                quick += since < HOLD
                since = 0
            since = since + 1 if t > 0 else 0
            at = cam.shot
            ticks += 1
            o = env.obs
            for c in o.get("clicks") or []:
                world.add((int(c[0]), int(c[1]), int(c[2])))
            cell = tuple(math.floor(v) for v in at[:3])
            inside += cell in world or (cell[0], cell[1] + 1, cell[2]) in world
            head = (o["pos"][0], o["pos"][1] + 1.6, o["pos"][2])
            steps = max(2, int(math.dist(at[:3], head) * 4))
            hidden += any(tuple(math.floor(at[i] + (head[i] - at[i]) * k / steps) for i in range(3)) in world for k in range(1, steps))
            far += math.dist(at[:3], head) > FAR * 1.6
            if term or trunc:
                break
    print(f"{rounds} rounds, {ticks} ticks: camera inside a block {inside}, view blocked {hidden}, too far {far}, cuts {cuts} ({quick} before {HOLD} ticks)")
    assert inside == 0, "the camera sat inside a block"
    assert hidden <= ticks * 0.1, "the view to the player was blocked too often"
    assert cuts <= ticks / 40, "the camera cut too often: more than one cut every two seconds"
    assert far <= ticks * 0.02, "the camera was too far away too often"
    print("camera ok")


if __name__ == "__main__":
    main()
