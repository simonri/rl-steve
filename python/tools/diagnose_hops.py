"""Why does it fall on single blocks? For every jump: where from, how fast, and where it lands. No training."""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import math

import numpy as np
from stable_baselines3 import PPO

from skill_env import HOPS, STAGES, SkillEnv

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--port", type=int, default=25601)
p.add_argument("--kind", default=HOPS)
p.add_argument("--episodes", type=int, default=30)
a = p.parse_args()
env = SkillEnv(a.port, stage=next(i for i, t in enumerate(STAGES) if t[6] == a.kind))
model = PPO.load(a.model, device="cpu")
for det in (True, False):
    results = {"goal": 0, "clean": 0, "died": 0, "stuck": 0}
    takeoff_left, takeoff_speed, flew, outcomes, sprint_at, reached, aim_at = [], [], [], {"short": 0, "long": 0, "side": 0, "landed": 0}, [], [], []
    for _ in range(a.episodes):
        env.reset()
        obs = env._seen(env._vec(env.obs))
        done, info = False, {}
        air_from = None
        cells = {(x, z) for x, z in [(c[0], c[2]) for c in env.course_list]} if hasattr(env, "course_list") else None
        farthest = 0
        while not done:
            act, _ = model.predict(obs, deterministic=det)
            before = env.obs
            obs, _, term, trunc, info = env.step(act)
            o = env.obs
            if before["on_ground"] and not o["on_ground"] and o["vel"][1] > 0.2:
                air_from = before["pos"][:]
                takeoff_left.append(env._edge_left(before) * 2)
                takeoff_speed.append(math.hypot(before["vel"][0], before["vel"][2]) * 20)
                sprint_at.append(bool(act[5]))
                aim_at.append(env._way_in_view(before)[0])
            if air_from is not None and (o["on_ground"] or info.get("respawned")):
                flew.append(math.hypot(before["pos"][0] - air_from[0], before["pos"][2] - air_from[2]))
                if info.get("respawned"):
                    rx, rz = env._route(air_from[0], air_from[2])
                    along = (before["pos"][0] - air_from[0]) * rx + (before["pos"][2] - air_from[2]) * rz
                    across = abs((before["pos"][0] - air_from[0]) * rz - (before["pos"][2] - air_from[2]) * rx)
                    outcomes["side" if across > 0.8 else "long" if along > 2.2 else "short"] += 1
                else:
                    outcomes["landed"] += 1
                air_from = None
            if o["on_ground"]:
                farthest = max(farthest, abs(o["pos"][0]) + abs(o["pos"][2]))
            done = term or trunc
        results["goal"] += info["success"]
        results["clean"] += bool(info["success"] and not info["deaths"])
        results["died"] += info["deaths"] > 0
        results["stuck"] += info["stalled"]
        reached.append(farthest)
    print(f"--- {'best action' if det else 'with randomness'}: {results}, farthest on the ground {np.mean(reached):.1f} of ~{env.goal_x and abs(env.goal_x) or abs(env.goal_z)}")
    print(f"jumps {len(takeoff_left)}, jump distance before the edge: median {np.median(takeoff_left):.2f}, quartiles {np.percentile(takeoff_left, 25):.2f}..{np.percentile(takeoff_left, 75):.2f}")
    print(f"speed at takeoff (blocks/s): median {np.median(takeoff_speed):.2f}, quartiles {np.percentile(takeoff_speed, 25):.2f}..{np.percentile(takeoff_speed, 75):.2f}; sprint on {np.mean(sprint_at):.0%}; looking along the way (1 is straight): {np.median(aim_at):.2f}")
    print(f"air distance: median {np.median(flew):.2f}; jump outcomes {outcomes}")
env.c.close()
