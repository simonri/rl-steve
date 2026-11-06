"""What bridging speed is physically possible, with and without our human limits? Scripted, no AI."""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import math
from collections import deque

import numpy as np

import skill_env
from skill_env import FLAT, Y, SkillEnv

GAP = 15
EDGE_Z = 2.0  # the start platform covers blocks -1..1, so its far edge is at z = 2


class Straight(SkillEnv):
    start_pitch = 80.0

    def _direction(self):
        return 0.0, 1.0

    def _spec(self):
        self.goal_x, self.goal_z, self.dy = 0, GAP + 4, 0
        gz = self.goal_z
        return dict(clear=[-10, Y - 8, -10, 10, Y + 8, gz + 12], platforms=[[-1, Y, -1, 1, Y, 1], [-2, Y, gz - 2, 2, Y, gz + 2]],
                    pos=[0.5, Y + 1, 1.5], yaw=180.0, pitch=self.start_pitch, blocks=64, block_radius=2)

    def _use_task(self, stage):
        self.played_stage = stage
        self.gap, self.directions, self.heights, self.yaw_range, self.ticks_per_block, self.start = GAP, "any", FLAT, 0, 40, "edge"
        self.extras, self.kind, self.blocked, self.detour, self.field = None, None, set(), 0, None
        self.budget = (3.0, 2)


LIMITS = {
    "no limits": dict(delay=(0, 0), late=0.0, noise=0.0, hold=1),
    "key hold 2 ticks": dict(delay=(0, 0), late=0.0, noise=0.0, hold=2),
    "+ late keys 30%": dict(delay=(0, 0), late=0.3, noise=0.0, hold=2),
    "+ reaction 3-5": dict(delay=(3, 5), late=0.3, noise=0.0, hold=2),
    "full limits": dict(delay=(3, 5), late=0.3, noise=0.3, hold=2),
}


def run(env, limits, policy, pitch, clicks, episodes, a=None):
    skill_env.MIN_KEY_TICKS = limits["hold"]
    env.delay_low, env.delay_high = limits["delay"]
    env.late_chance, env.aim_noise = limits["late"], limits["noise"]
    env.start_pitch = pitch
    wins, speeds = 0, []
    env.fails = {"fell": 0, "stalled": 0, "timeout": 0}
    for _ in range(episodes):
        env.reset()
        raw = deque([env.obs], maxlen=4)
        done, info = False, {}
        while not done:
            # raw holds one state per decision (2 ticks), newest last.
            back = min(math.ceil(env.delay / 2), len(raw) - 1)
            seen = raw[-1 - back]
            age = back * 2
            # Blocks laid so far decide where the next edge is. The agent also knows its own placements.
            edge = EDGE_Z + env.used
            if a.predict:
                # Knows what it pressed, so it counts on walking speed, plus the ticks until this input lands.
                speed_now = 0.065 if env.keys["sneak"] else 0.216
                ahead = seen["pos"][2] + speed_now * (age + 3)
            else:
                ahead = seen["pos"][2] + seen["vel"][2] * age
            sneak = policy(edge - ahead)
            action = [2, 0, int(sneak), clicks, 0, 0, 3, 3]
            _, _, term, trunc, info = env.step(action)
            raw.append(env.obs)
            done = term or trunc
        if not info.get("success"):
            env.fails["fell" if info.get("fell") else "stalled" if info.get("stalled") else "timeout"] += 1
        if info.get("success"):
            wins += 1
            speeds.append(info["blocks_per_s"])
    return wins, float(np.mean(speeds)) if speeds else 0.0


POLICIES = [("always sneak", lambda d: True, (78, 80, 82))]
POLICIES += [("no sneak", lambda d: False, (80, 82, 84, 86))]
for trigger in (0.15, 0.3, 0.5, 0.8):
    POLICIES.append((f"tap sneak at {trigger}", (lambda t: lambda d: d < t)(trigger), (76, 78, 80, 82)))

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=25601)
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--predict", action="store_true", help="tap sneak from a better prediction of its own movement")
p.add_argument("--final", action="store_true", help="full limits only")
p.add_argument("--sweep", action="store_true", help="tap sneak only, across reaction times, with the other limits on")
a = p.parse_args()
if a.final:
    LIMITS = {"full limits": LIMITS["full limits"]}
    POLICIES = [(n, f, (78, 80, 82)) for n, f, _ in POLICIES]
if a.predict:
    LIMITS = {f"reaction {d[0]}-{d[1]}": dict(delay=d, late=0.3, noise=0.3, hold=2) for d in ((1, 1), (2, 3), (3, 5))}
    POLICIES = [(f"tap sneak at {t}", (lambda t: lambda d: d < t)(t), (80, 82)) for t in (0.1, 0.2, 0.35, 0.5)]
    a.sweep = True
elif a.sweep:
    LIMITS = {"no limits": LIMITS["no limits"], "hold+late": LIMITS["+ late keys 30%"], "full": LIMITS["full limits"]}
    POLICIES = [p_ for p_ in POLICIES if p_[0].startswith("tap")]
    POLICIES = [(n, f, (78, 80, 82)) for n, f, _ in POLICIES]
env = Straight(a.port)
env.set_stage(0)
print(f"{'limits':18s} {'clicks/s':8s} {'technique':18s} pitch  goal  blocks/s", flush=True)
for lname, limits in LIMITS.items():
    for clicks, cps in (((2, 20),) if a.sweep or a.final else ((2, 20), (1, 10))):
        best = {}
        for pname, policy, pitches in POLICIES:
            for pitch in pitches:
                wins, speed = run(env, limits, policy, float(pitch), clicks, a.episodes, a)
                key = pname.split(" at ")[0]
                if wins >= math.ceil(a.episodes * 0.8) and speed > best.get(key, (0, 0, 0))[0]:
                    best[key] = (speed, f"{pname}", pitch, wins)
                print(f"RAW {lname} {cps} {pname} {pitch} {wins}/{a.episodes} {speed:.2f} {env.fails}", flush=True)
        for key, (speed, pname, pitch, wins) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            print(f"{lname:18s} {cps:<8d} {pname:18s} {pitch:<6d} {wins}/{a.episodes}  {speed:.2f}", flush=True)
        if not best:
            print(f"{lname:18s} {cps:<8d} no technique reached 80%", flush=True)
