"""How many falls come right after shift was let go by force? Uses a saved model, no training."""
import _path  # noqa: F401  (python/ on the import path)
import argparse

from stable_baselines3 import PPO

from skill_env import SkillEnv

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--port", type=int, default=25601)
p.add_argument("--stage", type=int, default=9)
p.add_argument("--episodes", type=int, default=80)
a = p.parse_args()


class Traced(SkillEnv):
    def _press(self, wanted):
        before = self.keys["sneak"]
        wanted_sneak = wanted["sneak"]
        out = super()._press(wanted)
        if before and not out["sneak"] and wanted_sneak:
            self.forced_at = self.t
        return out


env = Traced(a.port, stage=a.stage)
model = PPO.load(a.model, device="cpu")
for det in (True, False):
    goals = falls = stuck = 0
    near = {5: 0, 10: 0, 20: 0}
    forced_per_episode = 0
    for _ in range(a.episodes):
        env.forced_at = -1000
        forced = 0
        last = -1000
        obs, _ = env.reset()
        env.forced_at = -1000
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=det)
            obs, _, term, trunc, info = env.step(act)
            if env.forced_at != last:
                forced += 1
                last = env.forced_at
            done = term or trunc
        forced_per_episode += forced
        goals += info["success"]
        stuck += info["stalled"]
        if info["fell"]:
            falls += 1
            for n in near:
                near[n] += env.t - env.forced_at <= n
    print(f"{'best action' if det else 'sampled':12s} goal {goals}/{a.episodes} fell {falls} stuck {stuck} | "
          f"falls within 5/10/20 ticks of forced release: {near[5]}/{near[10]}/{near[20]} | forced releases per episode {forced_per_episode / a.episodes:.1f}", flush=True)
env.c.close()
