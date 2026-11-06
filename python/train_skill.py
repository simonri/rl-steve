"""Trains the bridging skill on several real game instances, and shows progress in the game."""
import argparse
import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.distributions import Distribution

import scoreboard as sb
from skill_env import BLOCK_GOAL_FROM_STAGE, LAST_JUMP_STAGE, OLD_STAGE_MIX, STAGES, SkillEnv

ROOT = Path(__file__).resolve().parent.parent
PASS_SUCCESS = 0.85
PASS_BY_KIND = {"wide_zigzag": 0.5}  # a hard stage: a script could make the jumps but not 8 in a row
PASS_BLOCK_USE = 1.5
PASS_UNUSED = 1.0  # blocks per episode that did nothing
PASS_IN_A_ROW = 2  # single tests are noisy, so one lucky result should not be enough
EVAL_EVERY = 10_000
EVAL_EPISODES = 30
SKILL_TEST_EVERY = 50_000  # how often an old skill is checked, so forgetting shows up


def make_env(port):
    def build():
        env = SkillEnv(port, mix=OLD_STAGE_MIX)
        env.camera_every = 2.0  # episodes run far faster than anyone can follow here
        return Monitor(env, info_keywords=("success",))

    return build


class Curriculum(BaseCallback):
    def __init__(self, vec, out, stage, window=100):
        super().__init__()
        self.vec = vec
        self.out = out
        self.stage = stage
        self.t0 = time.time()
        self.first_step = None
        self.episodes = 0
        self.last_log = 0
        self.last_save = 0
        self.last_eval = 0
        self.last_text = 0.0
        self.eval_success = 0.0
        self.eval_blocks = 0.0
        self.eval_unused = 0.0
        self.passes = 0
        self.eval_episodes = 0
        self.results = deque(maxlen=window)
        self.falls = deque(maxlen=window)
        self.stalls = deque(maxlen=window)
        self.damage = deque(maxlen=window)
        self.speeds = deque(maxlen=window)
        self.block_use = deque(maxlen=window)
        self.unused = deque(maxlen=window)
        self.shift = deque(maxlen=window)
        self.parts = deque(maxlen=window)
        self.jump_test = None
        self.last_skill_test = 0
        self.finished = False
        new = not (out / "progress.csv").exists()
        self.f = open(out / "progress.csv", "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["steps", "seconds", "stage", "gap", "directions", "heights", "ticks_per_block", "success", "fell",
                             "stalled", "damage", "blocks_per_s", "block_use", "unused", "eval_success", "eval_block_use", "episodes", "shift"])

    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def needed(self):
        """Share of test rounds this stage has to win."""
        return PASS_BY_KIND.get(STAGES[self.stage][6], PASS_SUCCESS)

    def lines(self):
        task = STAGES[self.stage]
        m = self.mean
        blocks_matter = self.stage >= BLOCK_GOAL_FROM_STAGE
        pass_dots = [["●" * self.passes, sb.GOOD], ["●" * (PASS_IN_A_ROW - self.passes), sb.DIM]]
        return [
            sb.stage_bar(self.stage + 1, len(STAGES)),
            sb.task_line(task[0], min(task[2]), max(task[2]), task[6]),
            sb.BLANK,
            [["Test ", sb.LABEL], [f"{self.eval_success:.0%} ", sb.higher_is_better(self.eval_success, self.needed(), 0.6)], *pass_dots],
            sb.stat("Train", f"{m(self.results):.0%}", sb.higher_is_better(m(self.results), 0.7, 0.4)),
            sb.stat("Died", f"{m(self.falls):.0%}", sb.lower_is_better(m(self.falls), 0.1, 0.3)),
            sb.stat("Stuck", f"{m(self.stalls):.0%}", sb.lower_is_better(m(self.stalls), 0.1, 0.3)),
            sb.stat("Shift", f"{m(self.shift):.0%}", sb.VALUE, "of the time"),
            sb.BLANK,
            sb.stat("Speed", f"{m(self.speeds):.2f}", sb.higher_is_better(m(self.speeds), 2.5, 1.7), "blocks/s"),
            sb.stat("Blocks", f"{m(self.block_use):.2f}x", sb.lower_is_better(m(self.block_use), PASS_BLOCK_USE, 2.0) if blocks_matter else sb.VALUE, f"need {PASS_BLOCK_USE}" if blocks_matter else None),
            sb.stat("Waste", f"{m(self.unused):.2f}", sb.lower_is_better(m(self.unused), PASS_UNUSED, 1.5) if blocks_matter else sb.VALUE, f"need {PASS_UNUSED:.0f}" if blocks_matter else None),
            sb.BLANK,
            sb.stat("Jump skill", "-" if self.jump_test is None else f"{self.jump_test:.0%}", sb.VALUE if self.jump_test is None else sb.higher_is_better(self.jump_test, PASS_SUCCESS, 0.6)),
            sb.stat("Steps", sb.compact(self.num_timesteps), sb.VALUE),
        ]

    def show(self, title="TRAINING"):
        """Keeps the scoreboard on the training server up to date, about once a second."""
        if time.time() - self.last_text >= 1.0:
            self.last_text = time.time()
            color = "yellow" if title == "TESTING" else "aqua"
            self.vec.env_method("sidebar", [["BRIDGE AI ", "white", True], [title, color, True]], self.lines(), indices=0)

    def mean_parts(self):
        """Average points per round from each rule, so it is visible what actually drives the model."""
        total = {}
        for round_parts in self.parts:
            for rule, value in round_parts.items():
                total[rule] = total.get(rule, 0.0) + value
        return {rule: round(value / max(len(self.parts), 1), 3) for rule, value in total.items()}

    def publish(self):
        task = STAGES[self.stage]
        status = dict(steps=self.num_timesteps, stage=self.stage, stages=len(STAGES), gap=task[0], directions=task[1], heights=len(task[2]),
                      ticks_per_block=task[4], success=self.mean(self.results), fell=self.mean(self.falls), stalled=self.mean(self.stalls),
                      damage=self.mean(self.damage), speed=self.mean(self.speeds), blocks=self.mean(self.block_use), unused=self.mean(self.unused), shift=self.mean(self.shift),
                      eval_success=self.eval_success, eval_blocks=self.eval_blocks, jump_test=self.jump_test, parts=self.mean_parts(),
                      seconds=time.time() - self.t0,
                      pass_success=self.needed(), pass_blocks=PASS_BLOCK_USE, pass_unused=PASS_UNUSED, pass_in_a_row=PASS_IN_A_ROW, block_goal_from=BLOCK_GOAL_FROM_STAGE)
        tmp = self.out / "status.tmp"
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.out / "status.json")
        return status

    def save_latest(self):
        self.model.save(self.out / "latest_tmp")
        os.replace(self.out / "latest_tmp.zip", self.out / "latest.zip")

    def _on_step(self):
        if self.first_step is None:
            self.first_step = self.num_timesteps
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and info.get("current"):
                self.episodes += 1
                self.results.append(bool(info.get("success")))
                self.falls.append(bool(info.get("fell")))
                self.stalls.append(bool(info.get("stalled")))
                self.damage.append(float(info.get("damage", 0.0)))
                self.shift.append(float(info.get("shift_share", 0.0)))
                self.parts.append(info.get("parts", {}))
                if info.get("success"):
                    self.speeds.append(info["blocks_per_s"])
                    self.block_use.append(info["blocks_used"] / info["blocks_needed"])
                    self.unused.append(info["unused"])
        self.show()
        if self.num_timesteps - self.last_log >= 4000:
            self.last_log = self.num_timesteps
            s = self.publish()
            self.w.writerow([s["steps"], round(s["seconds"]), s["stage"], s["gap"], s["directions"], s["heights"], s["ticks_per_block"], round(s["success"], 3),
                             round(s["fell"], 3), round(s["stalled"], 3), round(s["damage"], 2), round(s["speed"], 2), round(s["blocks"], 2), round(s["unused"], 2),
                             round(s["eval_success"], 3), round(s["eval_blocks"], 2), self.episodes, round(s["shift"], 3)])
            self.f.flush()
            speed = (self.num_timesteps - self.first_step) / max(s["seconds"], 1)
            print(self.num_timesteps, f"stage {s['stage']} gap {s['gap']} {s['directions']} h{s['heights']} tpb {s['ticks_per_block']}",
                  f"success {s['success']:.2f} test {s['eval_success']:.2f} fell {s['fell']:.2f} stalled {s['stalled']:.2f}",
                  f"damage {s['damage']:.2f} speed {s['speed']:.2f} blockuse {s['blocks']:.2f} unused {s['unused']:.2f} shift {s['shift']:.2f}", f"{speed:.0f} decisions/s", flush=True)
        if self.num_timesteps - self.last_save >= 20_000:
            self.last_save = self.num_timesteps
            self.save_latest()
        return not self.finished

    def _on_rollout_end(self):
        if self.num_timesteps - self.last_eval < EVAL_EVERY:
            return
        self.last_eval = self.num_timesteps
        self.evaluate()

    def test_jump_skill(self):
        """Is the jumping still there? Building stages no longer train it."""
        self.vec.env_method("set_mix", 0.0)
        self.vec.env_method("set_forced_task", LAST_JUMP_STAGE)
        wins, games = [0], [0]

        def count(local, _globals):
            self.show("TESTING")
            if local["done"]:  # SB3 calls this once per game, with that game's values
                games[0] += 1
                info = local["info"]
                wins[0] += bool(info.get("success") and not info.get("deaths"))

        evaluate_policy(self.model, self.vec, n_eval_episodes=20, deterministic=True, callback=count, warn=False)
        self.vec.env_method("set_forced_task", None)
        self.vec.env_method("set_mix", OLD_STAGE_MIX)
        self.jump_test = wins[0] / max(games[0], 1)
        print(f"jump test at {self.num_timesteps}: {wins[0]}/{games[0]}", flush=True)

    def evaluate(self):
        """Tests the current stage without exploration noise, and moves on when it is good enough."""
        self.vec.env_method("set_mix", 0.0)
        wins, games, blocks, unused = [0], [0], [], []

        def count(local, _globals):
            self.show("TESTING")
            if local["done"]:  # SB3 calls this once per game, with that game's values
                info = local["info"]
                games[0] += 1
                if info.get("success") and not info.get("deaths"):  # a pass has to be without dying
                    wins[0] += 1
                    blocks.append(info["blocks_used"] / info["blocks_needed"])
                    unused.append(info["unused"])

        evaluate_policy(self.model, self.vec, n_eval_episodes=EVAL_EPISODES, deterministic=True, callback=count, warn=False)
        self.vec.env_method("set_mix", OLD_STAGE_MIX)
        self.eval_success = wins[0] / max(games[0], 1)
        self.eval_blocks = self.mean(blocks)
        self.eval_unused = self.mean(unused)
        block_ok = self.stage < BLOCK_GOAL_FROM_STAGE or (self.eval_blocks <= PASS_BLOCK_USE and self.eval_unused <= PASS_UNUSED)
        self.passes = self.passes + 1 if self.eval_success >= self.needed() and block_ok else 0
        print(f"test at {self.num_timesteps}: stage {self.stage} {wins[0]}/{games[0]} blocks {self.eval_blocks:.2f}x unused {self.eval_unused:.2f} passes {self.passes}/{PASS_IN_A_ROW}", flush=True)
        if self.passes >= PASS_IN_A_ROW:
            self.passes = 0
            self.model.save(self.out / f"stage{self.stage}")
            self.save_latest()
            if self.stage + 1 >= len(STAGES):
                print("all stages passed at", self.num_timesteps, flush=True)
                self.finished = True
            else:
                self.stage += 1
                self.vec.env_method("set_stage", self.stage)
                for d in (self.results, self.falls, self.stalls, self.damage, self.speeds, self.block_use, self.unused, self.shift, self.parts):
                    d.clear()
                print(f"--> stage {self.stage} {STAGES[self.stage]} at {self.num_timesteps} steps, {time.time() - self.t0:.0f} s", flush=True)
                self.eval_success = 0.0
        if self.stage > LAST_JUMP_STAGE and self.num_timesteps - self.last_skill_test >= SKILL_TEST_EVERY:
            self.last_skill_test = self.num_timesteps
            self.test_jump_skill()
        # The test reset the games, so training has to continue from fresh observations.
        self.model._last_obs = self.vec.reset()
        self.model._last_episode_starts = np.ones(self.vec.num_envs, dtype=bool)
        self.publish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ports", type=int, nargs="+", default=[25602, 25603])
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="skill8")
    p.add_argument("--resume", help="model to continue from")
    p.add_argument("--stage", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(2)
    Distribution.set_default_validate_args(False)  # the checks take a large part of each decision
    out = ROOT / "runs" / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    vec = SubprocVecEnv([make_env(port) for port in a.ports])
    if a.resume:
        model = PPO.load(a.resume, env=vec, device="cpu")
    else:
        model = PPO("MlpPolicy", vec, seed=a.seed, n_steps=2048, batch_size=512, n_epochs=8, gamma=0.99,
                    learning_rate=3e-4, ent_coef=0.01, device="cpu",
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
        # Start with rare jumps and sprints and a mostly steady view, so random inputs do not throw the
        # agent off the platform before it has learned anything. It can still learn to use them freely.
        bias = model.policy.action_net.bias.data
        for index, value in ((11, 1.5), (12, -1.5), (13, 1.5), (14, -1.5), (15 + 3, 1.5), (22 + 3, 1.5)):
            bias[index] = value
    vec.env_method("set_stage", a.stage)
    model.learn(a.steps, callback=Curriculum(vec, out, a.stage), reset_num_timesteps=not a.resume)
    model.save(out / "final")
