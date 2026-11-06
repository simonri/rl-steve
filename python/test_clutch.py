"""Every push can be saved, and makes sense.

Plays rounds with pushes on the simulator (AutoEnv unchanged on FakeClient), with inputs that wander about so it often
stands by an edge. For every push it checks, in the whole world and not just the cube the model sees, that the scripted
clutch lands it above the fall line, and that the push came while standing, over void, and away from the goal.

    python test_clutch.py [rounds]
"""
import math
import sys
from collections import Counter

import numpy as np

import curriculum
import skill_env
from auto_env import AutoEnv
from sim_env import FakeClient
from skill_env import Y

ORDER = ("forward", "back", "left", "right", "jump", "sneak", "sprint")


class Checked(AutoEnv):
    """AutoEnv that checks each push it gives."""
    checked = Counter()

    def _maybe_push(self, before, keys, clicked, turn_yaw, turn_pitch):
        given = self.pushes_given
        o = self.obs
        super()._maybe_push(before, keys, clicked, turn_yaw, turn_pitch)
        if self.pushes_given == given:
            return
        p = self.c.single.state()
        v = list(p["vel"])
        fall_line = min(Y + 1, Y + 1 + self.dy) - self.fall_margin
        assert o["on_ground"], "a push while in the air"
        gx, gz = self.goal_x + 0.5 - o["pos"][0], self.goal_z + 0.5 - o["pos"][2]
        h = math.hypot(v[0], v[2])
        assert (v[0] * gx + v[2] * gz) / (h * math.hypot(gx, gz)) <= 0.3 + 1e-9, "a push toward the goal"
        saved = self.c.single.can_save(v, [bool(before[k]) for k in ORDER], [bool(keys[k]) for k in ORDER], bool(clicked), turn_yaw, turn_pitch, fall_line)
        Checked.checked["saved" if saved else "NOT SAVED"] += 1
        Checked.checked[f"strength {h:.1f}"] += 1


def actions(rng, n):
    """Mostly forward with turns now and then, sometimes shift and clicks: it reaches edges and stands by them."""
    a = np.zeros((n, 8), np.int64)
    cur = np.array([1, 0, 0, 0, 0, 0, 5, 5])
    for t in range(n):
        if rng.random() < 0.08:
            cur = np.array([rng.choice([0, 1, 1, 2]), rng.integers(3), rng.random() < 0.3, 0, rng.random() < 0.2, rng.random() < 0.3, rng.integers(11), rng.integers(11)])
        a[t] = cur
        a[t, 3] = rng.random() < 0.3
        if rng.random() < 0.6:
            a[t, 6] = a[t, 7] = 5
    return a


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    skill_env.McClient = FakeClient
    rng = np.random.default_rng(5)
    env = Checked(0)
    families = ["bridge", "stairs", "treenan", "bedwars"]
    for n in range(rounds):
        levels = curriculum.Curriculum().levels
        family = families[n % len(families)]
        for d, values in curriculum.FAMILIES[family].items():
            levels[family][d] = int(rng.integers(len(values)))
        levels[family]["push"] = int(rng.integers(1, len(curriculum.PUSHES)))
        env.set_levels(levels)
        env.set_exam(family)
        env.reset(seed=2000 + n)
        for a in actions(rng, 4000):
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    c = Checked.checked
    print(f"{rounds} rounds: {dict(sorted(c.items()))}")
    assert c["saved"] > 0, "no pushes at all"
    assert c["NOT SAVED"] == 0, "a push that the clutch cannot save in the real world"
    print("clutch ok")


if __name__ == "__main__":
    main()
