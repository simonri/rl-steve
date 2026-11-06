"""Training in the Rust simulator: the same courses, rules and inputs as AutoEnv, without the game.

CourseMaker draws rounds with AutoEnv's own generators. SimVecEnv hands them to mcsim.Batch, which plays every round
side by side on all cores, and gives Stable-Baselines3 a normal VecEnv. FakeClient lets the unchanged AutoEnv run on
the simulator, which is how test_sim_env.py checks that the Rust rules give exactly the same numbers.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import curriculum
import mcsim
from auto_env import MAP_RADIUS, RADIUS, TURN_STEPS, AutoEnv
from skill_env import FIRST_CLEAR, NO_BLOCKS, Y

KINDS = {}


def kind(state):
    if state not in KINDS:
        KINDS[state] = mcsim.kind_of(state)
    return KINDS[state]


class CourseMaker(AutoEnv):
    """AutoEnv's course drawing, with no game behind it."""

    def __init__(self, levels=None, seed=None):
        self.levels = levels or curriculum.Curriculum().levels
        self.mix = 0.0
        self.dirty = list(FIRST_CLEAR)
        self.camera_every = 0.0
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (AutoEnv.N_OBS,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2, 2, len(TURN_STEPS), len(TURN_STEPS)])
        self._np_random, _ = seeding.np_random(seed)

    def draw(self, seed=None):
        """One round: what Batch.set_course needs, and what the round's info should say."""
        if seed is not None:
            gym.Env.reset(self, seed=seed)
        for attempt in range(8):
            self._use_task(self._pick_stage())
            try:
                spec = self._spec()
                break
            except RuntimeError as e:  # a generator found nothing that fits: another round instead
                if attempt == 7:
                    raise
                print(f"round redrawn: {e}", flush=True)
        self.limit = self.time_limit()
        self._plan_pushes()
        boxes = [list(map(int, b)) for b in spec["platforms"]]
        placed = dict(self.next_cells) if self.extras == "treenan" else {}
        placed.update(self.shape_cells)  # slabs and other shapes, put in after the course is built
        cells = np.array([(x, y, z, kind(s)) for (x, y, z), s in placed.items() if s != "minecraft:air"], np.int32).reshape(-1, 4) if placed else None
        points = [b[:3] for b in boxes] + [b[3:] for b in boxes] + ([] if cells is None or not len(cells) else [cells[:, :3].min(0).tolist(), cells[:, :3].max(0).tolist()])
        points += [[0, Y, 0], [self.goal_x, Y + self.dy, self.goal_z]]
        lo = [min(p[i] for p in points) - 8 for i in range(3)]
        hi = [max(p[i] for p in points) + 8 for i in range(3)]
        field = None
        if self.field is not None:
            nx, nz = self.field.shape
            blocked = np.zeros((nx, nz), bool)
            for x, z in self.blocked:
                i, j = x - self.fx0, z - self.fz0
                if 0 <= i < nx and 0 <= j < nz:
                    blocked[i, j] = True
            field = dict(x0=int(self.fx0), z0=int(self.fz0), dist=np.ascontiguousarray(self.field, np.float64), blocked=blocked)
        return dict(
            hop=self.family == "hop", goal=[int(self.goal_x), int(self.goal_z)], dy=int(self.dy), goal_half=int(self.goal_half), fall_margin=float(self.fall_margin),
            landing=[list(map(int, c)) for c in self.landing_order], par=int(self.par), limit=int(self.limit), no_blocks=self.extras in NO_BLOCKS, field=field,
            home=[float(v) for v in spec["pos"]], yaw=float(spec["yaw"]), pitch=float(spec["pitch"]), blocks=int(spec["blocks"]), boxes=boxes, cells=cells, lo=lo, hi=hi,
            settle=bool(self.shape_cells),  # the game takes one tick to show blocks put in after the reset
            pushes=[(int(a), int(b), int(c), float(d), float(e)) for a, b, c, d, e in self.pushes],
        ), dict(family=self.family, tested=self.tested, at=[self.levels[self.family][d] for d in curriculum.FAMILIES[self.family]])


class SimVecEnv(VecEnv):
    """Many simulated rounds as one Stable-Baselines3 VecEnv. A finished round is replaced by a new one at once."""

    def __init__(self, n, levels=None, seed=0):
        self.makers = [CourseMaker(levels, seed * 100_003 + i) for i in range(n)]
        self.batch = mcsim.Batch(n)
        maker = self.makers[0]
        super().__init__(n, maker.observation_space, maker.action_space)
        self.obs = np.zeros((n, mcsim.N_OBS), np.float32)
        self.rewards = np.zeros(n, np.float32)
        self.dones = np.zeros(n, bool)
        self.returns = np.zeros(n, np.float64)
        self.meta = [None] * n
        self.actions = None

    def _new(self, i):
        spec, meta = self.makers[i].draw()
        self.batch.set_course(i, spec)
        self.batch.observe(i, self.obs[i])
        self.meta[i] = meta
        self.returns[i] = 0.0

    def reset(self):
        for i in range(self.num_envs):
            self._new(i)
        return self.obs.copy()

    def step_async(self, actions):
        self.actions = np.ascontiguousarray(actions, np.int64).reshape(self.num_envs, 8)

    def step_wait(self):
        self.batch.step(self.actions, self.obs, self.rewards, self.dones)
        self.returns += self.rewards
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(self.dones):
            info = self.batch.report(int(i))
            info.update(self.meta[i])
            info["terminal_observation"] = self.obs[i].copy()
            info["episode"] = {"r": float(self.returns[i]), "l": info["ticks"]}
            infos[i] = info
            self._new(int(i))
        return self.obs.copy(), self.rewards.copy(), self.dones.copy(), infos

    # ---- What training asks of its games ----

    def _targets(self, indices):
        return range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else indices)

    def env_method(self, name, *args, indices=None, **kwargs):
        if name == "sidebar":  # nothing to show without a game
            return [None for _ in self._targets(indices)]
        return [getattr(self.makers[i], name)(*args, **kwargs) for i in self._targets(indices)]

    def get_attr(self, name, indices=None):
        return [getattr(self.makers[i], name) for i in self._targets(indices)]

    def set_attr(self, name, value, indices=None):
        for i in self._targets(indices):
            setattr(self.makers[i], name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._targets(indices)]

    def close(self):
        pass

    def seed(self, seed=None):
        return [None for _ in range(self.num_envs)]


class FakeClient:
    """McClient's interface on top of the simulator, so the Python AutoEnv can run unchanged on it."""

    def __init__(self, port=0):
        self.single = None
        self.pasted = {}
        self.block_radius, self.map_radius = RADIUS, MAP_RADIUS
        self.keys = [False] * 7

    def render(self, on):
        return {}

    def close(self):
        pass

    def call(self, op=None, **msg):
        if op == "push":
            self.single.push([float(v) for v in msg["vel"]])
            return {"ok": True}
        if op == "paste":
            for x, y, z, i in msg["blocks"]:
                if msg.get("now"):  # into the round being played
                    self.single.set_block(x, y, z, msg["palette"][i])
                else:  # for the next round's world
                    self.pasted[(x, y, z)] = msg["palette"][i]
        return {"ok": True}

    def _obs(self, s, placed=None):
        o = dict(s)
        o["sneaking"] = bool(self.keys[5])
        o["feet"] = [math.floor(v) for v in o["pos"]]
        o["solid"] = self.single.solid_around(self.block_radius)
        o["surface"] = self.single.surface_around(self.map_radius)
        o["clicks"] = [[*placed[0], *placed[1]]] if placed else []
        o["vel"] = list(o["vel"])
        o["pos"] = list(o["pos"])
        return o

    def reset(self, **spec):
        self.block_radius = spec.get("block_radius", self.block_radius)
        self.map_radius = spec.get("map_radius", self.map_radius)
        pos = spec["pos"]
        if spec.get("keep_arena"):
            for x in range(math.floor(pos[0] - 0.31), math.floor(pos[0] + 0.31) + 1):
                for z in range(math.floor(pos[2] - 0.31), math.floor(pos[2] + 0.31) + 1):
                    for y in range(math.floor(pos[1]), math.floor(pos[1] + 1.81) + 1):
                        self.single.set_block(x, y, z, "minecraft:air")
        else:
            boxes = [list(map(int, b)) for b in spec.get("platforms", [])]
            cells = [(x, y, z, s) for (x, y, z), s in self.pasted.items() if s != "minecraft:air"]
            self.pasted = {}
            points = [b[:3] for b in boxes] + [b[3:] for b in boxes] + [c[:3] for c in cells] + [[0, Y, 0]]
            lo = [min(p[i] for p in points) - 16 for i in range(3)]
            hi = [max(p[i] for p in points) + 16 for i in range(3)]
            self.single = mcsim.Single(boxes, cells, lo, hi)
        self.keys = [False] * 7
        self.single.spawn(list(pos), float(spec["yaw"]), float(spec["pitch"]), int(spec.get("blocks", 0)))
        return self._obs(self.single.state())

    def step(self, ticks=1, **a):
        self.keys = [bool(a.get(k, False)) for k in ("forward", "back", "left", "right", "jump", "sneak", "sprint")]
        s = None
        for _ in range(ticks):
            s = self.single.tick(self.keys, bool(a.get("use", False)), float(a.get("yaw", 0.0)), float(a.get("pitch", 0.0)))
        return self._obs(s, s.get("placed"))
