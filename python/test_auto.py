"""Checks the generator, the dials, the view and the point rules with a fake game. No game needed."""
import math

import numpy as np

import curriculum
from auto_env import DEATH_COST, HEAD, KNOWN, LOOK_UP_COST, NO_TURN, PLACE_BONUS, RADIUS, VIEW_SIZE, AutoEnv
from curriculum import FAMILIES, Curriculum, draw
from jump_physics import hits_pole, jump_possible
from skill_env import GOAL_REWARD, HISTORY, LANDING_BONUS, MAX_DEATHS, Y

AutoEnv.reward_scale = 1.0
SIZE = 2 * RADIUS + 1


def solid_at(blocks, feet):
    chars = []
    for x in range(-RADIUS, RADIUS + 1):
        for y in range(-RADIUS, RADIUS + 1):
            for z in range(-RADIUS, RADIUS + 1):
                chars.append("1" if (feet[0] + x, feet[1] + y, feet[2] + z) in blocks else "0")
    return "".join(chars)


def state(x=0.5, y=Y + 1.0, z=0.5, yaw=0.0, ground=True, vel=(0, 0, 0), blocks=frozenset(), pitch=0.0):
    feet = [math.floor(x), math.floor(y), math.floor(z)]
    return dict(pos=[x, y, z], vel=list(vel), yaw=yaw, pitch=pitch, on_ground=ground, held_count=0, health=20.0, solid=solid_at(blocks, feet), feet=feet)


class Fake:
    def __init__(self):
        self.path, self.steps, self.last = [], [], state()

    def step(self, ticks=1, **a):
        self.steps.append(ticks)
        self.last = self.path.pop(0) if self.path else self.last
        return self.last

    def reset(self, **spec):
        self.spec = spec
        self.last = state(*spec["pos"][:1], spec["pos"][1], spec["pos"][2], spec.get("yaw", 0.0))
        return self.last

    def call(self, **msg):
        return {}


def make(family="hop", **settings):
    env = AutoEnv.__new__(AutoEnv)
    env.c, env.mix, env.dirty = Fake(), 0.0, [0] * 6
    env.levels = Curriculum().levels
    env._np_random = np.random.default_rng(settings.pop("seed", 0))
    base = {d: v[0] for d, v in FAMILIES[family].items()}
    env._use_task((family, {**base, **settings}, None))
    return env


def begin(env):
    spec = env._spec()
    env.home = dict(pos=spec["pos"], yaw=spec["yaw"], pitch=spec["pitch"])
    env.obs = env.c.reset(**spec)
    env.limit = env.time_limit()
    env._begin_episode()
    return spec


def tick(env):
    return env._tick(0, 0, 0, False, 0, 0, NO_TURN, NO_TURN)


# Every hop setting is possible by the math, from a block with a short run-up, and every pole is clear of the way.
for offset in FAMILIES["hop"]["offset"]:
    assert jump_possible(*offset, 0, run_up_ticks=2), offset
for seed in range(150):
    rng = np.random.default_rng(seed)
    t = {d: v[int(rng.integers(len(v)))] for d, v in FAMILIES["hop"].items()}
    env = make("hop", seed=seed, **t)
    spec = begin(env)
    route = [(0.5, 0.5)] + [(b[0] + 0.5, b[2] + 0.5) for b in env.course]
    assert len(env.course) <= t["count"] * t["longer"] and env.par == len(env.course) and env.blocks_given == 0
    for a, b in zip(env.course, env.course[1:]):
        assert jump_possible(abs(a[0] - b[0]), abs(a[2] - b[2]), b[1] - a[1], run_up_ticks=2), (a, b)
    for p, q in zip(route, route[1:]):
        assert not hits_pole(p, q, [(x, z) for x, _, z in env.poles])
    assert not {(x, z) for x, _, z in env.poles} & {(b[0], b[2]) for b in env.course}
    if t["offset"] == (2, 2) and t["count"] >= 3 and not t["mix"] and not t["bend"]:
        assert env.poles, "the zigzag keeps its poles"
    assert all(env.dirty[0] <= b[0] <= env.dirty[3] and env.dirty[2] <= b[2] <= env.dirty[5] for b in env.course)

# Harder settings stay possible, also when mixed and turning, and never touch themselves.
for seed in range(150):
    rng = np.random.default_rng(1000 + seed)
    t = {d: v[-1] for d, v in FAMILIES["hop"].items()}
    t["height"] = FAMILIES["hop"]["height"][int(rng.integers(4))]
    env = make("hop", seed=seed, **t)
    begin(env)
    assert t["count"] <= len(env.course) <= t["count"] * t["longer"], (len(env.course), t)
    start = (0, Y, 0) if t["takeoff"] == "block" else None
    for a, b in zip(env.course, env.course[1:]):
        assert jump_possible(abs(a[0] - b[0]) if abs(a[0] - b[0]) >= abs(a[2] - b[2]) else abs(a[2] - b[2]), min(abs(a[0] - b[0]), abs(a[2] - b[2])), b[1] - a[1], run_up_ticks=2), (a, b)
    assert len({(b[0], b[2]) for b in env.course}) == len(env.course)

# Paths in every shape are connected, cannot be cut short, and end on the goal platform.
for seed in range(300):
    rng = np.random.default_rng(seed)
    t = {d: v[int(rng.integers(len(v)))] for d, v in FAMILIES["path"].items()}
    env = make("path", seed=seed, **t)
    spec = begin(env)
    cells = env.course
    assert len(cells) >= t["length"] * t["longer"], (len(cells), t)
    assert (cells[0][0], cells[0][2]) in {(2, 0), (0, 2), (-2, 0), (0, -2)}
    for a, b in zip(cells, cells[1:]):
        across = max(abs(a[0] - b[0]), abs(a[2] - b[2]))
        assert across <= 1 + (3 if t["wide_holes"] else 2) + 1 and b[1] - a[1] <= 1, (a, b, t)
        if across > 1:
            assert a[1] == b[1], "holes only on the level"
    for i, a in enumerate(cells):
        for b in cells[i + 3 :]:
            assert max(abs(a[0] - b[0]), abs(a[2] - b[2])) > 1, ("touches itself", a, b, t)
    last = env.landing_order[-1]
    assert max(abs(last[0] - env.goal_x), abs(last[2] - env.goal_z)) <= 2 and last[1] == Y + env.dy + 1
# The wider kinds really differ: paths turn back and change height.
env = make("path", seed=1, length=20, wind="around", slope="stairs and drops")
begin(env)
assert len({c[1] for c in env.course}) > 2
# Harder dials wait until the rest of the course kind is at its limit.
assert "far" not in curriculum.open_dials(Curriculum().levels, "hop")
full = Curriculum({"hop": {d: len(v) - 1 for d, v in FAMILIES["hop"].items() if d not in curriculum.EXPERT["hop"]}}).levels
assert "far" in curriculum.open_dials(full, "hop")

# Chasing time: the faster levels leave almost no spare time, and they wait for the rest of the bridge kind.
env = make("bridge", gap=20, seed=11)
begin(env)
relaxed = env.time_limit()
env = make("bridge", gap=20, faster=3, seed=11)
begin(env)
assert env.ticks_per_block == 3 and env.time_limit() < relaxed / 2 and env.time_limit() == 20 + 3 * (env.par + 1 + max(0, -env.dy))
assert "faster" not in curriculum.open_dials(Curriculum().levels, "bridge")

# Dials: rounds at a limit raise it, failing lowers it, and it never leaves its range.
c = Curriculum()
for _ in range(29):
    assert c.record("hop", "offset", True) is None
assert c.record("hop", "offset", True) is None, "nothing goes up before rounds with everything at its limit have held"
for _ in range(10):
    c.record("hop", "all", True)
assert c.record("hop", "offset", True) == ("offset", 1) and c.levels["hop"]["offset"] == 1
for _ in range(30):
    assert c.record("hop", "offset", False) is None, "one bad window can be bad luck"
for _ in range(30):
    moved = c.record("hop", "offset", False)
assert moved == ("offset", 0)
for _ in range(100):
    assert c.record("hop", "offset", False) is None and c.levels["hop"]["offset"] == 0
# All dials together failing holds every dial back.
c = Curriculum()
for _ in range(10):
    c.record("hop", "all", False)
for _ in range(60):
    c.record("hop", "count", True)
assert c.levels["hop"]["count"] == 0
# A finished kind keeps being tested: a dial can come back down, and it is not mastered until it really wins.
top = Curriculum({f: {d: len(v) - 1 for d, v in dials.items()} for f, dials in FAMILIES.items()})
moved = None
rng2 = np.random.default_rng(1)
tests = {draw(top.levels, "path", rng2)[1] for _ in range(300)}
assert set(FAMILIES["path"]) <= tests
assert top.mastered() == []
for _ in range(30):
    top.record("path", "all", True)
assert top.mastered() == ["path"]
for _ in range(60):
    moved = top.record("path", "longer", False) or moved
assert moved == ("longer", 0) and "path" not in top.mastered()
assert sum(curriculum.pick_family(Curriculum().levels, rng2, ["hop"]) == "hop" for _ in range(600)) < 150
# The map: height and blocks from the game, turned with the round's frame, and what it has walked on is remembered.
from auto_env import MAP_CHANNELS, MAP_RADIUS, MAP_SIDE, MAP_START
env = make(count=2, takeoff="block", seed=16)
begin(env)
env.goal_x, env.goal_z, env.dy = 12, 0, 0
texts = ["."] * MAP_SIDE ** 2
texts[(MAP_RADIUS + 16) * MAP_SIDE + MAP_RADIUS] = "M"  # 16 blocks away along x, 4 under the feet
o = {**state(), "surface": "".join(texts)}
layers = env._vec(o)[MAP_START : MAP_START + MAP_CHANNELS * MAP_SIDE ** 2].reshape(MAP_CHANNELS, MAP_SIDE, MAP_SIDE)
assert layers[1].sum() == 1.0 and layers[0].sum() == -4 / 8, (layers[1].sum(), layers[0].sum())
env.goal_x, env.goal_z = 0, 12
turned = env._vec(o)[MAP_START : MAP_START + MAP_CHANNELS * MAP_SIDE ** 2].reshape(MAP_CHANNELS, MAP_SIDE, MAP_SIDE)
assert turned[1].sum() == 1.0 and not np.array_equal(turned[1], layers[1]), "the map turns with the frame"
env.visited = set()
env._visit(state(x=3.5))
been = env._vec(o)[MAP_START : MAP_START + MAP_CHANNELS * MAP_SIDE ** 2].reshape(MAP_CHANNELS, MAP_SIDE, MAP_SIDE)[2]
assert been.sum() == 1.0
# Older recordings get their map from the blocks they have.
assert env._vec(state(blocks={(1, Y, 0)}))[MAP_START : MAP_START + MAP_SIDE ** 2].sum() != 0
# It is seen by the network, and a whole batch goes through it.
import gymnasium as gym, torch
from spatial import SpatialExtractor
net = SpatialExtractor(gym.spaces.Box(-5.0, 5.0, (AutoEnv.N_OBS,), np.float32))
assert net(torch.zeros(3, AutoEnv.N_OBS)).shape == (3, 384)
# The newer eyes see a single block: moving one block in the cube changes what comes out.
from spatial import MemoryPolicy, SpatialNet
from auto_env import CUBE_START
space = gym.spaces.Box(-5.0, 5.0, (AutoEnv.N_OBS,), np.float32)
net = SpatialNet(space)
a, b = torch.zeros(2, AutoEnv.N_OBS), torch.zeros(2, AutoEnv.N_OBS)
a[:, CUBE_START + 600], b[:, CUBE_START + 601] = 1.0, 1.0
with torch.no_grad():
    assert net(a).shape == (2, 256) and not torch.allclose(net(a), net(b))
# The memory runs a whole stretch in one go, and gives exactly what stepping tick by tick gives, rounds starting
# only at the first tick of a stretch as the rollout buffer cuts them.
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
lstm = torch.nn.LSTM(8, 16)
feats, state0 = torch.randn(3 * 5, 8), (torch.randn(1, 3, 16), torch.randn(1, 3, 16))
starts = torch.zeros(3, 5)
starts[1, 0] = 1.0
with torch.no_grad():
    fast, (h1, _) = MemoryPolicy._process_sequence(feats, state0, starts.flatten(), lstm)
    starts[0, 2] = 1e-9  # forces the library onto its tick by tick path with the same numbers
    slow, (h2, _) = RecurrentActorCriticPolicy._process_sequence(feats, state0, starts.flatten(), lstm)
assert torch.allclose(fast, slow, atol=1e-5) and torch.allclose(h1, h2, atol=1e-5)
# It knows when a click would put a block down: a free spot in reach, with a block in hand, and not inside itself.
hit = lambda face, block: {**state(), "held_count": 5, "hit": dict(type="block", block=list(block), face=face, loc=[0, 0, 0])}
assert AutoEnv._can_place(hit("east", (1, Y, 0)))
assert not AutoEnv._can_place({**hit("east", (1, Y, 0)), "held_count": 0}), "no blocks in hand"
assert not AutoEnv._can_place(hit("up", (0, Y, 0))), "it stands right there"
assert not AutoEnv._can_place({**hit("east", (1, Y, 0)), "solid": solid_at({(2, Y, 0)}, [0, Y + 1, 0])}), "the spot is taken"
assert not AutoEnv._can_place({**state(), "held_count": 5}), "aiming at nothing"
# Giving up costs as much as falling.
from auto_env import STUCK_COST
assert STUCK_COST == DEATH_COST
# A dial that just went down needs two good windows before it goes up again, so it does not swing back and forth.
c = Curriculum()
for _ in range(10):
    c.record("hop", "all", True)
for _ in range(30):
    c.record("hop", "offset", True)
assert c.levels["hop"]["offset"] == 1
for _ in range(60):
    c.record("hop", "offset", False)
assert c.levels["hop"]["offset"] == 0
for _ in range(10):
    c.record("hop", "all", True)
for _ in range(30):
    c.record("hop", "offset", True)
assert c.levels["hop"]["offset"] == 0, "first good window after going down is not enough"
for _ in range(10):
    c.record("hop", "all", True)
for _ in range(30):
    c.record("hop", "offset", True)
assert c.levels["hop"]["offset"] == 1
# Close to the top, most rounds put every dial at its limit at once.
near = Curriculum({"hop": {d: len(v) - 1 for d, v in FAMILIES["hop"].items() if d not in curriculum.EXPERT["hop"]}})
rng3 = np.random.default_rng(2)
drawn = [draw(near.levels, "hop", rng3)[1] for _ in range(1000)]
assert 520 < drawn.count("all") < 680, drawn.count("all")
drawn = [draw(Curriculum().levels, "hop", rng3)[1] for _ in range(1000)]
assert 220 < drawn.count("all") < 380, drawn.count("all")
# Levels survive a save, and a round with a dial at its limit plays exactly that level.
c = Curriculum({"hop": {"offset": 4, "count": 2, "nonsense": 1}, "gone": {"x": 1}})
assert c.levels["hop"]["offset"] == 4 and c.total("hop") == 6
rng = np.random.default_rng(0)
seen = set()
for _ in range(400):
    s, tested = draw(c.levels, "hop", rng)
    seen.add(s["offset"])
    assert s["offset"] in FAMILIES["hop"]["offset"][:5] and s["count"] in FAMILIES["hop"]["count"][:3]
    if tested == "all" or tested == "offset":
        assert s["offset"] == (2, 2)
assert len(seen) == 5, "lower levels are still played"
# Closed until the bridge has come far enough, and obstacles wait for gaps that fit them.
assert not curriculum.is_open(Curriculum().levels, "treenan")
assert "obstacle" not in curriculum.open_dials(Curriculum().levels, "bridge")
assert all(curriculum.pick_family(Curriculum().levels, rng) != "treenan" for _ in range(100))
assert curriculum.is_open(Curriculum({"bridge": {"gap": 5, "directions": 2}}).levels, "treenan")

# The view: turning the whole world and the player by 90 degrees changes nothing the model sees.
world = {(1, Y, 0), (3, Y, 2), (0, Y, 0), (-1, Y + 1, 3)}
turned = {(-z, y, x) for x, y, z in world}  # yaw grows by 90: x, z becomes -z, x
for yaw in (0.0, 13.0, 45.0, 100.0):
    env = make()
    env.goal_x, env.goal_z, env.dy, env.par, env.used = 3, 2, 0, 1, 0
    env.landed = 0
    env.landing_order = [(3, Y + 1, 2)]
    env.course_axis = (1, 0)
    a = env._vec(state(0.5, Y + 1.0, 0.5, yaw, blocks=world))
    env.goal_x, env.goal_z = -2, 3
    env.landing_order = [(-2, Y + 1, 3)]
    b = env._vec(state(0.5, Y + 1.0, 0.5, yaw + 90.0, blocks=turned))
    head_diff = np.abs(a[:HEAD] - b[:HEAD])
    assert head_diff.max() < 0.05, (yaw, np.argmax(head_diff), head_diff.max())
    assert np.array_equal(a[HEAD + KNOWN :], b[HEAD + KNOWN :]), yaw
# And there is no jump at 45 degrees: a tiny turn barely changes the values.
env = make()
env.goal_x, env.goal_z, env.dy, env.par, env.used, env.landed = 3, 2, 0, 1, 0, 0
env.landing_order = [(3, Y + 1, 2)]
a, b = env._vec(state(yaw=44.0, blocks=world)), env._vec(state(yaw=46.0, blocks=world))
assert np.abs(a[:HEAD] - b[:HEAD]).max() < 0.15, np.abs(a[:HEAD] - b[:HEAD]).max()
assert a.shape == (AutoEnv.N_OBS - 8 * HISTORY,) and VIEW_SIZE == 11 ** 3
# The view reaches 5 blocks all around, also behind and below.
for block in ((-5, 0, 0), (5, 0, 0), (0, 0, 5), (0, 0, -5), (0, -5, 0), (0, 5, 0)):
    x, y, z = block
    env.landing_order = [(3, Y + 1, 2)]
    seen = env._vec(state(yaw=0.0, blocks={(x, Y + 1 + y, z)}))[HEAD + KNOWN : HEAD + KNOWN + VIEW_SIZE]
    assert seen.sum() == 1.0, block
# Real sprinting and health are seen as they are in the game, and held keys after the limits are known directly.
base_seen = env._vec(state())
assert (env._vec({**state(), "sprinting": True})[:HEAD] != base_seen[:HEAD]).sum() == 1
assert (env._vec({**state(), "health": 10.0})[:HEAD] != base_seen[:HEAD]).sum() == 1
env.t, env.limit, env.since_best, env.deaths, env.sneak_held = 0, 100, 0, 0, 0
env.keys = {k: False for k in ("forward", "back", "left", "right", "sneak", "jump", "sprint")}
env.keys["forward"] = env.keys["left"] = True
known = env._with_timer(np.zeros(AutoEnv.N_OBS, np.float32))[HEAD : HEAD + KNOWN]
assert list(known[2:]) == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], known
# The new body values and the side aimed at show up.
assert (env._vec({**state(), "collision": True})[:HEAD] != base_seen[:HEAD]).sum() == 1
assert (env._vec({**state(), "fall_distance": 3.0})[:HEAD] != base_seen[:HEAD]).sum() == 1
assert env._side(dict(type="block", face="up"), 0)[0] == 1.0 and env._side({}, 0) == [0.0] * 6
assert sum(env._side(dict(type="block", face="north"), 1)) == 1.0
# The wait between jumps only counts while the jump key stays down.
env.jump_wait, env.was_on_ground = 0, True
env._track_jump(dict(on_ground=False, vel=[0, 0.42, 0]), True)
assert env.jump_wait == 10
env._track_jump(dict(on_ground=False, vel=[0, 0.3, 0]), True)
assert env.jump_wait == 9
env._track_jump(dict(on_ground=True, vel=[0, 0, 0]), False)
assert env.jump_wait == 0
# Turning the head changes nothing in the block view, and blocks stay exact on the grid.
env.goal_x, env.goal_z, env.dy = 12, 0, 0
one = env._vec(state(yaw=10.0, blocks=world))[HEAD + KNOWN :]
assert np.array_equal(one, env._vec(state(yaw=200.0, blocks=world))[HEAD + KNOWN :])
# The frame follows the goal, so the same course in another direction looks exactly the same.
env.goal_x, env.goal_z = 0, 12
turned_world = {(-z, y, x) for x, y, z in world}
two = env._vec(state(yaw=100.0, blocks=turned_world))[HEAD + KNOWN :]
assert np.array_equal(one, two)

# Points: each block once and in order, the last one wins the round, and a win after a death pays less.
env = make(count=2, takeoff="block", seed=1)
begin(env)
first, last = env.landing_order
on = lambda c, **k: state(c[0] + 0.5, float(c[1]), c[2] + 0.5, **k)
env.c.path = [on(first), on(first), on(last)]
r1 = tick(env)[0]
assert env.landed == 1 and r1 > LANDING_BONUS * 0.9
assert tick(env)[3]["success"] is False and env.landed == 1
r3, term, _, info = tick(env)
assert term and info["clean"] and env.landed == 2 and r3 > GOAL_REWARD
env = make(count=2, takeoff="block", seed=1)
begin(env)
env.c.path = [on(env.landing_order[1])]
tick(env)
assert env.landed == 0, "skipping a block gives nothing"
# Shift is never let go by itself.
env = make(count=2, takeoff="block", seed=15)
begin(env)
for _ in range(100):
    assert env._press(dict(forward=False, back=False, left=False, right=False, sneak=True, jump=False, sprint=False))["sneak"] or env.key_age["sneak"] < 2
assert env.keys["sneak"]
# It sees the game as it is right now, not a tick late.
env = make(count=2, takeoff="block", seed=14)
begin(env)
env.delayed.append(env._vec(env.obs))
moved = state(x=0.9)
env.c.path = [moved]
seen, *_ = env.step([0, 0, 0, 0, 0, 0, NO_TURN, NO_TURN])
assert np.array_equal(seen[:HEAD], env._vec(moved)[:HEAD]), "no delay"
# Dying costs no game time in training, and the round ends on the last life.
env = make(count=1, takeoff="block", seed=3)
begin(env)
env.c.path = [state(y=Y - 5.0, ground=False)]
t_before = env.t
r, term, _, info = tick(env)
assert abs(env.parts["death"] + DEATH_COST) < 1e-9 and DEATH_COST == 4.0
assert not term and info["respawned"] and env.deaths == 1 and all(t == 1 for t in env.c.steps) and env.t == t_before + 1
env.c.path = [state(y=Y - 5.0, ground=False)]
r, term, _, info = tick(env)
assert term and env.deaths == MAX_DEATHS and not info["clean"]
# Looking above level at nothing costs, in proportion to how far up. Level or down is free, and so is looking up at a block.
env = make(count=2, takeoff="block", seed=4)
begin(env)
env.c.path = [state(pitch=-90.0), state(pitch=-45.0), state(pitch=0.0), state(pitch=60.0)]
for expected in (LOOK_UP_COST, LOOK_UP_COST / 2):
    before = env.parts.get("looking up", 0.0)
    tick(env)
    assert abs(env.parts["looking up"] - before + expected) < 1e-9
before = env.parts["looking up"]
tick(env), tick(env)
assert env.parts["looking up"] == before
env = make("bridge", gap=8, obstacle="wall", seed=6)
begin(env)
env.c.path = [{**state(pitch=-60.0), "hit": dict(type="block", block=[3, Y + 2, 0], face="west", loc=[3, Y + 2.5, 0.5])}]
tick(env)
assert "looking up" not in env.parts, "looking up at a block is fine"
# A block placed closer to the goal pays once, and neither a new block at the same spot nor one further back does.
env = make("bridge", gap=8, seed=10)
begin(env)
env.goal_x, env.goal_z = 12, 0
env.closest_block = 11.0
placed = lambda x, held: {**state(), "held_count": held, "clicks": [[x, Y, 0, x - 1, Y, 0]]}
env.obs = {**state(), "held_count": 10}
env.c.path = [placed(2, 9), placed(2, 8), placed(1, 7), placed(3, 6)]
tick(env)
assert abs(env.parts["placed toward the goal"] - PLACE_BONUS) < 1e-9
tick(env), tick(env)
assert abs(env.parts["placed toward the goal"] - PLACE_BONUS) < 1e-9
tick(env)
assert abs(env.parts["placed toward the goal"] - 2 * PLACE_BONUS) < 1e-9
# One decision is one tick, and a click still only happens once per tick.
env = make(count=2, takeoff="block", seed=12)
begin(env)
env.delayed.append(np.zeros(AutoEnv.N_OBS - 8 * HISTORY, np.float32))
ticks = len(env.c.steps)
env.step([0, 0, 0, 1, 0, 0, NO_TURN, NO_TURN])
assert len(env.c.steps) == ticks + 1
print("auto ok")
# Learning batches padded on the card hold exactly what the library's own padding gives, cut at the same places.
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import train_auto
spaces_ = (gym.spaces.Box(-1, 1, (5,), np.float32), gym.spaces.MultiDiscrete([3, 2]))
mine, theirs = (cls(64, *spaces_, (64, 1, 2, 3), "cpu", n_envs=2) for cls in (train_auto.ShortStretches, RecurrentRolloutBuffer))
rng = np.random.default_rng(0)
for t in range(64):
    row = (rng.normal(size=(2, 5)).astype(np.float32), rng.integers(0, 2, (2, 2)), rng.normal(size=2).astype(np.float32), rng.random(2) < 0.1,
           torch.randn(2), torch.randn(2))
    states = RNNStates(*(tuple(torch.randn(1, 2, 3) for _ in range(2)) for _ in range(2)))
    for b in (mine, theirs):
        b.add(*row, lstm_states=states)
for b in (mine, theirs):
    b.compute_returns_and_advantage(torch.zeros(2), np.zeros(2))
for b in (mine, theirs):
    list(b.get(40))  # lays the stretch out round by round, as learning does
inds = np.arange(128)[17:57]
change = np.zeros((128, 1))  # shaped as the library lays it out
change[[0, 64]] = 1.0
cut = change.copy()
cut[inds[:: train_auto.STRETCH]] = 1.0
a, b = mine._get_samples(inds, change), theirs._get_samples(inds, cut)
for name in a._fields:
    x, y = getattr(a, name), getattr(b, name)
    if name == "lstm_states":
        assert all(torch.allclose(p, q) for p, q in zip([*x.pi, *x.vf], [*y.pi, *y.vf])), name
    else:  # the padding of what it sees is marked as padding, where the library leaves zeros
        assert x.shape == y.shape and torch.allclose(torch.nan_to_num(x.float()), y.float()), name
assert torch.isnan(a.observations[:, 0]).sum() == (a.mask == 0).sum()
# The eyes skip marked padding and give it nothing, and see the played ticks as they would alone.
policy = MemoryPolicy(space, gym.spaces.MultiDiscrete([3, 3, 2, 2, 2, 2, 11, 11]), lambda _: 1e-3, features_extractor_class=SpatialNet, lstm_hidden_size=32)
played = torch.rand(3, AutoEnv.N_OBS)
padded = torch.cat([played, torch.full((2, AutoEnv.N_OBS), float("nan"))])
with torch.no_grad():
    both, alone = policy.extract_features(padded), policy.extract_features(played)
assert torch.allclose(both[:3], alone, atol=1e-6) and (both[3:] == 0).all()
# Copying recordings of different lengths gives a finite loss: their padding goes through the whole model.
rounds = [(np.random.rand(n, AutoEnv.N_OBS).astype(np.float32), np.zeros((n, 8), np.int64)) for n in (3, 7)]
distribution, actions, real = train_auto._whole_rounds(policy, rounds, "cpu")
assert torch.isfinite((distribution.log_prob(actions) * real).sum())
print("batches ok")
# A course kind that fails its hardest rounds is played more than one that wins them.
from collections import Counter
import auto_env
c = Curriculum()
for _ in range(30):
    c.all_rounds["hop"].append(False)
    c.all_rounds["path"].append(True)
w = c.practice()
assert w["hop"] == 3.0 and w["path"] == 1.0 and w["bridge"] == 2.0, w
rng = np.random.default_rng(0)
picked = Counter(curriculum.pick_family(c.levels, rng, (), w) for _ in range(3000))
assert picked["hop"] > 2.5 * picked["path"], picked
# A staircase climbs one block per block: the goal is as high as the gap is long, straight ahead.
from sim_env import CourseMaker
lv = Curriculum().levels
lv["stairs"]["rise"] = 2
m = CourseMaker(lv)
m.set_exam("stairs")
spec, meta = m.draw(seed=3)
gap = max(abs(spec["goal"][0]), abs(spec["goal"][1])) - 4
assert meta["family"] == "stairs" and spec["dy"] == 8 and gap == 8, (spec["goal"], spec["dy"])
# The other Treenan island can sit higher or lower than the map has it.
lv["treenan"]["height"] = 4
m = CourseMaker(lv)
m.set_exam("treenan")
rises = {m.draw(seed=s)[0]["dy"] for s in range(12)}
assert len(rises) > 3, rises
# Slabs: parkour lands half a block lower on them, paths get half steps up. The rounds say so, and the simulator
# gets them as blocks to put in after the reset.
for fam in ("hop", "path"):
    lv = Curriculum().levels
    for d, v in FAMILIES[fam].items():
        lv[fam][d] = len(v) - 1 if d in ("shapes", "slope", "count", "length") else 0
    m = CourseMaker(lv)
    m.set_exam(fam)
    found = [m.draw(seed=s)[0] for s in range(15)]
    assert any(sp["settle"] and sp["cells"] is not None and len(sp["cells"]) for sp in found), fam
# Every parkour jump from a slab is half a block higher, and must still pass the jump math.
lv = Curriculum().levels
for d, v in FAMILIES["hop"].items():
    lv["hop"][d] = len(v) - 1 if d != "longer" else 0
m = CourseMaker(lv)
m.set_exam("hop")
slabbed = 0
for s in range(400):
    m.draw(seed=20_000 + s)
    for a, b in zip(m.course, m.course[1:]):
        if a in m.shape_cells:
            slabbed += 1
            dx, dz = abs(a[0] - b[0]), abs(a[2] - b[2])
            assert jump_possible(max(dx, dz), min(dx, dz), b[1] - a[1] + 0.5, run_up_ticks=2), (s, a, b)
assert slabbed > 100, slabbed
# Every path with slabs can be done: never more than a block up between neighbours (a half step from below was 1.5).
lv = Curriculum().levels
for d, v in FAMILIES["path"].items():
    lv["path"][d] = len(v) - 1
m = CourseMaker(lv)
m.set_exam("path")
shaped = 0
for s in range(600):
    sp, _ = m.draw(seed=10_000 + s)
    sunk = {c for c in m.course if c in m.shape_cells}
    assert m._path_possible(sunk), s
    shaped += bool(m.shape_cells)
    for c in m.course:  # the case that was wrong: a half step on a block one higher than the one before
        i = m.course.index(c)
        if i and (c[0], c[1] + 1, c[2]) in m.shape_cells:
            assert m.course[i - 1][1] >= c[1], (s, c)
assert shaped > 300, shaped
# Treenan from anywhere on the bed island, and never more blocks than one stack when it climbs.
lv = Curriculum().levels
lv["treenan"].update(start=1, height=4, gap=4, budget=0)
m = CourseMaker(lv)
m.set_exam("treenan")
homes = set()
for s in range(10):
    sp, _ = m.draw(seed=s)
    assert sp["blocks"] <= 64 and sp["par"] >= m.map_gap + 2 * max(sp["dy"], 0) - 1, (sp["blocks"], sp["par"], sp["dy"])
    homes.add(tuple(sp["goal"]))
assert len(homes) > 5
# With the goal 5 up, going on under it gains nothing: only climbing does, once closer than the climb left.
climb = AutoEnv._climb
assert climb(10, 5) - climb(9, 5) == 1.0, "far away, getting closer still counts"
assert climb(4, 5) == climb(0, 5), "closer than the climb left: no gain on the level"
assert climb(4, 4) < climb(4, 5), "climbing always counts"
assert climb(6, -3) == 9 and climb(6, 0) == 6, "down or level: as before"
# Shift costs nothing on parkour: it can stop at an edge.
assert "shift without blocks" not in open(auto_env.__file__).read()
print("practice ok")
