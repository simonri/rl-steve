"""Real game check: after a death the player is home, the world is unchanged and blocks are kept."""
import _path  # noqa: F401  (python/ on the import path)
import sys

from skill_env import RESPAWN_TICKS, SkillEnv, Y

env = SkillEnv(int(sys.argv[1]), stage=16)
tested = 0
for _ in range(8):
    env.reset()
    start_blocks = env.obs["held_count"]
    goal = [env.goal_x, Y + env.dy, env.goal_z]
    done, info, t_before = False, {}, 0
    while not done:
        t_before = env.t
        # Run straight ahead without shift, which goes off the platform in whatever direction it faces.
        _, _, term, trunc, info = env.step([1, 0, 0, 0, 0, 1, 3, 3])
        done = term or trunc
        if info["respawned"]:
            tested += 1
            o = env.obs
            assert abs(o["pos"][0] - env.home["pos"][0]) < 0.01 and abs(o["pos"][2] - env.home["pos"][2]) < 0.01, o["pos"]
            assert abs(o["pos"][1] - (Y + 1)) < 0.01 and o["on_ground"], o["pos"]
            assert abs(o["vel"][0]) < 0.01 and abs(o["vel"][2]) < 0.01, o["vel"]
            assert o["held_count"] == start_blocks, (o["held_count"], start_blocks)
            assert env.c.call(op="blocks", min=goal, max=goal)["names"] == ["minecraft:stone"]
            assert env.t - t_before >= RESPAWN_TICKS, (env.t, t_before)
            assert not term, "the first death must not end the round"
        elif info["deaths"] >= 2:
            assert term, "the second death ends the round"
# A block placed on the spawn point must not stop the respawn. This is what crashed training once.
for _ in range(5):
    env.reset()
    hx, hz = int(env.home["pos"][0] // 1), int(env.home["pos"][2] // 1)
    home = dict(env.home)
    spec = env._spec()
    spec["clear"] = [-30, Y - 12, -30, 30, Y + 12, 30]
    spec["platforms"].append([hx, Y + 1, hz, hx, Y + 2, hz])
    built = dict(spec)
    built["pos"] = [10.5, Y + 1, 10.5]
    built["platforms"].append([10, Y, 10, 10, Y, 10])
    env.obs = env.c.reset(**built)
    env.home = home
    env._respawn()
    o = env.obs
    assert abs(o["pos"][0] - home["pos"][0]) < 0.01 and abs(o["pos"][2] - home["pos"][2]) < 0.01, (o["pos"], home)
    tested += 1
print(f"respawn checks passed on {tested} deaths", flush=True)
assert tested >= 3
env.c.close()
print("ok")
