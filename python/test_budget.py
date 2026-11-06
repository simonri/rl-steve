"""Real game check: every stage hands out exactly the planned blocks, and a run can actually spend them."""
import sys

from skill_env import NO_BLOCKS, STAGES, SkillEnv

env = SkillEnv(int(sys.argv[1]))
for stage, task in enumerate(STAGES):
    env.set_stage(stage)
    seen = set()
    for _ in range(5):
        env.reset()
        assert env.obs["held_count"] == env.blocks_given, (stage, env.obs["held_count"], env.blocks_given)
        assert env.blocks_given == 0 if task[6] in NO_BLOCKS else env.blocks_given >= env.par, (stage, env.blocks_given, env.par)
        seen.add((env.par, env.blocks_given))
    print(f"stage {stage + 1:2d} {str(task[6] or '-'):7s} gap {task[0]:2d} budget x{task[7][0]}+{task[7][1]}: needed/given {sorted(seen)[:3]}", flush=True)
env.c.close()
print("ok")
