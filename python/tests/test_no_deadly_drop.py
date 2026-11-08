"""No course needs a deadly fall to win: no goal lies more than auto_env.MAX_DROP below the start, whatever the levels.

    python test_no_deadly_drop.py
"""
import _path  # noqa: F401  (python/ on the import path)

import auto_env
import curriculum
from sim_env import CourseMaker

top = curriculum.Curriculum().levels
for fam, dials in curriculum.FAMILIES.items():
    for d, values in dials.items():
        top[fam][d] = len(values) - 1
lowest = {}
for fam in ("stairs", "bridge", "treenan", "bedwars", "hop", "path"):
    m = CourseMaker(top, seed=3)
    m.set_exam(fam, True)
    dys = []
    for s in range(120):
        m.draw(seed=s)
        dys.append(m.dy)
    lowest[fam] = min(dys)
    assert min(dys) >= -auto_env.MAX_DROP, (fam, min(dys))
assert lowest["stairs"] == -auto_env.MAX_DROP, "stairs down still goes as deep as is safe"
print("no deadly drop ok", lowest)
