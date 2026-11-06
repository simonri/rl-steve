"""Checks the rotated frame without starting the game."""
import _path  # noqa: F401  (python/ on the import path)
import math

import numpy as np

from skill_env import GRID, R, rotate


def forward(yaw):
    r = math.radians(yaw)
    return -math.sin(r), math.cos(r)


for yaw in (0, 37, 90, 135, 180, 225, 270, 359, -45, -120):
    k = int(round(yaw / 90))
    rx, rz = rotate(*forward(yaw), k)
    ex, ez = forward(yaw - 90 * k)
    assert abs(rx - ex) < 1e-9 and abs(rz - ez) < 1e-9, (yaw, rx, rz, ex, ez)

size = 2 * R + 1
for k in range(4):
    assert sorted(GRID[k]) == list(range(size**3))
    world = np.zeros(size**3, dtype=np.uint8)
    wx, wz = 1, 2
    world[(wx + R) * size * size + (0 + R) * size + (wz + R)] = 1
    out = world[GRID[k]]
    nx, nz = rotate(wx, wz, k)
    assert out[(nx + R) * size * size + (0 + R) * size + (nz + R)] == 1, k
print("rotation ok")
