"""Minecraft's movement in numbers, for deciding if a jump between two blocks is possible without playing it."""
import math

GROUND_FRICTION = 0.546  # block slipperiness 0.6 times the air drag 0.91
AIR_DRAG = 0.91
GROUND_ACCEL = 0.1274  # sprinting on the ground: 0.1 * sprint 1.3 * forward key 0.98
AIR_ACCEL = 0.02548  # sprinting in the air: 0.02 * 1.3 * 0.98
SPRINT_JUMP_BOOST = 0.2
JUMP_SPEED = 0.42
GRAVITY = 0.08
VERTICAL_DRAG = 0.98
HALF_WIDTH = 0.3


def jump_reach(run_up_ticks=0, height=0, sprint=True):
    """How far the middle of the player travels sideways before coming down to a block `height` above the start.
    Each tick: add acceleration, move, then apply the friction of where the tick began. That is still the ground
    on the jump tick. Measured in the game: 2.88 with no run-up and 3.63 at full speed, sprinting and on the flat."""
    moved_last_tick = 0.0
    accel = GROUND_ACCEL if sprint else GROUND_ACCEL / 1.3
    for _ in range(run_up_ticks):
        moved_last_tick = moved_last_tick * GROUND_FRICTION + accel
    speed = moved_last_tick * GROUND_FRICTION + accel + (SPRINT_JUMP_BOOST if sprint else 0.0)
    y, vy, travelled = 0.0, JUMP_SPEED, 0.0
    air_accel = AIR_ACCEL if sprint else AIR_ACCEL / 1.3
    for tick in range(60):
        travelled += speed
        y += vy
        vy = (vy - GRAVITY) * VERTICAL_DRAG
        speed = speed * (GROUND_FRICTION if tick == 0 else AIR_DRAG) + air_accel
        if vy < 0 and y <= height:
            return travelled
    return travelled


def edge_gap(dx, dz):
    """The distance between the closest points of two blocks whose positions differ by dx, dz."""
    return math.hypot(max(abs(dx) - 1, 0), max(abs(dz) - 1, 0))


def jump_possible(dx, dz, dy=0, run_up_ticks=0, sprint=True, margin=0.3):
    """The player's feet may hang over each block's edge by half its width, and the margin allows for human error."""
    if dy > 1:
        return False  # a jump does not lift the feet higher than this
    return edge_gap(dx, dz) + margin <= jump_reach(run_up_ticks, dy, sprint) + 2 * HALF_WIDTH


def hits_pole(start, end, poles):
    """Does a player moving in a straight line between two block centres hit any pole column?"""
    steps = 40
    for i in range(steps + 1):
        x = start[0] + (end[0] - start[0]) * i / steps
        z = start[1] + (end[1] - start[1]) * i / steps
        for px, pz in poles:
            if px - HALF_WIDTH < x < px + 1 + HALF_WIDTH and pz - HALF_WIDTH < z < pz + 1 + HALF_WIDTH:
                return True
    return False
