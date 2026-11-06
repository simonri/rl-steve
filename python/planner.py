"""A teacher that plays parkour and paths by planning every move with Minecraft's movement rules.

It knows the course (it is a teacher, not the model), and works out each tick from where the player really is, so a late key
or a bad landing is corrected on the next tick instead of ruining the round. Jumps are planned by simulating the flight:
it jumps when a landing on the next block is within reach, and in the air it keeps forward held only as long as that
keeps the landing on the block. The moves stay inside what the model itself can press.
"""
import math

from auto_env import TURN_STEPS

AIR_DRAG, GROUND_FRICTION = 0.91, 0.546
SPRINT_GROUND, SPRINT_AIR, WALK_GROUND, WALK_AIR = 0.1274, 0.02548, 0.098, 0.0196
JUMP_UP, JUMP_BOOST, GRAVITY, VERTICAL_DRAG = 0.42, 0.2, 0.08, 0.98
FEET = 0.3  # half the player's width: it stands on a block while any part of it is over the block


def turn_index(error):
    """The turn step that gets closest to the error without overshooting much."""
    return min(range(len(TURN_STEPS)), key=lambda i: abs(error - TURN_STEPS[i]))


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def facing(yaw_degrees):
    yaw = math.radians(yaw_degrees)
    return -math.sin(yaw), math.cos(yaw)


def fly(x, y, z, vx, vy, vz, push_ticks, dx, dz, top, from_ground=False, cell=None):
    """Where it comes down on a block whose top is `top`, if forward stays held for push_ticks more ticks.
    Returns the landing point, or None when it falls past that height without coming down on it first.
    On the tick it jumps, the ground's friction still applies afterward, not the air's."""
    for tick in range(60):
        before = y
        x, y, z = x + vx, y + vy, z + vz
        if cell is not None and before < top - 0.01 and y < top - 0.01 and on_block(x, z, cell, FEET):
            return None  # still under its top when it gets there: it runs into the side instead of landing on it
        if vy < 0 and y <= top:
            return x, z
        accel = SPRINT_AIR if tick < push_ticks else 0.0
        drag = GROUND_FRICTION if from_ground and tick == 0 else AIR_DRAG
        vx, vz = vx * drag + accel * dx, vz * drag + accel * dz
        vy = (vy - GRAVITY) * VERTICAL_DRAG
    return None


def on_block(x, z, cell, margin=FEET - 0.1):
    return cell[0] - margin < x < cell[0] + 1 + margin and cell[2] - margin < z < cell[2] + 1 + margin


class Teacher:
    WAIT_LIMIT = 8  # ticks of waiting for a jump that fits before trying something else

    def __init__(self, env):
        self.env = env
        self.branch = ""
        self.waited = 0
        self.backing = False

    def target(self):
        env = self.env
        if env.landed < len(env.landing_order):
            return env.landing_order[env.landed]
        return (env.goal_x, int(round(env.obs["pos"][1])), env.goal_z)

    def _jump_landings(self, x, y, z, vx, vz, fx, fz, top, sprint):
        """Where a jump right now could come down: sprinting or not, then forward for some ticks, or braking."""
        accel = SPRINT_GROUND + JUMP_BOOST if sprint else WALK_GROUND
        nvx, nvz = vx * GROUND_FRICTION + accel * fx, vz * GROUND_FRICTION + accel * fz
        spots = []
        for ticks in (0, 1, 2, 3, 4, 6, 8, 60):
            spot = fly(x, y, z, nvx, JUMP_UP, nvz, ticks, fx, fz, top, from_ground=True, cell=self.cell)
            if spot is not None:
                spots.append((ticks, spot))
        for ticks in (2, 4, 8, 60):  # holding back to brake
            spot = fly(x, y, z, nvx, JUMP_UP, nvz, ticks, -fx, -fz, top, from_ground=True, cell=self.cell)
            if spot is not None:
                spots.append((-ticks, spot))
        return spots

    def act(self):
        env, o = self.env, self.env.obs
        x, y, z = o["pos"]
        vx, vy, vz = o["vel"]
        cell = self.cell = self.target()
        tx, tz = cell[0] + 0.5, cell[2] + 0.5
        top = cell[1]
        dx, dz = tx - x, tz - z
        distance = math.hypot(dx, dz) or 1e-6
        ux, uz = dx / distance, dz / distance
        wanted_yaw = math.degrees(math.atan2(-ux, uz))
        yaw_i = turn_index(wrap(wanted_yaw - o["yaw"]))
        pitch_i = turn_index(max(-60.0, min(60.0, 10.0 - o["pitch"])))
        fx, fz = facing(o["yaw"])
        aimed = fx * ux + fz * uz > 0.97
        here = (math.floor(x), math.floor(y) - 1, math.floor(z))
        adjacent = max(abs(cell[0] - here[0]), abs(cell[2] - here[2])) <= 1
        rises = top - math.floor(y) >= 1

        if not o["on_ground"]:
            # In the air: whatever input brings the landing closest to the middle of the block.
            self.branch = "air"
            best, choice = None, 0
            for ticks in (0, 1, 2, 4, 8, 60):
                for sign in (1, -1):
                    spot = fly(x, y, z, vx, vy, vz, ticks, sign * fx, sign * fz, top, cell=cell)
                    if spot is None:
                        continue
                    miss = math.hypot(spot[0] - tx, spot[1] - tz)
                    if best is None or miss < best - 1e-3:
                        best, choice = miss, sign * ticks
            move = 1 if choice > 0 else 2 if choice < 0 else 0
            return [move, 0, 0, 0, 0, int(choice > 0), yaw_i, pitch_i]

        if not aimed:
            self.branch = "turn"
            return [0, 0, 0, 0, 0, 0, yaw_i, pitch_i]
        speed = math.hypot(vx, vz)
        if adjacent and not rises:
            self.branch = "step"
            return [1, 0, 0, 0, 0, int(distance > 1.2), yaw_i, pitch_i]
        if adjacent and rises:
            self.branch = "step up"
            return [1, 0, 0, 0, int(distance < 1.25), 0, yaw_i, pitch_i]

        # A gap. Jump now if some way of jumping comes down on the block.
        edge = env._edge_left(o) * 2  # ground left toward the block
        for sprint in (True, False):
            fits = [t for t, (sx, sz) in self._jump_landings(x, y, z, vx, vz, fx, fz, top, sprint) if on_block(sx, sz, cell)]
            if fits:
                self.branch, self.waited, self.backing = "jump", 0, False
                return [1, 0, 0, 0, 1, int(sprint), yaw_i, pitch_i]
        if edge < 0.1 + speed:
            self.branch = "jump at the edge"
            return [1, 0, 0, 0, 1, 1, yaw_i, pitch_i]
        reach = max((math.hypot(sx - x, sz - z) for _, (sx, sz) in self._jump_landings(x, y, z, vx, vz, fx, fz, top, True)), default=0.0)
        if self.backing:
            # Backing up for a longer run: until the ground behind runs out, then go.
            self.waited += 1
            if self.waited > 12 or speed < 0.01 and self.waited > 3:
                self.backing = False
            self.branch = "back up"
            return [2, 0, 0, 0, 0, 0, yaw_i, pitch_i]
        if reach < distance:
            if speed < 0.05 and edge < 0.6:
                self.backing, self.waited = True, 0  # too close to the edge to build up speed: step back first
                self.branch = "back up"
                return [2, 0, 0, 0, 0, 0, yaw_i, pitch_i]
            self.branch = "run up"
            return [1, 0, 0, 0, 0, 1, yaw_i, pitch_i]
        self.waited += 1
        self.branch = "wait"
        if self.waited > self.WAIT_LIMIT:
            self.waited = 0
            return [1, 0, 0, 0, 0, 0, yaw_i, pitch_i]  # creep forward: the walk changes what a jump can reach
        return [0, 0, 0, 0, 0, 0, yaw_i, pitch_i]
