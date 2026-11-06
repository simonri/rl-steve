"""A camera for people watching in the real game: it follows the player, never sits inside a block, keeps a clear line
to the player, and frames only what matters now, for a field of view of 90 degrees.

    cam = Follow()
    cam.new_round(env)                      # after env.reset()
    at = cam.update(env)                    # every tick: [x, y, z, yaw, pitch] for the mod's watch_at, or None
"""
import math

FOV = 90.0  # vertical, as Minecraft counts it; the screen is wider than it is high, so this is the tighter one
NEAR, FAR = 5.0, 13.0  # blocks from the player: closer is cramped, further makes the player a speck
AHEAD = 6.0  # blocks of the way ahead kept in the picture, so the next jump or the bridge's end shows
ELEVATION = 28.0  # degrees above the player
FOLLOW = 0.18  # share of the way to the wanted spot moved each tick: smooth, still keeps up with a sprint
SNAP = 20.0  # further than this from the wanted spot (a new round, a respawn): jump straight there
TURNS = (0, 25, -25, 50, -50, 80, -80, 115, -115, 150, -150, 180)  # degrees tried around the preferred side


class Follow:
    def __init__(self):
        self.at = None  # [x, y, z] now
        self.side = 1.0  # which side of the way it watches from; kept until that side is blocked
        self.turn = 0.0
        self.solid = set()

    def new_round(self, env):
        """Everything known to be built: the course's boxes, a pasted map, and nothing yet placed by the player."""
        self.solid = set()
        spec = getattr(env, "spec", {}) or {}
        for b in spec.get("platforms", []):
            for x in range(int(b[0]), int(b[3]) + 1):
                for y in range(int(b[1]), int(b[4]) + 1):
                    for z in range(int(b[2]), int(b[5]) + 1):
                        self.solid.add((x, y, z))
        if getattr(env, "extras", None) == "treenan":
            self.solid |= {c for c, s in getattr(env, "next_cells", {}).items() if s != "minecraft:air"}
        self.at = None

    def _local(self, o):
        """Blocks right around the player as the game reports them, including any the player placed."""
        solid, feet = o.get("solid"), o.get("feet")
        if not solid or not feet:
            return set()
        size = round(len(solid) ** (1 / 3))
        r = size // 2
        fx, fy, fz = (math.floor(v) for v in feet)
        out = set()
        for i, ch in enumerate(solid):
            if ch == "1":
                x, rest = divmod(i, size * size)
                y, z = divmod(rest, size)
                out.add((fx + x - r, fy + y - r, fz + z - r))
        return out

    def _blocked(self, cell, local):
        return cell in self.solid or cell in local

    def _clear(self, a, b, local):
        """No block on the line from a to b, nor at a or the block above a camera that would see through it."""
        ax, ay, az = a
        for c in ((math.floor(ax), math.floor(ay), math.floor(az)), (math.floor(ax), math.floor(ay) + 1, math.floor(az)), (math.floor(ax), math.floor(ay) - 1, math.floor(az))):
            if self._blocked(c, local):
                return False
        n = max(2, int(math.dist(a, b) * 4))
        for i in range(1, n):
            t = i / n
            c = (math.floor(ax + (b[0] - ax) * t), math.floor(ay + (b[1] - ay) * t), math.floor(az + (b[2] - az) * t))
            if self._blocked(c, local):
                return False
        return True

    def update(self, env):
        o = env.obs
        x, y, z = o["pos"]
        for c in o.get("clicks") or []:
            self.solid.add((int(c[0]), int(c[1]), int(c[2])))
        local = self._local(o)
        try:
            rx, rz = env._route(x, z)
        except Exception:
            rx, rz = 1.0, 0.0
        # What to frame: the player and the stretch just ahead, never the whole course.
        look = (x + rx * AHEAD * 0.4, y + 1.0, z + rz * AHEAD * 0.4)
        spread = AHEAD * 0.6 + 1.5
        half = math.radians(FOV / 2) * 0.8  # a margin, so the player is not at the edge
        dist = min(FAR, max(NEAR, spread / math.tan(half) + 2.0))
        head = (x, y + 1.6, z)
        best = None
        for d in (dist, dist * 0.75, dist * 0.55, NEAR * 0.8):
            for sign in (self.side, -self.side):
                for turn in sorted(TURNS, key=lambda t: abs(t - self.turn)):
                    ang = math.atan2(rz, rx) + sign * math.pi / 2 + math.radians(turn) * sign
                    elev = math.radians(ELEVATION)
                    cx = look[0] + math.cos(ang) * d * math.cos(elev)
                    cz = look[2] + math.sin(ang) * d * math.cos(elev)
                    cy = look[1] + d * math.sin(elev)
                    if self._clear((cx, cy, cz), head, local):
                        best = (cx, cy, cz, sign, turn)
                        break
                if best:
                    break
            if best:
                break
        if best is None:  # nothing clear: straight above, where nothing usually is
            best = (look[0], look[1] + NEAR, look[2] + 0.01, self.side, self.turn)
        tx, ty, tz, self.side, self.turn = best
        if self.at is None or math.dist(self.at, (tx, ty, tz)) > SNAP:
            self.at = [tx, ty, tz]
        else:
            nxt = [a + (t - a) * FOLLOW for a, t in zip(self.at, (tx, ty, tz))]
            if self._clear(tuple(nxt), head, local):
                self.at = nxt
            else:  # the smooth path would go through something: cut straight to the clear spot
                self.at = [tx, ty, tz]
        dx, dy, dz = look[0] - self.at[0], look[1] - self.at[1], look[2] - self.at[2]
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = math.degrees(-math.atan2(dy, math.hypot(dx, dz)))
        return [*self.at, yaw, pitch]
