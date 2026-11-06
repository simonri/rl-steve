"""A camera for people watching in the real game, like a TV director: it holds still and cuts to a new shot only when
it has to. A watcher's view is moved by teleporting, and teleports are not smoothed, so a camera that follows smoothly
is a camera that jerks. Each shot never sits inside a block, starts with a clear line to the player, frames the player
and the way just ahead for a field of view of 90 degrees, and is set ahead of the player so it walks into the picture.

    cam = Shots()
    cam.new_round(env)                      # after env.reset()
    at = cam.update(env)                    # every tick: a new shot [x, y, z, yaw, pitch] for watch_at, or None
"""
import math

FOV = 90.0  # vertical, as Minecraft counts it; the screen is wider than it is high, so this is the tighter one
NEAR, FAR = 6.0, 13.0  # blocks from the player: closer is cramped, further makes the player a speck
AHEAD = 7.0  # blocks ahead along the way that the shot is aimed toward, so the player moves into it
ELEVATION = 28.0  # degrees above the player
HOLD = 60  # ticks a shot stays at least: three seconds
BLOCKED_FOR = 20  # ticks the view may be blocked before cutting: a second, so a passing block does not cause a cut
OUT_OF_FRAME = 38.0  # degrees off the middle of the picture before the player counts as leaving it
TURNS = (0, 25, -25, 50, -50, 80, -80, 115, -115, 150, -150, 180)  # degrees tried around the preferred side


class Shots:
    def __init__(self):
        self.shot = None  # (x, y, z, yaw, pitch)
        self.age = 0
        self.blocked = 0
        self.side = 1.0
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
        self.shot = None

    @staticmethod
    def _local(o):
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

    def _clear(self, a, b, local):
        """No block at the camera, just above or below it, or anywhere on the line from it to b."""
        blocked = lambda c: c in self.solid or c in local
        ax, ay, az = a
        cx, cy, cz = math.floor(ax), math.floor(ay), math.floor(az)
        if blocked((cx, cy, cz)) or blocked((cx, cy + 1, cz)) or blocked((cx, cy - 1, cz)):
            return False
        n = max(2, int(math.dist(a, b) * 4))
        return not any(blocked((math.floor(ax + (b[0] - ax) * i / n), math.floor(ay + (b[1] - ay) * i / n), math.floor(az + (b[2] - az) * i / n)))
                       for i in range(1, n))

    def _new_shot(self, env, head, local):
        o = env.obs
        x, y, z = o["pos"]
        try:
            rx, rz = env._route(x, z)
        except Exception:
            rx, rz = 1.0, 0.0
        # Aimed between the player and the way a few blocks ahead: the player starts on one side and walks across.
        look = (x + rx * AHEAD * 0.45, y + 1.0, z + rz * AHEAD * 0.45)
        half = math.radians(FOV / 2) * 0.75
        dist = min(FAR, max(NEAR, (AHEAD * 0.6 + 1.5) / math.tan(half) + 2.0))
        elev = math.radians(ELEVATION)
        for d in (dist, dist * 0.75, NEAR):
            for sign in (self.side, -self.side):
                for turn in TURNS:
                    ang = math.atan2(rz, rx) + sign * (math.pi / 2 + math.radians(turn))
                    at = (look[0] + math.cos(ang) * d * math.cos(elev), look[1] + d * math.sin(elev), look[2] + math.sin(ang) * d * math.cos(elev))
                    if self._clear(at, head, local) and self._clear(at, look, local):
                        self.side = sign
                        return self._aim(at, look)
        return self._aim((look[0], look[1] + NEAR, look[2] + 0.01), look)  # nothing clear: from straight above

    @staticmethod
    def _aim(at, look):
        dx, dy, dz = look[0] - at[0], look[1] - at[1], look[2] - at[2]
        return (at[0], at[1], at[2], math.degrees(math.atan2(-dx, dz)), math.degrees(-math.atan2(dy, math.hypot(dx, dz))))

    def _off_middle(self, head):
        """Degrees between the middle of the picture and the player's head."""
        x, y, z, yaw, pitch = self.shot
        ry, rp = math.radians(yaw), math.radians(pitch)
        fwd = (-math.sin(ry) * math.cos(rp), -math.sin(rp), math.cos(ry) * math.cos(rp))
        to = (head[0] - x, head[1] - y, head[2] - z)
        n = math.sqrt(sum(v * v for v in to)) or 1.0
        return math.degrees(math.acos(max(-1.0, min(1.0, sum(f * t for f, t in zip(fwd, to)) / n))))

    def update(self, env):
        o = env.obs
        for c in o.get("clicks") or []:
            self.solid.add((int(c[0]), int(c[1]), int(c[2])))
        x, y, z = o["pos"]
        head = (x, y + 1.6, z)
        local = self._local(o)
        if self.shot is None:
            self.shot, self.age, self.blocked = self._new_shot(env, head, local), 0, 0
            return list(self.shot)
        self.age += 1
        self.blocked = self.blocked + 1 if not self._clear(self.shot[:3], head, local) else 0
        inside = not self._clear(self.shot[:3], self.shot[:3], local)  # a block was placed where the camera is
        dist = math.dist(self.shot[:3], head)
        leaving = self._off_middle(head) > OUT_OF_FRAME or dist > FAR * 1.4
        if inside or self.blocked >= BLOCKED_FOR or (self.age >= HOLD and leaving) or dist > FAR * 2.5:
            self.shot, self.age, self.blocked = self._new_shot(env, head, local), 0, 0
            return list(self.shot)
        return None
