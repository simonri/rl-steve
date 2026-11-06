"""A camera for people watching in the real game. The round opens on one shot with the whole course in it, start to
goal, and that shot stays: it only cuts when it must (the player has been out of the picture or hidden for a while,
or a block now sits where the camera is). After such a cut the new shot frames the player and the rest of the way.

A watcher's view is moved by teleporting, which the game does not smooth, so every move is a jump; fewer is better.

    cam = Shots()
    cam.new_round(env)                      # after env.reset()
    at = cam.update(env)                    # every tick: a new shot [x, y, z, yaw, pitch] for watch_at, or None
"""
import math

V_FOV = 90.0  # vertical field of view, as Minecraft's setting counts it
ASPECT = 16 / 9
H_HALF = math.degrees(math.atan(math.tan(math.radians(V_FOV / 2)) * ASPECT))  # about 60.6 degrees
FIT = 0.85  # share of the half-angles the framed area may fill
ELEVATIONS = (35.0, 50.0, 25.0, 65.0)  # degrees above the framed area, best first
TURNS = (0, 20, -20, 40, -40, 70, -70, 100, -100, 140, -140, 180)
MIN_DIST, MAX_DIST = 8.0, 60.0  # the far end stays inside what a watcher's game shows
LOST_FOR = 30  # ticks out of the picture before a cut
HIDDEN_FOR = 60  # ticks with the view to the player blocked before a cut


class Shots:
    def __init__(self):
        self.shot = None  # (x, y, z, yaw, pitch)
        self.lost = self.hidden = 0
        self.solid = set()
        self.extent = None  # (min corner, max corner) of the course

    def new_round(self, env):
        """Everything known to be built (the course's boxes, a pasted map), and the area the opening shot frames."""
        self.solid = set()
        spec = getattr(env, "spec", {}) or {}
        points = [(0, 100, 0), (env.goal_x, 100 + 1 + env.dy, env.goal_z)]
        for b in spec.get("platforms", []):
            for x in range(int(b[0]), int(b[3]) + 1):
                for y in range(int(b[1]), int(b[4]) + 1):
                    for z in range(int(b[2]), int(b[5]) + 1):
                        self.solid.add((x, y, z))
            points += [tuple(b[:3]), tuple(b[3:])]
        if getattr(env, "extras", None) == "treenan":
            cells = [c for c, s in getattr(env, "next_cells", {}).items() if s != "minecraft:air"]
            self.solid |= set(cells)
            points += cells
        points += [tuple(c) for c in getattr(env, "landing_order", [])]
        lo = [min(p[i] for p in points) for i in range(3)]
        hi = [max(p[i] for p in points) + 1 for i in range(3)]
        hi[1] += 2  # room to stand on top
        self.extent = (lo, hi)
        self.shot = None
        self.lost = self.hidden = 0

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
        n = max(2, int(math.dist(a, b) * 3))
        return not any(blocked((math.floor(ax + (b[0] - ax) * i / n), math.floor(ay + (b[1] - ay) * i / n), math.floor(az + (b[2] - az) * i / n)))
                       for i in range(1, n))

    @staticmethod
    def _angles(shot, p):
        """Horizontal and vertical degrees between the middle of the picture and point p."""
        x, y, z, yaw, pitch = shot
        ry, rp = math.radians(yaw), math.radians(pitch)
        fwd = (-math.sin(ry) * math.cos(rp), -math.sin(rp), math.cos(ry) * math.cos(rp))
        right = (-math.cos(ry), 0.0, -math.sin(ry))
        up = (fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2], fwd[0] * right[1] - fwd[1] * right[0])
        d = (p[0] - x, p[1] - y, p[2] - z)
        f = sum(a * b for a, b in zip(d, fwd))
        if f <= 0:
            return 180.0, 180.0
        return math.degrees(math.atan2(abs(sum(a * b for a, b in zip(d, right))), f)), math.degrees(math.atan2(abs(sum(a * b for a, b in zip(d, up))), f))

    def _frame(self, lo, hi, axis, head, local):
        """A shot that holds the box lo..hi, from the side of `axis` (the way through it), clear of every block."""
        centre = [(a + b) / 2 for a, b in zip(lo, hi)]
        corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        base = math.atan2(axis[1], axis[0]) + math.pi / 2
        best = None
        for elev_deg in ELEVATIONS:
            for side in (1, -1):
                for turn in TURNS:
                    ang = base + math.radians(turn) + (0 if side == 1 else math.pi)
                    elev = math.radians(elev_deg)
                    back = (math.cos(ang) * math.cos(elev), math.sin(elev), math.sin(ang) * math.cos(elev))
                    # The nearest distance at which every corner is inside the picture.
                    for d in [MIN_DIST * 1.25 ** k for k in range(20)]:
                        if d > MAX_DIST:
                            break
                        at = (centre[0] + back[0] * d, centre[1] + back[1] * d, centre[2] + back[2] * d)
                        shot = self._aim(at, centre)
                        if all(h <= H_HALF * FIT and v <= V_FOV / 2 * FIT for h, v in (self._angles(shot, c) for c in corners)):
                            break
                    if self._clear(at, head, local) and self._clear(at, tuple(centre), local):
                        score = d + abs(turn) * 0.05 + (elev_deg - 35) ** 2 * 0.01
                        if best is None or score < best[0]:
                            best = (score, shot)
            if best:
                return best[1]
        return self._aim((centre[0], hi[1] + MIN_DIST * 2, centre[2] + 0.01), centre)  # nothing clear: straight above

    @staticmethod
    def _aim(at, look):
        dx, dy, dz = look[0] - at[0], look[1] - at[1], look[2] - at[2]
        return (at[0], at[1], at[2], math.degrees(math.atan2(-dx, dz)), math.degrees(-math.atan2(dy, math.hypot(dx, dz))))

    def update(self, env):
        o = env.obs
        for c in o.get("clicks") or []:
            self.solid.add((int(c[0]), int(c[1]), int(c[2])))
        x, y, z = o["pos"]
        head = (x, y + 1.6, z)
        local = self._local(o)
        if self.shot is None:  # the opening shot: the whole course
            lo, hi = self.extent
            self.shot = self._frame(lo, hi, (env.goal_x, env.goal_z), head, local)
            return list(self.shot)
        h, v = self._angles(self.shot, head)
        self.lost = self.lost + 1 if (h > H_HALF * 0.95 or v > V_FOV / 2 * 0.95) else 0
        self.hidden = self.hidden + 1 if not self._clear(self.shot[:3], head, local) else 0
        inside = not self._clear(self.shot[:3], self.shot[:3], local)  # a block was placed where the camera is
        if inside or self.lost >= LOST_FOR or self.hidden >= HIDDEN_FOR or math.dist(self.shot[:3], head) > MAX_DIST * 1.2:
            # The rest of the way: from the player to the goal.
            gx, gy, gz = env.goal_x + 0.5, 100 + 1 + env.dy, env.goal_z + 0.5
            lo = [min(x, gx) - 2, min(y, gy) - 1, min(z, gz) - 2]
            hi = [max(x, gx) + 2, max(y, gy) + 3, max(z, gz) + 2]
            self.shot = self._frame(lo, hi, (gx - x, gz - z), head, local)
            self.lost = self.hidden = 0
            return list(self.shot)
        return None
