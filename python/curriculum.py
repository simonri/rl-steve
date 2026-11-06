"""Decides what to practise. Every kind of course has dials, each with levels from easy to hard.

A dial goes up when rounds with it at its limit are won often enough, and back down when they stop being won,
so nothing has to be scheduled by hand and a skill that fades gets practised again.
"""
from collections import deque

WINDOW = 30  # rounds at a limit before judging it
RAISE_AT = 0.7
LOWER_AT = 0.25
ALL_DIALS_FLOOR = 0.3  # rounds with every dial at its limit have to stay above this before any dial goes up
ALL_DIALS_MIN_ROUNDS = 10
SHARE_ALL_DIALS, SHARE_ONE_DIAL = 0.3, 0.5  # of the rest, every dial is drawn freely below its limit
NEAR_TOP, SHARE_ALL_DIALS_NEAR_TOP, SHARE_ONE_DIAL_NEAR_TOP = 0.75, 0.6, 0.3  # the real test is everything at once, so close to the top it gets most of the practice
MASTERED_AT = 0.9  # share of rounds with every dial at its limit
WEAK_BOOST = 2.0  # a kind that never wins with every dial at its limit is played 3x as often as one that always does
FINISHED_WEIGHT = 0.3  # how often a mastered course kind is still played, compared with one in progress

YAW = (15, 45, 90, 180)  # degrees the start can be turned away from the way to go

ISLAND, PILLAR, WALL, SLALOM = "island", "pillar", "wall", "slalom"
FLAT, SMALL_STEPS, HILLS, DROPS = (0,), (-1, 0, 1), (-3, -2, -1, 0, 1, 2, 3), (-8, -6, -4, -3, -1, 0, 1, 3)
ROOMY, NORMAL, TIGHT, STRICT = (3.0, 2), (2.0, 2), (1.75, 2), (1.5, 2)

FAMILIES = {
    "hop": dict(
        # Next block: blocks ahead and to the side. Sideways ones switch side every jump.
        offset=((2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2), (4, 0), (4, 1), (4, 2)),
        yaw=YAW,
        takeoff=("platform", "block"),
        count=(1, 2, 3, 4, 5, 6, 8),
        height=(FLAT, (0, 1), SMALL_STEPS, (-2, -1, 0, 1)),
        ticks=(60, 40, 30, 20),
        # Harder, and only open once all of the above is at its limit. Every jump still has to pass the jump math.
        far=(False, True),  # adds the longest jump that is still possible from a single block
        mix=(False, True),  # every jump of a course can be a different one
        bend=(False, True),  # the course can turn, so the next block is not always further the same way
        longer=(1, 2, 3),  # times the number of blocks
        shapes=(False, True),  # some blocks are slabs: half a block lower to land on, half a block harder to jump from
    ),
    "path": dict(
        length=(4, 6, 8, 11, 14, 17, 20),
        holes=(False, True),
        yaw=YAW,
        ticks=(40, 30, 20),
        wind=("gentle", "tight", "around"),  # how often it turns, and if it can double back
        slope=("flat", "some steps", "stairs", "stairs and drops"),
        # Harder, and only open once all of the above is at its limit.
        wide_holes=(False, True),  # holes up to 3 blocks long
        longer=(1, 2),
        shapes=(False, True),  # slabs as on real maps: half steps up, and blocks sunk half a block
    ),
    "bridge": dict(
        gap=(5, 8, 12, 16, 20),  # from 5, since anything shorter can be jumped and teaches nothing about building
        yaw=YAW,
        start=("edge", "center"),
        directions=("axis", "eight", "any"),
        heights=(FLAT, SMALL_STEPS, HILLS, DROPS),
        budget=(ROOMY, NORMAL, TIGHT, STRICT),
        obstacle=(None, ISLAND, PILLAR, WALL, SLALOM),
        ticks=(40, 30, 20, 15, 10, 7),
        # Harder, and only open once all of the above is at its limit.
        longer=(1.0, 1.5),
        faster=(None, 6, 5, 4, 3),  # ticks per block with almost no spare time: chasing the record. The game itself sets the ceiling.
    ),
    "treenan": dict(
        gap=(10, 15, 20, 25, 30),
        budget=(NORMAL, TIGHT, STRICT),
        ticks=(40, 30, 20, 15, 10, 7),
        height=(0, 2, 4, 6, 8),  # the other island moved up or down by up to this many blocks
        start=("bed", "anywhere"),  # anywhere on the bed island, with its trees and walls between it and the edge
    ),
    # A bridge that climbs one block per block ahead, 45 degrees: a staircase built while walking up it.
    "stairs": dict(
        rise=(3, 5, 8, 12, 16),
        yaw=YAW,
        start=("edge", "center"),
        directions=("axis", "eight"),
        budget=(ROOMY, NORMAL, TIGHT),
        ticks=(60, 40, 30, 20),
        # Harder, and only open once all of the above is at its limit.
        steep=(1.0, 1.5, 2.0),  # blocks up per block ahead: steeper than 45 degrees means towering on the way
        way=("up", "up or down"),  # down too: building a way down without falling off it
    ),
}
# A dial stays at its easiest until another dial on the same course has reached a level.
EXPERT = {"hop": ("far", "mix", "bend", "longer", "shapes"), "path": ("wide_holes", "longer", "shapes"), "bridge": ("longer", "faster"), "stairs": ("steep", "way")}  # dials that wait for the rest of the course kind
DIAL_NEEDS = {("bridge", "obstacle"): ("gap", 1)}  # obstacles need a gap of 8 to fit in
# A course kind stays closed until another one has come this far.
FAMILY_NEEDS = {"treenan": ("bridge", dict(gap=3, directions=2)), "stairs": ("bridge", dict(gap=2))}


class Curriculum:
    def __init__(self, levels=None):
        self.levels = {f: {d: 0 for d in dials} for f, dials in FAMILIES.items()}
        for family, dials in (levels or {}).items():
            for dial, level in dials.items():
                if family in self.levels and dial in self.levels[family]:
                    self.levels[family][dial] = min(level, len(FAMILIES[family][dial]) - 1)
        self.limit_rounds = {f: {d: deque(maxlen=WINDOW) for d in dials} for f, dials in FAMILIES.items()}
        self.all_rounds = {f: deque(maxlen=WINDOW) for f in FAMILIES}
        self.lowered = set()  # dials that just went down: they have to prove the level below twice before going up again
        self.failed_once = set()  # dials whose last window was bad: one bad window can be bad luck, two in a row lower it

    def record(self, family, tested, clean):
        """One finished round. Returns (dial, new level) when that changed a dial."""
        if tested == "all":
            self.all_rounds[family].append(clean)
            return None
        if tested is None:
            return None
        rounds = self.limit_rounds[family][tested]
        rounds.append(clean)
        if len(rounds) < WINDOW:
            return None
        rate = sum(rounds) / WINDOW
        level = self.levels[family][tested]
        if rate > LOWER_AT:
            self.failed_once.discard((family, tested))
        if rate >= RAISE_AT and level + 1 < len(FAMILIES[family][tested]) and self._all_dials_hold(family):
            if (family, tested) in self.lowered:
                self.lowered.discard((family, tested))
                rounds.clear()  # one more full window first, so a level it just failed at is not tried again right away
                return None
            return self._move(family, tested, level + 1)
        if rate <= LOWER_AT and level > 0:
            if (family, tested) not in self.failed_once:
                self.failed_once.add((family, tested))
                rounds.clear()
                return None
            self.failed_once.discard((family, tested))
            self.lowered.add((family, tested))
            return self._move(family, tested, level - 1)
        return None

    def _all_dials_hold(self, family):
        """Nothing goes up until enough rounds with everything at its limit have shown that this level holds."""
        rounds = self.all_rounds[family]
        return len(rounds) >= ALL_DIALS_MIN_ROUNDS and sum(rounds) / len(rounds) >= ALL_DIALS_FLOOR

    def _move(self, family, dial, level):
        self.levels[family][dial] = level
        self.limit_rounds[family][dial].clear()
        self.all_rounds[family].clear()
        return dial, level

    def practice(self):
        """How much each course kind is played: more the less often it wins with every dial at its limit, which is
        what the exam asks. Up to 1 + WEAK_BOOST times as often as one that always wins."""
        weights = {}
        for f, rounds in self.all_rounds.items():
            rate = sum(rounds) / len(rounds) if len(rounds) >= ALL_DIALS_MIN_ROUNDS else 0.5
            weights[f] = round(1.0 + WEAK_BOOST * (1.0 - rate), 2)
        return weights

    def mastered(self):
        return [f for f, rounds in self.all_rounds.items() if finished(self.levels, f) and len(rounds) == WINDOW and sum(rounds) / WINDOW >= MASTERED_AT]

    def total(self, family=None):
        families = [family] if family else FAMILIES
        return sum(sum(self.levels[f].values()) for f in families)

    @staticmethod
    def most(family=None):
        families = [family] if family else FAMILIES
        return sum(len(levels) - 1 for f in families for levels in FAMILIES[f].values())


def is_open(levels, family):
    needs = FAMILY_NEEDS.get(family)
    return not needs or all(levels[needs[0]][d] >= n for d, n in needs[1].items())


def open_dials(levels, family):
    """Dials that can still go up."""
    dials = []
    for dial, values in FAMILIES[family].items():
        if dial in EXPERT.get(family, ()) and not base_finished(levels, family):
            continue
        needs = DIAL_NEEDS.get((family, dial))
        if needs and levels[family][needs[0]] < needs[1]:
            continue
        dials.append(dial)
    return dials


def pick_family(levels, rng, mastered=(), practice=None):
    """A course kind at its top and nearly always won gets less practice; one that fails its hardest rounds gets more."""
    names = [f for f in FAMILIES if is_open(levels, f)]
    weights = [FINISHED_WEIGHT if f in mastered else (practice or {}).get(f, 1.0) for f in names]
    return names[int(rng.choice(len(names), p=[w / sum(weights) for w in weights]))]


def base_finished(levels, family):
    return all(levels[family][d] == len(v) - 1 for d, v in FAMILIES[family].items() if d not in EXPERT.get(family, ()))


def finished(levels, family):
    return all(levels[family][d] == len(values) - 1 for d, values in FAMILIES[family].items())


def draw(levels, family, rng, tested=None):
    """One round's settings, and what the round tests: a dial's name, "all", or None."""
    dials = FAMILIES[family]
    current = levels[family]
    chance = rng.random()
    if tested is None:
        near_top = sum(current.values()) / sum(len(v) - 1 for v in dials.values()) >= NEAR_TOP
        share_all = SHARE_ALL_DIALS_NEAR_TOP if near_top else SHARE_ALL_DIALS
        share_one = SHARE_ONE_DIAL_NEAR_TOP if near_top else SHARE_ONE_DIAL
        if chance < share_all:
            tested = "all"
        elif chance < share_all + share_one:
            # Dials at their top are tested too, or a skill that fades could never come back down.
            testable = open_dials(levels, family)
            tested = testable[int(rng.integers(len(testable)))]
    settings = {}
    for dial, values in dials.items():
        at_limit = tested == "all" or tested == dial
        settings[dial] = values[current[dial] if at_limit else int(rng.integers(current[dial] + 1))]
    return settings, tested
