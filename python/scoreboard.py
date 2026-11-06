"""Colored, compact scoreboard lines. Green is good, yellow is close, red needs work."""
GOOD, CLOSE, BAD = "green", "yellow", "red"
LABEL, VALUE, DIM = "gray", "white", "dark_gray"


def compact(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1_000:.0f}k" if n >= 1_000 else f"{n:.0f}"


def higher_is_better(value, good, close):
    return GOOD if value >= good else CLOSE if value >= close else BAD


def lower_is_better(value, good, close):
    return GOOD if value <= good else CLOSE if value <= close else BAD


def stage_bar(stage, stages, cells=14):
    done = round(cells * stage / stages)
    return [["■" * done, "aqua"], ["■" * (cells - done), DIM], [f" {stage}/{stages}", VALUE, True]]


def extras_text(extra):
    names = {"island": "stepping stone", "pillar": "pillar", "wall": "wall", "slalom": "slalom", "mixed": "mixed", "jump": "jump, no blocks", "treenan": "Treenan map", "climb": "climbing", "hops": "single blocks", "islands": "stop on blocks", "wide_zigzag": "zigzag with holes", "run": "narrow path", "run_holes": "path with holes"}
    return names.get(extra, "")


def task_line(gap, height_low, height_high, extra):
    line = [["Gap ", LABEL], [str(gap), VALUE]]
    if height_low != 0 or height_high != 0:
        line += [["  Height ", LABEL], [f"{height_low:+d}..{height_high:+d}", VALUE]]
    if extra:
        line += [["  ", LABEL], [extras_text(extra), "light_purple"]]
    return line


def stat(label, value, color=VALUE, note=None):
    line = [[f"{label} ", LABEL], [value, color]]
    if note:
        line.append([f" {note}", DIM])
    return line


BLANK = [[" ", DIM]]
