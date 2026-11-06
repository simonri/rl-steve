"""Reads the blocks out of a Minecraft world folder (Anvil region files) without running the game."""
import gzip
import io
import json
import struct
import sys
import zlib
from pathlib import Path


def _read_tag(f, kind):
    if kind == 1:
        return struct.unpack(">b", f.read(1))[0]
    if kind == 2:
        return struct.unpack(">h", f.read(2))[0]
    if kind == 3:
        return struct.unpack(">i", f.read(4))[0]
    if kind == 4:
        return struct.unpack(">q", f.read(8))[0]
    if kind == 5:
        return struct.unpack(">f", f.read(4))[0]
    if kind == 6:
        return struct.unpack(">d", f.read(8))[0]
    if kind == 7:
        n = struct.unpack(">i", f.read(4))[0]
        return f.read(n)
    if kind == 8:
        n = struct.unpack(">H", f.read(2))[0]
        return f.read(n).decode("utf-8", "replace")
    if kind == 9:
        inner = f.read(1)[0]
        n = struct.unpack(">i", f.read(4))[0]
        return [_read_tag(f, inner) for _ in range(max(n, 0))]
    if kind == 10:
        out = {}
        while True:
            inner = f.read(1)[0]
            if inner == 0:
                return out
            n = struct.unpack(">H", f.read(2))[0]
            name = f.read(n).decode("utf-8", "replace")
            out[name] = _read_tag(f, inner)
    if kind == 11:
        n = struct.unpack(">i", f.read(4))[0]
        return list(struct.unpack(f">{n}i", f.read(4 * n)))
    if kind == 12:
        n = struct.unpack(">i", f.read(4))[0]
        return list(struct.unpack(f">{n}q", f.read(8 * n)))
    raise ValueError(f"unknown NBT tag {kind}")


def read_nbt(data):
    f = io.BytesIO(data)
    kind = f.read(1)[0]
    n = struct.unpack(">H", f.read(2))[0]
    f.read(n)
    return _read_tag(f, kind)


def chunks(region):
    raw = region.read_bytes()
    for i in range(1024):
        entry = struct.unpack(">I", raw[i * 4 : i * 4 + 4])[0]
        offset, count = entry >> 8, entry & 0xFF
        if not offset or not count:
            continue
        start = offset * 4096
        length = struct.unpack(">I", raw[start : start + 4])[0]
        scheme = raw[start + 4]
        body = raw[start + 5 : start + 4 + length]
        yield read_nbt(zlib.decompress(body) if scheme == 2 else gzip.decompress(body))


def blocks(world):
    """Yields (x, y, z, name, properties) for every non-air block."""
    for region in sorted(Path(world, "region").glob("*.mca")):
        for chunk in chunks(region):
            level = chunk.get("Level", chunk)
            cx, cz = level["xPos"], level["zPos"]
            for section in level.get("Sections", level.get("sections", [])):
                if "Palette" in section:
                    palette, states = section["Palette"], section.get("BlockStates")
                elif "block_states" in section:
                    palette, states = section["block_states"].get("palette", []), section["block_states"].get("data")
                else:
                    continue
                if not palette:
                    continue
                sy = section["Y"]
                bits = max(4, (len(palette) - 1).bit_length())
                per = 64 // bits
                mask = (1 << bits) - 1
                for index in range(4096):
                    if states:
                        value = (states[index // per] & 0xFFFFFFFFFFFFFFFF) >> ((index % per) * bits) & mask
                    else:
                        value = 0
                    state = palette[value] if value < len(palette) else palette[0]
                    name = state["Name"]
                    if name.endswith(":air"):
                        continue
                    yield (cx * 16 + (index & 15), sy * 16 + (index >> 8 & 15), cz * 16 + (index >> 4 & 15), name, state.get("Properties", {}))


if __name__ == "__main__":
    world, out = sys.argv[1], sys.argv[2]
    level = gzip.decompress(Path(world, "level.dat").read_bytes())
    data = read_nbt(level)["Data"]
    print("version", data.get("Version", {}).get("Name"), "data", data.get("DataVersion"), "name", data.get("LevelName"), "spawn", data.get("SpawnX"), data.get("SpawnY"), data.get("SpawnZ"))
    found = [(x, y, z, name, props) for x, y, z, name, props in blocks(world)]
    Path(out).write_text(json.dumps(found))
    print(len(found), "blocks")
