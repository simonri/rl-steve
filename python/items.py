"""What can be in a player's hotbar: every item of Hypixel Bed Wars' item shop, numbered as sim/src/items.rs.

Source: the Hypixel forum guide "Every Item In Bedwars" (ThePrestige, 2020, edited April 2024); the official wiki was
unreachable (docs/training-design.md#foremal). The numbering is fixed and only grows at the end: an item's number is
part of what the model sees. The game reports each slot by its registry name; NAMES maps names to numbers (team-colored
wool, terracotta and stained glass are one kind each, as in the shop). Items the simulator cannot do yet are listed but
never given out.
"""
import numpy as np

KINDS = [
    "empty",
    # blocks
    "wool", "hardened clay", "blast-proof glass", "end stone", "ladder", "oak planks", "obsidian",
    # melee
    "wooden sword", "stone sword", "iron sword", "diamond sword", "knockback stick",
    # tools
    "wooden pickaxe", "iron pickaxe", "golden pickaxe", "diamond pickaxe", "wooden axe", "stone axe", "iron axe", "diamond axe", "shears",
    # ranged
    "bow", "arrow",
    # potions
    "speed potion", "jump potion", "invisibility potion",
    # utility
    "golden apple", "bed bug", "dream defender", "fireball", "tnt", "ender pearl", "water bucket", "bridge egg", "magic milk", "sponge",
    "pop-up tower",
    "other",
]
INDEX = {k: i for i, k in enumerate(KINDS)}
SLOTS = 9
HOTBAR_F = SLOTS * (len(KINDS) + 1) + SLOTS  # items.rs HOTBAR_F

# Blocks the simulator places as full cubes (items.rs placeable): what the courses give out to build with.
BUILD_BLOCKS = ("wool", "hardened clay", "blast-proof glass", "end stone", "oak planks", "obsidian")
SWORD_FOR_DAMAGE = {5: "wooden sword", 6: "stone sword", 7: "iron sword", 8: "diamond sword"}  # env.rs sword_for


def sword_for(damage):
    """env.rs sword_for: the sword for an attack attribute, wood at 5 or less, diamond at 8 or more."""
    return SWORD_FOR_DAMAGE[min(max(int(round(damage)), 5), 8)]

# The game's registry name for each kind given out (the mod's give), and what its names mean when it reports a hotbar.
GIVE = {"wool": "minecraft:white_wool", "hardened clay": "minecraft:terracotta", "blast-proof glass": "minecraft:white_stained_glass",
        "end stone": "minecraft:end_stone", "ladder": "minecraft:ladder", "oak planks": "minecraft:oak_planks", "obsidian": "minecraft:obsidian",
        "wooden sword": "minecraft:wooden_sword", "stone sword": "minecraft:stone_sword", "iron sword": "minecraft:iron_sword",
        "diamond sword": "minecraft:diamond_sword", "knockback stick": "minecraft:stick", "wooden pickaxe": "minecraft:wooden_pickaxe",
        "iron pickaxe": "minecraft:iron_pickaxe", "golden pickaxe": "minecraft:golden_pickaxe", "diamond pickaxe": "minecraft:diamond_pickaxe",
        "wooden axe": "minecraft:wooden_axe", "stone axe": "minecraft:stone_axe", "iron axe": "minecraft:iron_axe", "diamond axe": "minecraft:diamond_axe",
        "shears": "minecraft:shears", "bow": "minecraft:bow", "arrow": "minecraft:arrow", "golden apple": "minecraft:golden_apple",
        "bed bug": "minecraft:snowball", "dream defender": "minecraft:iron_golem_spawn_egg", "fireball": "minecraft:fire_charge", "tnt": "minecraft:tnt",
        "ender pearl": "minecraft:ender_pearl", "water bucket": "minecraft:water_bucket", "bridge egg": "minecraft:egg", "magic milk": "minecraft:milk_bucket",
        "sponge": "minecraft:sponge", "pop-up tower": "minecraft:chest"}
_NAMES = {v: k for k, v in GIVE.items()}


def kind_of(name):
    """The kind (a number) of a registry name the game reports; "other" for anything not in the shop. Potions share
    one name in the game and are "other" until the mod tells them apart."""
    if not name or name == "minecraft:air":
        return INDEX["empty"]
    if name.endswith("_wool"):
        return INDEX["wool"]
    if name.endswith("terracotta"):
        return INDEX["hardened clay"]
    if name.endswith("stained_glass") or name == "minecraft:glass":
        return INDEX["blast-proof glass"]
    return INDEX[_NAMES[name]] if name in _NAMES else INDEX["other"]


def placeable(kind):
    return KINDS[kind] in BUILD_BLOCKS


def is_sword(kind):
    return KINDS[kind].endswith("sword")


def layout(rng, blocks, block_kind, sword_kind, hold):
    """A round's hotbar: the blocks (if any) and the sword in two different random slots, the rest empty, and the slot
    selected (hold: "blocks" or "sword"). Random slots, so the model has to look at its hotbar to find things."""
    slots = [(0, 0)] * SLOTS
    where = rng.permutation(SLOTS)
    b, s = int(where[0]), int(where[1])
    if blocks > 0:
        slots[b] = (INDEX[block_kind], int(blocks))
    slots[s] = (INDEX[sword_kind], 1)
    if hold == "random" and blocks > 0:  # any slot but the blocks': the sword or an empty one
        selected = int(rng.choice([i for i in range(SLOTS) if i != b]))
    else:
        selected = b if hold == "blocks" and blocks > 0 else s if hold == "sword" else b
    return slots, int(selected)


def features(hotbar, selected):
    """items.rs features, for checks: one-hot kind and count/64 per slot, then the selected slot."""
    out = np.zeros(HOTBAR_F, np.float32)
    for i, (kind, count) in enumerate(hotbar):
        o = out[i * (len(KINDS) + 1) : (i + 1) * (len(KINDS) + 1)]
        if count > 0 and kind != 0:
            o[min(kind, len(KINDS) - 1)] = 1.0
            o[len(KINDS)] = count / 64.0
        else:
            o[0] = 1.0
    out[SLOTS * (len(KINDS) + 1) + min(selected, SLOTS - 1)] = 1.0
    return out
