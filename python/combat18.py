"""Combat as in 1.8.9 (Hypixel), the reference both the simulator (sim/src/combat.rs) and the server mod follow.

Our players run 26.3 clients, so what the client decides stays 26.3: which entity the crosshair is on (eyes along the
view, 3 blocks, hitbox not inflated: Entity.getPickRadius is 0), and a click that hits nothing at all locks attacks for
10 ticks (Minecraft.missTime; a click on a block does not). With no attack cooldown (the server sets attack speed so
high the charge is always full), 26.3's own client side of an attack matches 1.8: a sprinting hit slows the attacker
to 0.6 and stops its sprint.

What the server decides follows 1.8.9 (MCP names; line numbers from the 1.8.9 source):
- EntityPlayer.attackTargetEntityWithCurrentItem (1305): damage = attack attribute (1 + sword), x1.5 when the attacker
  is falling (fallDistance > 0, not on the ground), plus Sharpness 1.25 a level. Knockback level = enchant + 1 if
  sprinting. When the hit counts and that level > 0: the target gets (-sin(yaw) * level * 0.5, 0.1, cos(yaw) * level *
  0.5) added, from the attacker's yaw.
- EntityLivingBase.attackEntityFrom (863): with hurtResistantTime above 10 (of 20) a hit only counts if it is bigger
  than the last one, and then only the difference is dealt, with no knockback. Otherwise the full hit, 20 ticks of
  resistance, and knockBack.
- EntityLivingBase.knockBack (1076): motion halved, then 0.4 away from the attacker and 0.4 up, y capped at 0.4.
- EntityLivingBase.applyArmorCalculations (1215): damage * (25 - armor points) / 25.
- FoodStats: with full food, one health point every 80 ticks (1.8 has no fast saturation healing).
The server's own motion for a player it does not move itself is taken as 0, as for fall damage in the simulator.
"""
import math

ATTACK_BASE = 1.0  # EntityPlayer: attackDamage base value
SWORDS = {"hand": 0.0, "wood": 4.0, "stone": 5.0, "iron": 6.0, "diamond": 7.0}  # ItemSword: 4 + material
ARMOR = {"leather": (1, 3, 2, 1), "chain": (2, 5, 4, 1), "iron": (2, 6, 5, 2), "diamond": (3, 8, 6, 3)}  # helmet, chest, legs, boots
MAX_RESIST = 20  # maxHurtResistantTime
HEAL_EVERY = 80  # FoodStats.foodTimer
REACH = 3.0  # 26.3 entity interaction range
MISS_LOCK = 10  # 26.3 Minecraft.missTime after a click on nothing
KNOCK = 0.4
SPRINT_KNOCK = 0.5
DEG = math.pi / 180


def f32(x):
    """Round to a Java float."""
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def bedwars_armor(tier):
    """Bedwars keeps the team's leather helmet and chestplate; boots and leggings are what was bought."""
    leather = ARMOR["leather"]
    bought = ARMOR[tier]
    return leather[0] + leather[1] + bought[2] + bought[3]


def hit_damage(sword, falling, sharpness=0):
    dmg = f32(ATTACK_BASE + SWORDS[sword])
    if falling:
        dmg = f32(dmg * 1.5)
    return f32(dmg + sharpness * 1.25)


def take_hit(victim, amount, armor_points):
    """attackEntityFrom with damageEntity. victim: dict with health, resist, last. Returns (counted, knock)."""
    if victim["health"] <= 0:
        return False, False
    if victim["resist"] > MAX_RESIST / 2:
        if amount <= victim["last"]:
            return False, False
        dealt = amount - victim["last"]
        victim["last"] = amount
        knock = False
    else:
        victim["last"] = amount
        victim["resist"] = MAX_RESIST
        dealt = amount
        knock = True
    dealt = f32(dealt * (25 - armor_points) / 25.0)
    victim["health"] = f32(victim["health"] - dealt)
    return True, knock


def knockback(motion, dx, dz):
    """knockBack(attacker, amount, dx = attacker.x - victim.x, dz = attacker.z - victim.z) on the victim's motion."""
    f = f32(math.sqrt(dx * dx + dz * dz))
    mx, my, mz = motion[0] / 2.0, motion[1] / 2.0, motion[2] / 2.0
    mx -= dx / f * KNOCK
    my += KNOCK
    mz -= dz / f * KNOCK
    if my > 0.4000000059604645:
        my = 0.4000000059604645
    return [mx, my, mz]


def sprint_knock(motion, yaw, level):
    """The extra from a sprinting (or Knockback-enchanted) hit, from the attacker's yaw."""
    s = math.sin(f32(yaw * f32(math.pi) / 180.0))
    c = math.cos(f32(yaw * f32(math.pi) / 180.0))
    return [motion[0] + f32(-s * level * 0.5), motion[1] + 0.1, motion[2] + f32(c * level * 0.5)]


def attack(attacker, victim, sword="wood", sharpness=0, kb_enchant=0, armor_points=7):
    """A hit from attacker on victim, as the server works it out. Both are dicts with pos, yaw, sprinting, falling;
    the victim also health, resist, last. Returns the victim's new motion (None when it does not change) and whether
    the attacker's sprint was broken."""
    amount = hit_damage(sword, attacker["falling"], sharpness)
    level = kb_enchant + (1 if attacker["sprinting"] else 0)
    counted, knock = take_hit(victim, amount, armor_points)
    motion = None
    if counted and knock:
        dx = attacker["pos"][0] - victim["pos"][0]
        dz = attacker["pos"][2] - victim["pos"][2]
        if dx * dx + dz * dz < 1.0e-4:
            dx, dz = 0.0, 0.01  # 1.8 draws a random tiny direction here; a fixed one keeps both sides equal
        motion = knockback([0.0, 0.0, 0.0], dx, dz)
    if counted and level > 0:
        motion = sprint_knock(motion or [0.0, 0.0, 0.0], attacker["yaw"], level)
    return motion, counted and level > 0


def duel_gear(choice):
    """(attack attribute, armor points) for the model and its opponents, by the duel course's gear dial."""
    tiers = {"wood": ("wood", "leather"), "stone": ("stone", "leather"), "iron": ("iron", "iron")}
    if choice == "theirs better":
        return (ATTACK_BASE + SWORDS["stone"], bedwars_armor("leather")), (ATTACK_BASE + SWORDS["iron"], bedwars_armor("iron"))
    sword, armor = tiers[choice]
    gear = (ATTACK_BASE + SWORDS[sword], bedwars_armor(armor))
    return gear, gear


def vectors():
    """Cases with their answers, for the simulator's and the mod's tests."""
    out = []
    for sword in ("hand", "wood", "iron", "diamond"):
        for falling in (False, True):
            for sprinting in (False, True):
                for armor in ("leather", "iron", "diamond"):
                    v = dict(pos=[0.0, 100.0, 0.0], health=20.0, resist=0, last=0.0)
                    a = dict(pos=[1.3, 100.0, 0.4], yaw=-72.0, sprinting=sprinting, falling=falling)
                    pts = bedwars_armor(armor)
                    motion, broke = attack(a, v, sword, armor_points=pts)
                    out.append(dict(sword=sword, falling=falling, sprinting=sprinting, armor=pts, health=v["health"], motion=motion, broke=broke))
    # resistance: a second, weaker hit 5 ticks later does nothing; a stronger one deals only the difference
    v = dict(pos=[0.0, 100.0, 0.0], health=20.0, resist=0, last=0.0)
    take_hit(v, 5.0, 0)
    v["resist"] -= 5
    weaker = take_hit(v, 4.0, 0)
    stronger = take_hit(v, 7.5, 0)
    out.append(dict(resist_case=True, weaker=weaker[0], stronger=stronger[0], health=v["health"]))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(vectors(), indent=1)[:2000])
