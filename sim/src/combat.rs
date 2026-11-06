//! Players hitting players, as on Hypixel: 26.3 clients deciding what the crosshair is on, a server with 1.8.9's rules
//! (python/combat18.py has the source lines and the same numbers, sim/combat18_vectors.json the cases both must match).
//!
//! A tick with several players goes: each client turns, picks its hotbar slot and, on a left click, works out what it
//! aims at with everyone where they are now ([`aim`]); then every player moves; then the server handles the hits
//! with the places it knew when the clicks came ([`hit`]). A hit's motion reaches the victim before its next tick.

use crate::player::{clip_box, Player, REACH, WIDTH};
use crate::world::{Aabb, World};

pub const ENTITY_REACH: f64 = 3.0; // Attributes.ENTITY_INTERACTION_RANGE for a player
pub const MISS_LOCK: i32 = 10; // Minecraft.missTime after a click on nothing
const MAX_RESIST: i32 = 20; // maxHurtResistantTime
const KNOCK: f64 = 0.4;

/// What a left click lands on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Aim {
    Player(usize),
    Block,
    Nothing,
}

/// A player's box where it stands now.
pub fn body(p: &Player) -> Aabb {
    let h = p.height();
    let w = WIDTH / 2.0;
    Aabb::new(p.pos[0] - w, p.pos[1], p.pos[2] - w, p.pos[0] + w, p.pos[1] + h, p.pos[2] + w)
}

/// LocalPlayer.pick: the nearer of the block in reach (4.5) and a player's box on the line (3), as the client sees it.
/// `others` are the other players' boxes, `None` for one that cannot be hit (itself, dead, gone).
pub fn aim(p: &Player, world: &World, others: &[Option<Aabb>]) -> Aim {
    let eye = [p.pos[0], p.pos[1] + p.eye_height(), p.pos[2]];
    let v = p.view();
    let block = p.pick(world);
    let block_d2 = block.map(|h| {
        let d = [h.loc[0] - eye[0], h.loc[1] - eye[1], h.loc[2] - eye[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    });
    let max_d = block_d2.map_or(REACH, f64::sqrt);
    let mut nearest: Option<(usize, f64)> = None;
    for (i, b) in others.iter().enumerate() {
        let Some(b) = b else { continue };
        let inside = b.min[0] <= eye[0] && eye[0] <= b.max[0] && b.min[1] <= eye[1] && eye[1] <= b.max[1] && b.min[2] <= eye[2] && eye[2] <= b.max[2];
        let d2 = if inside {
            0.0
        } else {
            match clip_box(b, eye, v, max_d) {
                Some((t, _)) => t * t,
                None => continue,
            }
        };
        if nearest.map_or(true, |(_, best)| d2 < best) {
            nearest = Some((i, d2));
        }
    }
    match (nearest, block_d2) {
        (Some((i, d2)), b) if b.map_or(true, |b| d2 < b) => {
            if d2 < ENTITY_REACH * ENTITY_REACH {
                Aim::Player(i)
            } else {
                Aim::Nothing // past 3 blocks: filterHitResult makes it a miss
            }
        }
        (_, Some(_)) => Aim::Block,
        _ => Aim::Nothing,
    }
}

/// A left click: hits whoever `aim` finds, or locks attacks for 10 ticks when it finds nothing. Returns the target.
pub fn click(p: &mut Player, world: &World, others: &[Option<Aabb>]) -> Option<usize> {
    if p.miss_lock > 0 {
        return None;
    }
    match aim(p, world, others) {
        Aim::Player(i) => Some(i),
        Aim::Block => None, // starts breaking it; blocks do not break in these rounds
        Aim::Nothing => {
            p.miss_lock = MISS_LOCK;
            None
        }
    }
}

/// The attacker as the server knew it when the click came.
#[derive(Clone, Copy, Debug)]
pub struct Swing {
    pub pos: [f64; 3],
    pub yaw: f32,
    pub falling: bool,
    pub sprinting: bool, // the server's sprint flag
    pub damage: f32,     // attack attribute
    pub kb_enchant: i32,
}

impl Swing {
    pub fn of(p: &Player) -> Swing {
        Swing {
            pos: p.pos,
            yaw: p.yaw,
            falling: p.fall_distance > 0.0 && !p.on_ground,
            sprinting: p.server_sprinting,
            damage: p.attack_damage,
            kb_enchant: p.kb_enchant,
        }
    }
}

/// The server's side of a hit (EntityPlayer.attackTargetEntityWithCurrentItem, EntityLivingBase.attackEntityFrom,
/// knockBack, applyArmorCalculations in 1.8.9). `victim_pos` is where the server had the victim. Returns whether the
/// hit counted and whether it broke the attacker's sprint (the caller clears that attacker's server sprint flag).
pub fn hit(s: &Swing, victim: &mut Player, victim_pos: [f64; 3]) -> (bool, bool) {
    if victim.health <= 0.0 {
        return (false, false);
    }
    let mut amount = s.damage;
    if s.falling {
        amount *= 1.5;
    }
    let level = s.kb_enchant + s.sprinting as i32;
    // attackEntityFrom: inside the second half of the resistance only a bigger hit counts, by the difference.
    let (dealt, knock) = if victim.resist as f32 > MAX_RESIST as f32 / 2.0 {
        if amount <= victim.last_damage {
            return (false, false);
        }
        let d = amount - victim.last_damage;
        victim.last_damage = amount;
        (d, false)
    } else {
        victim.last_damage = amount;
        victim.resist = MAX_RESIST;
        victim.hurt_time = 10;
        (amount, true)
    };
    let dealt = dealt * (25 - victim.armor) as f32 / 25.0;
    victim.health = (victim.health - dealt).max(0.0);
    let mut motion: Option<[f64; 3]> = None;
    if knock {
        let (mut dx, mut dz) = (s.pos[0] - victim_pos[0], s.pos[2] - victim_pos[2]);
        if dx * dx + dz * dz < 1.0e-4 {
            (dx, dz) = (0.0, 0.01); // 1.8 draws a random tiny direction; a fixed one keeps the game and this equal
        }
        let f = ((dx * dx + dz * dz) as f32).sqrt() as f64;
        let m = victim.server_motion;
        let mut v = [m[0] / 2.0 - dx / f * KNOCK, m[1] / 2.0 + KNOCK, m[2] / 2.0 - dz / f * KNOCK];
        if v[1] > 0.4000000059604645 {
            v[1] = 0.4000000059604645;
        }
        motion = Some(v);
    }
    if level > 0 {
        let r = s.yaw * std::f32::consts::PI / 180.0;
        let base = motion.unwrap_or(victim.server_motion);
        motion = Some([
            base[0] + (-r.sin() * level as f32 * 0.5) as f64,
            base[1] + 0.1,
            base[2] + (r.cos() * level as f32 * 0.5) as f64,
        ]);
    }
    if motion.is_some() {
        victim.pending_knock = motion;
    }
    (true, level > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::FULL;

    fn victim() -> Player {
        let mut v = Player::new([0.0, 100.0, 0.0], 0.0, 0.0, 0);
        v.health = 20.0;
        v
    }

    fn swing(sword: f32, falling: bool, sprinting: bool) -> Swing {
        Swing { pos: [1.3, 100.0, 0.4], yaw: -72.0, falling, sprinting, damage: 1.0 + sword, kb_enchant: 0 }
    }

    /// The same cases as python/combat18.py vectors(): health and motion after one hit.
    #[test]
    fn matches_the_reference() {
        let text = include_str!("../combat18_vectors.json");
        let mut checked = 0;
        for case in text.split('{').skip(1) {
            if case.contains("resist_case") {
                continue;
            }
            let field = |k: &str| -> &str {
                let i = case.find(&format!("\"{k}\":")).unwrap() + k.len() + 3;
                case[i..].split(|c| c == ',' || c == '}').next().unwrap().trim()
            };
            let sword = match field("sword").trim_matches('"') {
                "hand" => 0.0,
                "wood" => 4.0,
                "stone" => 5.0,
                "iron" => 6.0,
                _ => 7.0,
            };
            let mut v = victim();
            v.armor = field("armor").parse().unwrap();
            let s = swing(sword, field("falling") == "true", field("sprinting") == "true");
            let (counted, broke) = { let at = v.pos; hit(&s, &mut v, at) };
            assert!(counted);
            assert_eq!(broke, field("broke") == "true");
            let health: f32 = field("health").parse().unwrap();
            assert!((v.health - health).abs() < 1e-5, "{} vs {}", v.health, health);
            let i = case.find("\"motion\":").unwrap() + 10;
            let nums: Vec<f64> = case[i..].split(']').next().unwrap().trim_start_matches('[').split(',').map(|x| x.trim().parse().unwrap()).collect();
            let m = v.pending_knock.unwrap();
            for k in 0..3 {
                assert!((m[k] - nums[k]).abs() < 1e-6, "motion {:?} vs {:?}", m, nums);
            }
            checked += 1;
        }
        assert_eq!(checked, 48);
    }

    #[test]
    fn resistance() {
        let mut v = victim();
        let s = |d: f32| Swing { pos: [1.0, 100.0, 0.0], yaw: 0.0, falling: false, sprinting: false, damage: d, kb_enchant: 0 };
        assert!({ let at = v.pos; hit(&s(5.0), &mut v, at) }.0);
        v.resist -= 5;
        assert!(!{ let at = v.pos; hit(&s(4.0), &mut v, at) }.0, "a weaker hit inside the resistance does nothing");
        assert!({ let at = v.pos; hit(&s(7.5), &mut v, at) }.0);
        assert!((v.health - 12.5).abs() < 1e-6, "only the difference counts: {}", v.health);
        v.resist = 10;
        assert!({ let at = v.pos; hit(&s(1.0), &mut v, at) }.0, "from the second half on, any hit counts in full");
    }

    #[test]
    fn aim_and_miss_lock() {
        let mut w = World::new([-10, 90, -10], [10, 110, 10]);
        w.fill([-10, 99, -10], [10, 99, 10], FULL);
        // Looking along +z (yaw 0) at a player 2.5 ahead: hit. 3.5 ahead: nothing in reach, and the lock starts.
        let mut p = Player::new([0.5, 100.0, 0.5], 0.0, 0.0, 0);
        let near = Player::new([0.5, 100.0, 3.0], 0.0, 0.0, 0);
        let far = Player::new([0.5, 100.0, 4.2], 0.0, 0.0, 0);
        assert_eq!(click(&mut p, &w, &[Some(body(&near))]), Some(0));
        assert_eq!(click(&mut p, &w, &[Some(body(&far))]), None);
        assert_eq!(p.miss_lock, MISS_LOCK);
        assert_eq!(click(&mut p, &w, &[Some(body(&near))]), None, "locked");
        // Looking at the ground in front: a block, no lock.
        let mut q = Player::new([0.5, 100.0, 0.5], 0.0, 60.0, 0);
        assert_eq!(click(&mut q, &w, &[None]), None);
        assert_eq!(q.miss_lock, 0);
        // A wall between: the block is nearer, so no hit.
        w.set(0, 101, 2, FULL);
        let mut r = Player::new([0.5, 100.0, 0.5], 0.0, 0.0, 0);
        assert_eq!(aim(&r, &w, &[Some(body(&near))]), Aim::Block);
        r.miss_lock = 0;
    }
}
