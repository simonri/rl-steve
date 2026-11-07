//! Fighting rounds: what the model sees of the players around it, and the scripted opponents it trains against.
//!
//! Both are used by the simulator's rounds and, through python.rs, by the Python side that plays the real game, so the
//! two always agree. Like clutch.rs, what depends on blocks reads only the cube 5 around the feet the model sees.
//!
//! The model's input grows by [`EXTRA`] values after the 4681 of the movement model: the three nearest opponents, its
//! own fighting state, four values for the command a strategist will give, and its last attack and hotbar inputs.

use crate::clutch::Cube;
use crate::combat::{self, Aim, ENTITY_REACH};
use crate::env::{rotate, HISTORY, TURN_STEPS};
use crate::player::{Player, STAND_EYES};

pub const ENEMIES: usize = 3;
pub const ENEMY_F: usize = 20;
pub const SELF_F: usize = 10;
pub const CMD_F: usize = 4;
pub const HIST_F: usize = 2; // attack, hotbar choice, per remembered tick
pub const EXTRA: usize = ENEMIES * ENEMY_F + SELF_F + CMD_F + HIST_F * HISTORY;
pub const ACTIONS: usize = 10; // the movement model's 8, then attack (0, 1) and hotbar (0 keep, 1 blocks, 2 sword)
/// How many choices each action has (AutoEnv.action_space): move, strafe, sneak, use, jump, sprint, yaw, pitch, attack, hotbar.
pub const CHOICES: [i64; ACTIONS] = [3, 3, 2, 2, 2, 2, 11, 11, 2, 3];

/// A player as the game shows it to someone else.
#[derive(Clone, Copy, Debug, Default)]
pub struct View {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub health: f32,
    pub armor: i32,
    pub damage: f32, // attack attribute of what it holds
    pub hurt_time: i32,
    pub on_ground: bool,
    pub sprinting: bool,
    pub crouching: bool,
}

impl View {
    pub fn of(p: &Player) -> View {
        View {
            pos: p.pos,
            vel: p.dm,
            yaw: p.yaw,
            pitch: p.pitch,
            health: p.shown_health(), // an opponent's is the server's (its unseen is always 0)
            armor: p.armor,
            damage: if p.slot == 1 { p.attack_damage } else { 1.0 },
            hurt_time: p.hurt_time,
            on_ground: p.on_ground,
            sprinting: p.sprinting,
            crouching: p.crouch_pose,
        }
    }

    fn eye(&self) -> [f64; 3] {
        [self.pos[0], self.pos[1] + if self.crouching { 1.27 } else { STAND_EYES }, self.pos[2]]
    }
}

/// The model's own fighting state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Me {
    pub slot: u8,
    pub miss_lock: i32,
    pub resist: i32,
    pub damage: f32,
    pub armor: i32,
    pub server_sprinting: bool,
    pub since_hit: i32,  // ticks since one of its hits counted
    pub since_hurt: i32, // ticks since it was hit
    pub duel: bool,
}

/// The values after the movement model's 4681: opponents nearest first, then itself, the command, the history.
/// `me_player` is the model's own player (position, view and aim), `history` its last attack and hotbar inputs.
/// `cube` makes the blocks around it, only asked for when there is someone to see.
pub fn features(me_player: &Player, me: &Me, enemies: &[View], k: i32, cube: &dyn Fn() -> Cube, history: &[[f32; HIST_F]; HISTORY], out: &mut [f32]) {
    debug_assert_eq!(out.len(), EXTRA);
    out.fill(0.0);
    let [x, y, z] = me_player.pos;
    let mut order: Vec<usize> = (0..enemies.len()).filter(|&i| enemies[i].health > 0.0).collect();
    order.sort_by(|&a, &b| dist2(&enemies[a].pos, &me_player.pos).total_cmp(&dist2(&enemies[b].pos, &me_player.pos))); // NaN-safe, stable for ties
    // What the crosshair is on, only when someone is there to aim at: building the cube's world is most of the cost,
    // and outside fights every tick would pay it for nothing.
    let cube = if order.is_empty() { None } else { Some(cube()) };
    let aimed = match &cube {
        None => Aim::Nothing,
        Some(cube) => {
            let boxes: Vec<Option<_>> = enemies.iter().map(|e| if e.health > 0.0 { Some(body(e)) } else { None }).collect();
            combat::aim(me_player, &cube.world(), &boxes)
        }
    };
    let my_eye = me_player.eye();
    for (slot, &i) in order.iter().take(ENEMIES).enumerate() {
        let e = &enemies[i];
        let o = &mut out[slot * ENEMY_F..(slot + 1) * ENEMY_F];
        let (dx, dy, dz) = (e.pos[0] - x, e.pos[1] - y, e.pos[2] - z);
        let (rx, rz) = rotate(dx, dz, k);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        let (vx, vz) = rotate(e.vel[0], e.vel[2], k);
        // Where it looks, against the way from it to the model: 1 when it looks straight at it.
        let to_me = (-dz).atan2(-dx);
        let its = (e.yaw as f64 + 90.0).to_radians();
        let off = its - to_me;
        let reach = box_distance(&body(e), &my_eye) < ENTITY_REACH;
        o[0] = 1.0;
        o[1] = (rx / 8.0) as f32;
        o[2] = (dy / 4.0) as f32;
        o[3] = (rz / 8.0) as f32;
        o[4] = (d.ln_1p() / 3.0) as f32;
        o[5] = (vx * 5.0) as f32;
        o[6] = (vz * 5.0) as f32;
        o[7] = (e.vel[1] * 2.0) as f32;
        o[8] = off.sin() as f32;
        o[9] = off.cos() as f32;
        o[10] = e.pitch / 90.0;
        o[11] = e.health / 20.0;
        o[12] = e.armor as f32 / 20.0;
        o[13] = e.damage / 8.0;
        o[14] = e.hurt_time as f32 / 10.0;
        o[15] = e.on_ground as i32 as f32;
        o[16] = e.sprinting as i32 as f32;
        o[17] = (aimed == Aim::Player(i)) as i32 as f32;
        o[18] = reach as i32 as f32;
        o[19] = cube.as_ref().is_some_and(|c| clear(c, my_eye, e.eye())) as i32 as f32;
    }
    let s = &mut out[ENEMIES * ENEMY_F..ENEMIES * ENEMY_F + SELF_F];
    s[0] = (me.slot == 0) as i32 as f32;
    s[1] = (me.slot == 1) as i32 as f32;
    s[2] = me.miss_lock as f32 / 10.0;
    s[3] = me.resist as f32 / 20.0;
    s[4] = me.damage / 8.0;
    s[5] = me.armor as f32 / 20.0;
    s[6] = me.server_sprinting as i32 as f32;
    s[7] = (me.since_hit.min(20) as f32) / 20.0;
    s[8] = (me.since_hurt.min(20) as f32) / 20.0;
    s[9] = me.duel as i32 as f32;
    // the command (CMD_F) stays 0 until there is a strategist
    let h = &mut out[ENEMIES * ENEMY_F + SELF_F + CMD_F..];
    for (t, v) in history.iter().enumerate() {
        h[t * HIST_F..(t + 1) * HIST_F].copy_from_slice(v);
    }
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn body(e: &View) -> crate::world::Aabb {
    let h = if e.crouching { 1.5 } else { 1.8 };
    crate::world::Aabb::new(e.pos[0] - 0.3, e.pos[1], e.pos[2] - 0.3, e.pos[0] + 0.3, e.pos[1] + h, e.pos[2] + 0.3)
}

fn box_distance(b: &crate::world::Aabb, p: &[f64; 3]) -> f64 {
    let mut d2 = 0.0;
    for a in 0..3 {
        let v = if p[a] < b.min[a] { b.min[a] - p[a] } else if p[a] > b.max[a] { p[a] - b.max[a] } else { 0.0 };
        d2 += v * v;
    }
    d2.sqrt()
}

/// No block of the cube on the line between two eyes. Beyond the cube nothing is known, and counts as clear.
fn clear(cube: &Cube, a: [f64; 3], b: [f64; 3]) -> bool {
    let n = ((dist2(&a, &b).sqrt() * 4.0) as usize).max(2);
    (1..n).all(|i| {
        let t = i as f64 / n as f64;
        !cube.at_world([(a[0] + (b[0] - a[0]) * t).floor() as i32, (a[1] + (b[1] - a[1]) * t).floor() as i32, (a[2] + (b[2] - a[2]) * t).floor() as i32])
    })
}

// ---- Scripted opponents ----

/// The opponents the duel course's dial goes through, easiest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bot {
    Dummy,   // stands still
    Walker,  // walks up and hits five times a second
    Strafer, // circles while sprinting, hits ten times a second
    Combo,   // also lets go of sprint after each hit (a w-tap) for the full extra knockback, and jumps for crits
    Pusher,  // like Combo, and keeps itself on the far side from the void
}

impl Bot {
    pub fn from_index(i: i32) -> Bot {
        match i {
            0 => Bot::Dummy,
            1 => Bot::Walker,
            2 => Bot::Strafer,
            3 => Bot::Combo,
            _ => Bot::Pusher,
        }
    }
}

/// The turn step (an index into TURN_STEPS) closest to what is left to turn, so a bot turns no faster than the model.
fn step_toward(left: f64) -> i64 {
    let mut best = 5;
    for (i, &s) in TURN_STEPS.iter().enumerate() {
        if (s - left).abs() < (TURN_STEPS[best] - left).abs() {
            best = i;
        }
    }
    best as i64
}

/// A bot's inputs for this tick, in the model's own action space (so it gets the same key and click limits).
/// `me` is the bot, `target` the one it fights, `centre` the middle of the arena (the side away from the void), `t`
/// the tick, `since_hit` ticks since one of its hits counted, `ground` whether a block is at a spot (the floor it
/// must not walk off).
pub fn action(bot: Bot, me: &View, target: &View, centre: [f64; 3], t: i32, since_hit: i32, ground: &dyn Fn(i32, i32, i32) -> bool) -> [i64; ACTIONS] {
    let mut a = [0, 0, 0, 0, 0, 0, 5, 5, 0, 2]; // nothing pressed, no turn, sword in hand
    if bot == Bot::Dummy || target.health <= 0.0 {
        return a;
    }
    let eye = me.eye();
    let aim_at = [target.pos[0], target.pos[1] + 1.2, target.pos[2]];
    let (dx, dy, dz) = (aim_at[0] - eye[0], aim_at[1] - eye[1], aim_at[2] - eye[2]);
    let flat = dx.hypot(dz);
    let want_yaw = (-dx).atan2(dz).to_degrees();
    let want_pitch = (-dy).atan2(flat).to_degrees();
    let dyaw = (want_yaw - me.yaw as f64 + 540.0).rem_euclid(360.0) - 180.0;
    a[6] = step_toward(dyaw);
    a[7] = step_toward(want_pitch - me.pitch as f64);
    let aimed = dyaw.abs() < 12.0;
    let close = flat < 2.8;
    let clicks_every = match bot {
        Bot::Walker => 4,
        _ => 2,
    };
    if close && aimed && t % clicks_every == 0 {
        a[8] = 1;
    }
    match bot {
        Bot::Walker => {
            a[0] = if flat > 2.2 { 1 } else { 0 };
        }
        Bot::Strafer => {
            a[0] = if flat > 2.0 { 1 } else { 0 };
            a[1] = if (t / 20) % 2 == 0 { 1 } else { 2 };
            a[5] = 1;
        }
        Bot::Combo | Bot::Pusher => {
            // W-tap: forward and sprint let go for a moment after each hit, so the next one is a fresh sprint hit.
            let tapping = since_hit < 3;
            a[0] = if tapping || flat < 1.6 { 0 } else { 1 };
            a[5] = (!tapping) as i64;
            a[1] = if (t / 15) % 2 == 0 { 1 } else { 2 };
            if close && me.on_ground && t % 10 == 0 {
                a[4] = 1; // a jump, for a falling (critical) hit on the way down
            }
            if bot == Bot::Pusher {
                // Circle toward the side of the target away from the arena's middle: hits then push it outward.
                let (ox, oz) = (target.pos[0] - centre[0], target.pos[2] - centre[2]);
                let (mx, mz) = (me.pos[0] - target.pos[0], me.pos[2] - target.pos[2]);
                let side = ox * mz - oz * mx;
                a[1] = if side > 0.0 { 1 } else { 2 };
            }
        }
        Bot::Dummy => {}
    }
    if !floor_ahead(me, &a, ground) {
        a[0] = 0;
        a[1] = 0;
        a[4] = 0;
        a[5] = 0; // a bot keeps to the arena as a player would: knocked off it may be, it never walks off
    }
    a
}

/// Whether there is floor a step ahead (further when sprinting) in the way the keys `a` move a player at `me`,
/// after this tick's turn; true when the keys do not move it.
fn floor_ahead(me: &View, a: &[i64; ACTIONS], ground: &dyn Fn(i32, i32, i32) -> bool) -> bool {
    let fwd = match a[0] {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    };
    let left = match a[1] {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    };
    if fwd == 0.0 && left == 0.0 {
        return true;
    }
    // Entity.moveRelative: forward is (-sin, cos) of the yaw, left (cos, sin).
    let (sin, cos) = (me.yaw as f64 + TURN_STEPS[a[6] as usize]).to_radians().sin_cos();
    let (vx, vz) = (-sin * fwd + cos * left, cos * fwd + sin * left);
    let n = vx.hypot(vz);
    let reach = if a[5] != 0 { 1.6 } else { 1.0 };
    let (x, z) = ((me.pos[0] + vx / n * reach).floor() as i32, (me.pos[2] + vz / n * reach).floor() as i32);
    let below = me.pos[1].floor() as i32 - 1;
    (0..3).any(|d| ground(x, below - d, z)) // a step or two down is fine too
}
