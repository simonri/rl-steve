//! Knockback as a hit gives it, and the check that a push can always be saved.
//!
//! A push is only given when it makes sense and can be saved: the player stands on the ground, there is void right
//! beside it in the direction of the push, the push never goes toward the goal, and a scripted clutch with the same
//! inputs the model has (keys at most every second tick, a click only with shift down, turns from TURN_STEPS) lands
//! on a block above the fall line. If the strongest push cannot be saved, a weaker one is tried; if none can, there is
//! no push this tick. A push is also only given when doing nothing after it would be a fall: one that lands safely by
//! itself teaches nothing about clutching.
//!
//! Everything here sees only what the model sees, the cube of blocks 5 around the feet, so Python (the real game) and
//! the Rust rounds decide the same.

use crate::env::{RADIUS, TURN_STEPS};
use crate::player::{Input, Keys, Player, FACE_STEP};
use crate::world::{World, FULL};

/// Horizontal speed of a hit, a sprinting hit and a Knockback I hit, from LivingEntity.knockback and
/// Player.causeExtraKnockback with the server's own motion for the player at rest (0): 0.4, then halved plus 0.5,
/// then halved plus 1.0. Upward it is min(0.4, y / 2 + power) = 0.4 every time, since the player is on the ground.
pub const STRENGTHS: [f64; 3] = [0.4, 0.7, 1.2];
pub const UP: f64 = 0.4;
const DIRECTIONS: usize = 16;
const RESCUE_TICKS: usize = 40;
const MAX_CANDIDATES: usize = 48;
const TOWARD_GOAL: f64 = 0.3; // a push this much along the way to the goal would help, not hurt
const NEAR_GOAL: f64 = 3.0;

/// The cube around the feet, as the model sees it: `present(dx, dy, dz)` for -RADIUS..=RADIUS.
pub struct Cube {
    pub feet: [i32; 3],
    cells: Vec<bool>,
}

const SIDE: i32 = 2 * RADIUS + 1;

impl Cube {
    pub fn from_fn(feet: [i32; 3], present: impl Fn(i32, i32, i32) -> bool) -> Cube {
        let mut cells = Vec::with_capacity((SIDE * SIDE * SIDE) as usize);
        for x in -RADIUS..=RADIUS {
            for y in -RADIUS..=RADIUS {
                for z in -RADIUS..=RADIUS {
                    cells.push(present(feet[0] + x, feet[1] + y, feet[2] + z));
                }
            }
        }
        Cube { feet, cells }
    }

    /// From the mod's "solid" text: '1' where something is, x then y then z.
    pub fn from_text(feet: [i32; 3], text: &str) -> Cube {
        Cube { feet, cells: text.bytes().map(|b| b == b'1').collect() }
    }

    fn at(&self, x: i32, y: i32, z: i32) -> bool {
        let (i, j, k) = (x - self.feet[0] + RADIUS, y - self.feet[1] + RADIUS, z - self.feet[2] + RADIUS);
        if i < 0 || j < 0 || k < 0 || i >= SIDE || j >= SIDE || k >= SIDE {
            return false;
        }
        self.cells[((i * SIDE + j) * SIDE + k) as usize]
    }

    fn world(&self) -> World {
        let f = self.feet;
        let mut w = World::new([f[0] - RADIUS, f[1] - RADIUS, f[2] - RADIUS], [f[0] + RADIUS, f[1] + RADIUS, f[2] + RADIUS]);
        for x in -RADIUS..=RADIUS {
            for y in -RADIUS..=RADIUS {
                for z in -RADIUS..=RADIUS {
                    if self.at(f[0] + x, f[1] + y, f[2] + z) {
                        w.set(f[0] + x, f[1] + y, f[2] + z, FULL);
                    }
                }
            }
        }
        w
    }

    /// Nothing under this column for four blocks below the feet: stepping off here is a fall.
    fn void_below(&self, x: f64, z: f64) -> bool {
        let (cx, cz) = (x.floor() as i32, z.floor() as i32);
        (1..=4).all(|d| !self.at(cx, self.feet[1] - d, cz))
    }
}

/// What the player is doing when the push would come.
#[derive(Clone, Copy)]
pub struct State {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub held: i32,
    pub keys: Keys,
    pub sprinting: bool,
}

/// One planned push: from this tick on, the first moment it fits.
#[derive(Clone, Copy, Debug)]
pub struct Push {
    pub from: i32,
    pub until: i32,
    pub level: usize, // index into STRENGTHS: the strongest it may be
    pub turn: f64,    // 0..1: which of the possible directions
    pub jitter: f64,  // degrees added to the 16 compass directions
}

/// The velocity of the push to give now, or None when no push fits or none can be saved.
pub fn decide(cube: &Cube, s: &State, first: &Input, goal: [f64; 2], fall_line: f64, push: &Push) -> Option<[f64; 3]> {
    if s.held <= 0 {
        return None;
    }
    let (gx, gz) = (goal[0] - s.pos[0], goal[1] - s.pos[2]);
    let gd = gx.hypot(gz);
    if gd < NEAR_GOAL {
        return None;
    }
    let mut ways = Vec::with_capacity(DIRECTIONS);
    for i in 0..DIRECTIONS {
        let a = (i as f64 * 360.0 / DIRECTIONS as f64 + push.jitter).to_radians();
        let (dx, dz) = (a.cos(), a.sin());
        if (dx * gx + dz * gz) / gd > TOWARD_GOAL {
            continue;
        }
        if cube.void_below(s.pos[0] + dx * 1.0, s.pos[2] + dz * 1.0) && cube.void_below(s.pos[0] + dx * 2.0, s.pos[2] + dz * 2.0) {
            ways.push((dx, dz));
        }
    }
    if ways.is_empty() {
        return None;
    }
    let (dx, dz) = ways[((push.turn * ways.len() as f64) as usize).min(ways.len() - 1)];
    let world = cube.world();
    for level in (0..=push.level.min(STRENGTHS.len() - 1)).rev() {
        let v = [dx * STRENGTHS[level], UP, dz * STRENGTHS[level]];
        if falls(&world, s, v, first, fall_line) && can_save(&world, cube.feet, s, v, first, fall_line) {
            return Some(v);
        }
    }
    None
}

/// Puts the push on a player, as ClientboundSetEntityMotionPacket and the hurt event do.
pub fn apply(p: &mut Player, v: [f64; 3]) {
    p.dm = v;
    p.hurt_time = 10;
}

/// Whether a scripted clutch saves this push: any block next to an empty spot under or beside the way it flies,
/// aimed at with the model's turn steps, clicked with shift down.
pub fn can_save(world: &World, feet: [i32; 3], s: &State, v: [f64; 3], first: &Input, fall_line: f64) -> bool {
    let mut targets = Vec::new();
    for dy in 1..=3 {
        let y = feet[1] - dy;
        for x in feet[0] - 4..=feet[0] + 4 {
            for z in feet[2] - 4..=feet[2] + 4 {
                if world.present(x, y, z) {
                    continue;
                }
                let (cx, cz) = (x as f64 + 0.5 - s.pos[0], z as f64 + 0.5 - s.pos[2]);
                if cx * v[0] + cz * v[2] < 0.0 {
                    continue; // behind where it flies
                }
                for (f, step) in FACE_STEP.iter().enumerate() {
                    let n = [x - step[0], y - step[1], z - step[2]];
                    if world.present(n[0], n[1], n[2]) {
                        targets.push((cx.hypot(cz), [x, y, z], n, f as u8));
                    }
                }
            }
        }
    }
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    targets.truncate(MAX_CANDIDATES);
    targets.iter().any(|&(_, cell, against, face)| rescue(world, s, v, first, fall_line, cell, against, face))
}

/// The player as it is when the push comes, with the push on it.
fn pushed(s: &State, v: [f64; 3]) -> Player {
    let mut p = Player::new(s.pos, s.yaw, s.pitch, s.held);
    p.on_ground = true;
    p.keys = s.keys;
    p.sprinting = s.sprinting;
    p.crouch_pose = s.keys.shift;
    apply(&mut p, v);
    p
}

/// Whether the push is a danger: with the model's input for this tick and then nothing, it ends below the fall line.
fn falls(world: &World, s: &State, v: [f64; 3], first: &Input, fall_line: f64) -> bool {
    let mut w = world.clone();
    let mut p = pushed(s, v);
    let rest = Input { keys: first.keys, ..Input::default() };
    for t in 0..RESCUE_TICKS {
        p.tick(&mut w, if t == 0 { first } else { &rest });
        if p.pos[1] < fall_line {
            return true;
        }
        if t > 0 && p.on_ground {
            return false;
        }
    }
    false
}

fn rescue(world: &World, s: &State, v: [f64; 3], first: &Input, fall_line: f64, cell: [i32; 3], against: [i32; 3], face: u8) -> bool {
    let mut w = world.clone();
    let mut p = pushed(s, v);
    // The keys as the model holds them: one that changed waits MIN_KEY_TICKS before it can change again (env.press).
    let mut keys = key_list(first.keys);
    let before = key_list(s.keys);
    let mut age: [i32; 7] = std::array::from_fn(|i| if keys[i] != before[i] { 0 } else { MIN_KEY_TICKS });
    let aim = [
        against[0] as f64 + 0.5 + FACE_STEP[face as usize][0] as f64 * 0.5,
        against[1] as f64 + 0.5 + FACE_STEP[face as usize][1] as f64 * 0.5,
        against[2] as f64 + 0.5 + FACE_STEP[face as usize][2] as f64 * 0.5,
    ];
    let mut was_off = false;
    for t in 0..RESCUE_TICKS {
        // Tick 0 is the model's own input, already chosen. Its keys can change only every second tick, and a click
        // needs shift down, so the script holds those as they were for one more tick.
        let input = if t == 0 {
            *first
        } else {
            let eye = [p.pos[0], p.pos[1] + p.eye_height(), p.pos[2]];
            let (ax, ay, az) = (aim[0] - eye[0], aim[1] - eye[1], aim[2] - eye[2]);
            let want_yaw = (-ax).atan2(az).to_degrees();
            let want_pitch = (-ay).atan2(ax.hypot(az)).to_degrees();
            let dyaw = (want_yaw - p.yaw as f64 + 540.0).rem_euclid(360.0) - 180.0;
            let dpitch = want_pitch - p.pitch as f64;
            let want = if t == 1 { keys } else { key_list(toward(&p, cell)) };
            for i in 0..7 {
                age[i] += 1;
                if want[i] != keys[i] && age[i] >= MIN_KEY_TICKS {
                    keys[i] = want[i];
                    age[i] = 0;
                }
            }
            let (ty, tp) = (step(dyaw), step(dpitch));
            let placed = w.present(cell[0], cell[1], cell[2]);
            // The game turns first and then picks for the click, so aim is judged after this tick's turn.
            let mut turned = p.clone();
            turned.turn(ty, tp);
            let on_face = turned.pick(&w).is_some_and(|h| h.block == against && h.face == face);
            let k = from_list(keys);
            Input { keys: k, click: t >= 2 && k.shift && !placed && on_face, turn_yaw: ty, turn_pitch: tp }
        };
        p.tick(&mut w, &input);
        if p.pos[1] < fall_line {
            return false;
        }
        if !p.on_ground {
            was_off = true;
        } else if was_off {
            return true;
        }
    }
    false
}

const MIN_KEY_TICKS: i32 = 2; // env.rs: a key changes at most every second tick

fn key_list(k: Keys) -> [bool; 7] {
    [k.forward, k.back, k.left, k.right, k.jump, k.shift, k.sprint]
}

fn from_list(k: [bool; 7]) -> Keys {
    Keys { forward: k[0], back: k[1], left: k[2], right: k[3], jump: k[4], shift: k[5], sprint: k[6] }
}

/// Keys toward the middle of the spot, shift down: the best the model could press.
fn toward(p: &Player, cell: [i32; 3]) -> Keys {
    let (dx, dz) = (cell[0] as f64 + 0.5 - p.pos[0], cell[2] as f64 + 0.5 - p.pos[2]);
    let turn = (p.yaw as f64).to_radians();
    let (f, r) = (dx * -turn.sin() + dz * turn.cos(), dx * -turn.cos() + dz * -turn.sin());
    let len = f.hypot(r).max(1e-9);
    let (f, r) = (f / len, r / len);
    Keys { forward: f > 0.38, back: f < -0.38, left: r < -0.38, right: r > 0.38, jump: false, shift: true, sprint: false }
}

/// The turn step closest to what is left to turn.
fn step(left: f64) -> f64 {
    let mut best = 0.0;
    for &s in TURN_STEPS.iter() {
        if (s - left).abs() < (best - left).abs() {
            best = s;
        }
    }
    best
}
