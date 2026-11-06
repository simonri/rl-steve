//! The training round, as auto_env.AutoEnv plays it: key limits, points, the end of a round, respawns, and the
//! 4681 values the model sees. Courses come from Python (the same generators as for the real game); everything per
//! tick happens here.
//!
//! Every rule mirrors a line in auto_env.py or skill_env.py; test_sim_env.py checks the two give the same numbers.

use std::collections::HashSet;

use crate::clutch::{self, Cube, Push};
use crate::player::{Hit, Input, Keys, Player, FACE_STEP};
use crate::world::World;

pub const Y: i32 = 100;
pub const RADIUS: i32 = 5;
pub const SIZE: i32 = 2 * RADIUS + 1;
pub const MAP_RADIUS: i32 = 16;
pub const MAP_SIDE: i32 = 2 * MAP_RADIUS + 1;
pub const MAP_ABOVE: i32 = 8;
pub const MAP_BELOW: i32 = 16;
pub const HEAD: usize = 34;
pub const KNOWN: usize = 9;
pub const CUBE_START: usize = HEAD + KNOWN;
pub const MAP_START: usize = CUBE_START + (SIZE * SIZE * SIZE) as usize;
pub const MAP_CELLS: usize = (MAP_SIDE * MAP_SIDE) as usize;
pub const HISTORY_START: usize = MAP_START + 3 * MAP_CELLS;
pub const HISTORY: usize = 5;
pub const N_OBS: usize = HISTORY_START + 8 * HISTORY;

pub const TURN_STEPS: [f64; 11] = [-60.0, -15.0, -5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 15.0, 60.0];
const JUMP_WAIT: i32 = 10;
const MIN_KEY_TICKS: i32 = 2;
const SNEAK_CLICK_TICKS: i32 = 2;
const STALL_TICKS: i32 = 40;
const MOVED_ON: f64 = 1.5; // blocks from where it last was to count as going somewhere
const DETOUR_TICKS: i32 = 200; // longest it may go around something without getting closer: 10 s
const MAX_DEATHS: i32 = 2;
const DAMAGE_SETTLE_TICKS: i32 = 3;
const REWARD_SCALE: f64 = 0.1;

const TICK_COST: f64 = 0.002;
const EMPTY_CLICK_COST: f64 = 0.05;
const LOOK_UP_COST: f64 = 0.05;
const LANDING_BONUS: f64 = 1.0;
const BLOCK_COST: f64 = 0.02;
const WASTE_COST: f64 = 0.3;
const PLACE_BONUS: f64 = 1.0;
const GOAL_REWARD: f64 = 10.0;
const GOAL_AFTER_DEATH: f64 = 0.5;
const SPEED_BONUS: f64 = 3.0;
const DEATH_COST: f64 = 4.0;
const STUCK_COST: f64 = 4.0;
const STAYED_HOME_COST: f64 = 2.0;
const CLUTCH_BONUS: f64 = 2.0; // back on a block after a push, without falling
const SAVED_BLOCK_BONUS: f64 = 0.3; // at the goal, per block under the estimated need: as much as a block over it costs

/// The point rules, in the order info["parts"] lists them.
pub const PARTS: [&str; 16] = [
    "time", "click without blocks", "shift without blocks", "looking up", "landed on a block", "progress", "blocks", "placed toward the goal", "fall damage", "goal",
    "speed", "death", "stuck", "stayed on the start", "clutched", "blocks saved",
];
const P_TIME: usize = 0;
const P_EMPTY_CLICK: usize = 1;
const P_LOOK_UP: usize = 3;
const P_LANDED: usize = 4;
const P_PROGRESS: usize = 5;
const P_BLOCKS: usize = 6;
const P_PLACED: usize = 7;
const P_DAMAGE: usize = 8;
const P_GOAL: usize = 9;
const P_SPEED: usize = 10;
const P_DEATH: usize = 11;
const P_STUCK: usize = 12;
const P_HOME: usize = 13;
const P_CLUTCH: usize = 14;
const P_SAVED: usize = 15;

/// Walking distance to the goal around obstacles, worked out in Python (SkillEnv._build_field).
#[derive(Clone)]
pub struct Field {
    pub x0: i32,
    pub z0: i32,
    pub nx: i32,
    pub nz: i32,
    pub dist: Vec<f64>,
    pub blocked: Vec<bool>,
    pub max: f64,
}

impl Field {
    #[inline]
    fn at(&self, i: i32, j: i32) -> f64 {
        self.dist[(i * self.nz + j) as usize]
    }
    #[inline]
    fn blocked(&self, i: i32, j: i32) -> bool {
        i >= 0 && j >= 0 && i < self.nx && j < self.nz && self.blocked[(i * self.nz + j) as usize]
    }
}

/// One round as Python drew it.
#[derive(Clone)]
pub struct Course {
    pub hop: bool,
    pub goal: [i32; 2],
    pub dy: i32,
    pub goal_half: i32,
    pub fall_margin: f64,
    pub landing: Vec<[i32; 3]>,
    pub par: i32,
    pub limit: i32,
    pub no_blocks: bool,
    pub field: Option<Field>,
    pub home: [f64; 3],
    pub home_yaw: f32,
    pub home_pitch: f32,
    pub blocks: i32,
    pub world: World,
    /// Blocks were put in after the reset, and the game takes one tick with no keys to show them.
    pub settle: bool,
    /// Pushes, as a hit would give them, in the order they come (clutch.rs).
    pub pushes: Vec<Push>,
}

/// What a finished or ongoing round reports, for info.
#[derive(Clone, Copy, Default)]
pub struct Report {
    pub success: bool,
    pub clean: bool,
    pub deaths: i32,
    pub respawned: bool,
    pub stalled: bool,
    pub damage: f64,
    pub used: i32,
    pub landed: i32,
    pub shift_share: f64,
    pub t: i32,
    pub blocks_per_s: f64,
    pub pushes: i32,
}

pub struct Env {
    pub course: Course,
    pub world: World,
    pub player: Player,
    keys: [bool; 7], // forward, back, left, right, sneak, jump, sprint (the env's own order, as in _press)
    key_age: [i32; 7],
    since_sneak: i32,
    sneak_ticks: i32,
    pub t: i32,
    landed: usize,
    suffix: Vec<f64>,
    visited: HashSet<(i32, i32)>,
    best: f64,
    since_best: i32,
    anchor: [f64; 3],
    since_moved: i32,
    ground_distance: f64,
    used: i32,
    deaths: i32,
    damage: f64,
    closest_block: f64,
    jump_wait: i32,
    was_on_ground: bool,
    history: [[f32; 8]; HISTORY],
    pub parts: [f64; 16],
    pub report: Report,
    hit: Option<Hit>,
    k: i32,
    tick_count: i64,
    next_push: usize,
    pushed: bool,        // a push came and it has not landed again yet
    push_airborne: bool, // it has left the ground since
    pushes_given: i32,
}

const K_FORWARD: usize = 0;
const K_BACK: usize = 1;
const K_LEFT: usize = 2;
const K_RIGHT: usize = 3;
const K_SNEAK: usize = 4;
const K_JUMP: usize = 5;
const K_SPRINT: usize = 6;

/// AutoEnv._climb: distance left with `ahead` to go on the level and `up` to climb. With the goal higher up, getting
/// closer on the level than the climb left gains nothing: the way up has to start before it, not under the goal.
#[inline]
fn climb(ahead: f64, up: f64) -> f64 {
    if up > 0.0 {
        ahead.max(up) + up
    } else {
        ahead + up.abs()
    }
}

/// skill_env.rotate: turns (x, z) so yaw falls by 90 degrees k times.
#[inline]
pub fn rotate(mut x: f64, mut z: f64, k: i32) -> (f64, f64) {
    for _ in 0..k.rem_euclid(4) {
        let t = x;
        x = z;
        z = -t;
    }
    (x, z)
}

#[inline]
fn rotate_i(mut x: i32, mut z: i32, k: i32) -> (i32, i32) {
    for _ in 0..k.rem_euclid(4) {
        let t = x;
        x = z;
        z = -t;
    }
    (x, z)
}

impl Env {
    pub fn new(course: Course, tick_count: i64) -> Env {
        let mut world = course.world.clone();
        let mut player = Player::new(course.home, course.home_yaw, course.home_pitch, course.blocks);
        settle(&mut player);
        if course.settle {
            player.tick(&mut world, &Input::default());
        }
        let mut suffix = vec![0.0; course.landing.len() + 1];
        for i in (0..course.landing.len().saturating_sub(1)).rev() {
            let (a, b) = (course.landing[i], course.landing[i + 1]);
            suffix[i] = suffix[i + 1] + ((a[0] - b[0]) as f64).hypot((a[2] - b[2]) as f64) + (a[1] - b[1]).abs() as f64;
        }
        // The frame of the round: the goal does not move, so neither does this.
        let k = ((-(course.goal[0] as f64)).atan2(course.goal[1] as f64).to_degrees() / 90.0).round_ties_even() as i32;
        let mut env = Env {
            world,
            player,
            keys: [false; 7],
            key_age: [MIN_KEY_TICKS; 7],
            since_sneak: SNEAK_CLICK_TICKS + 1,
            sneak_ticks: 0,
            t: 0,
            landed: 0,
            suffix,
            visited: HashSet::new(),
            best: 0.0,
            since_best: 0,
            anchor: course.home,
            since_moved: 0,
            ground_distance: 0.0,
            used: 0,
            deaths: 0,
            damage: 0.0,
            closest_block: climb((course.goal[0] as f64).hypot(course.goal[1] as f64) - 1.0, course.dy as f64),
            jump_wait: 0,
            was_on_ground: true,
            history: [[0.0; 8]; HISTORY],
            parts: [0.0; 16],
            report: Report::default(),
            hit: None,
            k,
            tick_count,
            next_push: 0,
            pushed: false,
            push_airborne: false,
            pushes_given: 0,
            course,
        };
        env.best = env.distance();
        env.ground_distance = env.best;
        env.hit = env.player.pick(&env.world);
        env
    }

    // ---- Where it is on the course ----

    fn walk(&self, x: f64, z: f64) -> f64 {
        let c = &self.course;
        let Some(f) = &c.field else {
            return (c.goal[0] as f64 + 0.5 - x).hypot(c.goal[1] as f64 + 0.5 - z);
        };
        let ci = x.floor() as i32 - f.x0;
        let cj = z.floor() as i32 - f.z0;
        let mut best = f64::INFINITY;
        for i in (ci - 1).max(0)..(ci + 2).min(f.nx) {
            for j in (cj - 1).max(0)..(cj + 2).min(f.nz) {
                if f.blocked(i, j) {
                    continue;
                }
                if i != ci && j != cj && (f.blocked(i, cj) || f.blocked(ci, j)) {
                    continue;
                }
                let (cx, cz) = ((i + f.x0) as f64 + 0.5, (j + f.z0) as f64 + 0.5);
                let total = f.at(i, j) + (cx - x).hypot(cz - z);
                if total < best {
                    best = total;
                }
            }
        }
        if best < f64::INFINITY {
            best
        } else {
            f.max
        }
    }

    fn distance(&self) -> f64 {
        let [x, y, z] = self.player.pos;
        let c = &self.course;
        if c.landing.is_empty() {
            return climb(self.walk(x, z), (Y + 1 + c.dy) as f64 - y);
        }
        let i = self.landed.min(c.landing.len() - 1);
        let t = c.landing[i];
        let (tx, ty, tz) = (t[0] as f64 + 0.5, t[1] as f64, t[2] as f64 + 0.5);
        (tx - x).hypot(tz - z) + (ty - y).abs() + self.suffix[self.landed.min(c.landing.len())]
    }

    fn reached(&self) -> bool {
        let c = &self.course;
        if c.hop {
            return self.landed >= c.landing.len();
        }
        let [x, y, z] = self.player.pos;
        let reach = c.goal_half as f64 + 0.5;
        self.player.on_ground && (x - c.goal[0] as f64 - 0.5).abs() <= reach && (z - c.goal[1] as f64 - 0.5).abs() <= reach && (y - (Y + 1 + c.dy) as f64).abs() < 0.1
    }

    fn fell(&self) -> bool {
        self.player.pos[1] < self.fall_line()
    }

    fn on_start_platform(&self) -> bool {
        let [x, y, z] = self.player.pos;
        (x - 0.5).abs().max((z - 0.5).abs()) <= 1.8 && (y - (Y + 1) as f64).abs() < 0.6
    }

    fn is_on(&self, cell: [i32; 3]) -> bool {
        let [x, y, z] = self.player.pos;
        y.floor() as i32 == cell[1] && (cell[0] as f64 - 0.3) < x && x < cell[0] as f64 + 1.3 && (cell[2] as f64 - 0.3) < z && z < cell[2] as f64 + 1.3
    }

    fn points(&mut self, part: usize, value: f64) -> f64 {
        self.parts[part] += value * REWARD_SCALE;
        value
    }

    // ---- Playing ----

    /// SkillEnv._press: a key changes at most every MIN_KEY_TICKS.
    fn press(&mut self, wanted: [bool; 7]) {
        for i in 0..7 {
            self.key_age[i] += 1;
            if wanted[i] == self.keys[i] || self.key_age[i] < MIN_KEY_TICKS {
                continue;
            }
            self.keys[i] = wanted[i];
            self.key_age[i] = 0;
        }
    }

    fn release_inputs(&mut self) {
        self.keys = [false; 7];
        self.key_age = [MIN_KEY_TICKS; 7];
        self.since_sneak = SNEAK_CLICK_TICKS + 1;
        self.jump_wait = 0;
        self.was_on_ground = true;
    }

    fn game_keys(&self) -> Keys {
        let k = &self.keys;
        Keys { forward: k[K_FORWARD], back: k[K_BACK], left: k[K_LEFT], right: k[K_RIGHT], jump: k[K_JUMP], shift: k[K_SNEAK], sprint: k[K_SPRINT] }
    }

    fn game_tick(&mut self, input: &Input) -> crate::player::Events {
        let ev = self.player.tick(&mut self.world, input);
        self.tick_count += 1;
        ev
    }

    /// One decision: AutoEnv.step with one tick. Returns the reward and whether the round is over.
    pub fn step(&mut self, a: [i64; 8]) -> (f64, bool) {
        let (mv, strafe, sneak, use_, jump, sprint, yaw, pitch) = (a[0], a[1], a[2] != 0, a[3] != 0, a[4] != 0, a[5] != 0, a[6] as usize, a[7] as usize);
        self.history.rotate_left(1);
        self.history[HISTORY - 1] = [mv as f32 / 2.0, strafe as f32 / 2.0, sneak as i32 as f32, use_ as i32 as f32 / 2.0, jump as i32 as f32, sprint as i32 as f32, yaw as f32 / 10.0, pitch as f32 / 10.0];
        let held = self.player.held;
        let health = self.player.health;
        let before = self.game_keys();
        self.press([mv == 1, mv == 2, strafe == 1, strafe == 2, sneak, jump, sprint]);
        self.since_sneak = if self.keys[K_SNEAK] { 0 } else { self.since_sneak + 1 };
        self.sneak_ticks += self.keys[K_SNEAK] as i32;
        let clicked = use_ && self.since_sneak <= SNEAK_CLICK_TICKS;
        let input = Input { keys: self.game_keys(), click: clicked, turn_yaw: TURN_STEPS[yaw], turn_pitch: TURN_STEPS[pitch] };
        self.maybe_push(before, &input);
        let ev = self.game_tick(&input);
        self.t += 1;
        let on_ground = self.player.on_ground;
        // _track_jump
        if !self.keys[K_JUMP] {
            self.jump_wait = 0;
        } else if self.jump_wait > 0 {
            self.jump_wait -= 1;
        }
        if self.keys[K_JUMP] && self.was_on_ground && !on_ground && self.player.dm[1] > 0.2 {
            self.jump_wait = JUMP_WAIT;
        }
        self.was_on_ground = on_ground;
        if on_ground {
            self.visited.insert((self.player.pos[0].floor() as i32, self.player.pos[2].floor() as i32));
        }
        self.hit = self.player.pick(&self.world);
        let mut reward = self.points(P_TIME, -TICK_COST);
        if use_ && held == 0 {
            reward += self.points(P_EMPTY_CLICK, -EMPTY_CLICK_COST);
        }
        // Shift is free everywhere: on parkour it stops at an edge, lines up a jump, or takes a careful step.
        if self.player.pitch < 0.0 && self.hit.is_none() {
            reward += self.points(P_LOOK_UP, -LOOK_UP_COST * -(self.player.pitch as f64) / 90.0);
        }
        if self.pushed {
            if !on_ground {
                self.push_airborne = true;
            } else if self.push_airborne {
                self.pushed = false;
                reward += self.points(P_CLUTCH, CLUTCH_BONUS);
            }
        }
        let n = self.course.landing.len();
        if self.course.hop {
            if self.landed < n && on_ground && self.is_on(self.course.landing[self.landed]) {
                self.landed += 1;
                reward += self.points(P_LANDED, LANDING_BONUS);
            }
        } else if n > 0 && on_ground {
            for i in self.landed..(self.landed + 3).min(n) {
                if self.is_on(self.course.landing[i]) {
                    self.landed = i + 1;
                    break;
                }
            }
        }
        let now = self.distance();
        if on_ground {
            let gained = self.ground_distance - now;
            if self.deaths == 0 || gained > 0.0 {
                reward += self.points(P_PROGRESS, gained);
                self.ground_distance = now;
            }
        }
        if now < self.best - 0.01 {
            self.best = now;
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        for _ in 0..(held - self.player.held).max(0) {
            self.used += 1;
            let cost = if self.used <= self.course.par { BLOCK_COST } else { WASTE_COST };
            reward += self.points(P_BLOCKS, -cost);
        }
        if let Some((target, _)) = ev.placed {
            // As far as standing on it would leave: a block under the goal while still below it is no step closer.
            let reach = climb(((self.course.goal[0] - target[0]) as f64).hypot((self.course.goal[1] - target[2]) as f64), (self.course.dy - target[1] + Y) as f64);
            if reach < self.closest_block - 0.5 {
                self.closest_block = reach;
                reward += self.points(P_PLACED, PLACE_BONUS);
            }
        }
        let success = self.reached();
        let fell = self.fell() && !success;
        if success {
            // The landing's damage arrives a moment later; the keys are let go meanwhile.
            for _ in 0..DAMAGE_SETTLE_TICKS {
                self.game_tick(&Input::default());
            }
            self.hit = self.player.pick(&self.world);
        }
        // Stuck: no closer for 2 s and still about where it was then. A way around something first leads away from the
        // goal, so moving on counts too, for up to 10 s without getting any closer.
        let p = self.player.pos;
        if ((p[0] - self.anchor[0]).powi(2) + (p[1] - self.anchor[1]).powi(2) + (p[2] - self.anchor[2]).powi(2)).sqrt() > MOVED_ON {
            self.anchor = p;
            self.since_moved = 0;
        } else {
            self.since_moved += 1;
        }
        let stalled = !(success || fell) && (self.since_best >= STALL_TICKS && self.since_moved >= STALL_TICKS || self.since_best >= DETOUR_TICKS);
        let lost = (health - self.player.health).max(0.0) as f64;
        reward += self.points(P_DAMAGE, -(0.5 * lost));
        self.damage += lost;
        let clean = success && self.deaths == 0;
        if success {
            reward += self.points(P_GOAL, GOAL_REWARD * if clean { 1.0 } else { GOAL_AFTER_DEATH });
            reward += self.points(P_SPEED, SPEED_BONUS * (1.0 - self.t as f64 / self.course.limit as f64).max(0.0));
            if self.course.blocks > 0 && self.used < self.course.par {
                reward += self.points(P_SAVED, SAVED_BLOCK_BONUS * (self.course.par - self.used) as f64);
            }
        }
        if fell {
            reward += self.points(P_DEATH, -DEATH_COST);
        }
        if stalled {
            reward += self.points(P_STUCK, -STUCK_COST);
        }
        let mut respawned = false;
        let mut out_of_lives = false;
        if fell {
            self.deaths += 1;
            out_of_lives = self.deaths >= MAX_DEATHS;
            if !out_of_lives {
                self.respawn();
                respawned = true;
            }
        }
        let timed_out = self.t >= self.course.limit && !(success || stalled || out_of_lives);
        if (stalled || timed_out) && !fell && self.on_start_platform() {
            reward += self.points(P_HOME, -STAYED_HOME_COST);
        }
        let done = success || stalled || out_of_lives || timed_out;
        self.report = Report {
            success,
            clean,
            deaths: self.deaths,
            respawned,
            stalled,
            damage: self.damage,
            used: self.used,
            landed: self.landed as i32,
            shift_share: self.sneak_ticks as f64 / self.t.max(1) as f64,
            t: self.t,
            pushes: self.pushes_given,
            blocks_per_s: if success { (self.course.goal[0] as f64).hypot(self.course.goal[1] as f64) / (self.t as f64 / 20.0).max(0.05) } else { 0.0 },
        };
        (reward * REWARD_SCALE, done)
    }

    /// Dead: back at the start with what was built and the blocks left (SkillEnv._respawn, the mod's keep_arena).
    fn respawn(&mut self) {
        let h = self.course.home;
        for x in (h[0] - 0.31).floor() as i32..=(h[0] + 0.31).floor() as i32 {
            for z in (h[2] - 0.31).floor() as i32..=(h[2] + 0.31).floor() as i32 {
                for y in h[1].floor() as i32..=(h[1] + 1.81).floor() as i32 {
                    self.world.set(x, y, z, crate::world::AIR);
                }
            }
        }
        let held = self.player.held;
        self.player = Player::new(h, self.course.home_yaw, self.course.home_pitch, held);
        settle(&mut self.player);
        self.history = [[0.0; 8]; HISTORY];
        self.release_inputs();
        self.best = self.distance();
        self.since_best = 0;
        self.anchor = h;
        self.since_moved = 0;
        self.hit = self.player.pick(&self.world);
        self.pushed = false;
    }

    /// AutoEnv._maybe_push: the next planned push, the first tick it fits and can be saved.
    fn maybe_push(&mut self, before: Keys, input: &Input) {
        let pushes = &self.course.pushes;
        while self.next_push < pushes.len() && self.t > pushes[self.next_push].until {
            self.next_push += 1;
        }
        if self.next_push >= pushes.len() || self.pushed || !self.player.on_ground || self.t < pushes[self.next_push].from {
            return;
        }
        let p = &self.player;
        let feet = [p.pos[0].floor() as i32, p.pos[1].floor() as i32, p.pos[2].floor() as i32];
        let cube = Cube::from_fn(feet, |x, y, z| self.world.present(x, y, z));
        let state = clutch::State { pos: p.pos, vel: p.dm, yaw: p.yaw, pitch: p.pitch, held: p.held, keys: before };
        let goal = [self.course.goal[0] as f64 + 0.5, self.course.goal[1] as f64 + 0.5];
        if let Some(v) = clutch::decide(&cube, &state, input, goal, self.fall_line(), &pushes[self.next_push]) {
            clutch::apply(&mut self.player, v);
            self.next_push += 1;
            self.pushed = true;
            self.push_airborne = false;
            self.pushes_given += 1;
        }
    }

    fn fall_line(&self) -> f64 {
        ((Y + 1).min(Y + 1 + self.course.dy)) as f64 - self.course.fall_margin
    }

    // ---- What it sees ----

    /// AutoEnv._vec, _with_timer and the history: everything the model gets, into `out`.
    pub fn observe(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), N_OBS);
        let p = &self.player;
        let c = &self.course;
        let [x, y, z] = p.pos;
        let [vx, vy, vz] = p.dm;
        let k = self.k;
        let turn = (p.yaw as f64).to_radians();
        let (fw, rt) = ((-turn.sin(), turn.cos()), (-turn.cos(), -turn.sin()));
        let ego = |dx: f64, dz: f64| (dx * fw.0 + dz * fw.1, dx * rt.0 + dz * rt.1);
        let gx = c.goal[0] as f64 + 0.5 - x;
        let gz = c.goal[1] as f64 + 0.5 - z;
        let distance = gx.hypot(gz);
        let (goal_f, goal_r) = ego(gx, gz);
        let goal_up = (Y + 1 + c.dy) as f64 - y;
        let (cx, cz) = rotate(x - x.floor() - 0.5, z - z.floor() - 0.5, k);
        let (sf, sr) = ego(vx, vz);
        let look = turn - k as f64 * std::f64::consts::PI / 2.0;
        let mut head = [0.0f64; HEAD];
        head[0] = goal_f / distance.max(0.01);
        head[1] = goal_r / distance.max(0.01);
        head[2] = distance.ln_1p() / 4.0;
        head[3] = goal_up.abs().ln_1p().copysign(goal_up) / 3.0;
        head[4] = cx;
        head[5] = cz;
        head[6] = y - y.floor() - 0.5;
        head[7] = sf * 5.0;
        head[8] = sr * 5.0;
        head[9] = vy * 2.0;
        head[10] = p.pitch as f64 / 90.0;
        head[11] = look.sin();
        head[12] = look.cos();
        head[13] = p.on_ground as i32 as f64;
        head[14] = p.sprinting as i32 as f64;
        head[15] = p.health as f64 / 20.0;
        head[16] = p.horizontal_collision as i32 as f64;
        head[17] = p.fall_distance.ln_1p() / 3.0;
        head[18] = p.hurt_time as f64 / 10.0;
        if let Some(h) = &self.hit {
            let b = h.block;
            head[19] = 1.0;
            let (hx, hz) = rotate(b[0] as f64 + 0.5 - x, b[2] as f64 + 0.5 - z, k);
            head[20] = hx / 3.0;
            head[21] = (b[1] as f64 + 0.5 - y) / 3.0;
            head[22] = hz / 3.0;
            let (lx, lz) = rotate((h.loc[0] - b[0] as f64 - 0.5) * 2.0, (h.loc[2] - b[2] as f64 - 0.5) * 2.0, k);
            head[23] = lx;
            head[24] = (h.loc[1] - b[1] as f64 - 0.5) * 2.0;
            head[25] = lz;
            // _side: up, down, then the four sides in the round's frame
            let side = match h.face {
                1 => 0,
                0 => 1,
                f => {
                    let s = FACE_STEP[f as usize];
                    let (fx, fz) = rotate_i(s[0], s[2], k);
                    if fx > 0 {
                        2
                    } else if fx < 0 {
                        3
                    } else if fz > 0 {
                        4
                    } else {
                        5
                    }
                }
            };
            head[26 + side] = 1.0;
        }
        head[32] = (p.held as f64 / 20.0).min(2.0);
        head[33] = self.can_place() as i32 as f64;
        for i in 0..HEAD {
            out[i] = (head[i] as f32).clamp(-5.0, 5.0);
        }
        // _with_timer
        let ks = &self.keys;
        let known = [
            ks[K_SNEAK] as i32 as f32,
            (1.0 - self.t as f64 / c.limit as f64).max(0.0) as f32,
            ks[K_SPRINT] as i32 as f32,
            ks[K_JUMP] as i32 as f32,
            ks[K_FORWARD] as i32 as f32,
            ks[K_BACK] as i32 as f32,
            ks[K_LEFT] as i32 as f32,
            ks[K_RIGHT] as i32 as f32,
            (self.jump_wait as f64 / JUMP_WAIT as f64) as f32,
        ];
        out[HEAD..HEAD + KNOWN].copy_from_slice(&known);
        // _view: every block 5 around the feet, in the round's frame
        let feet = [x.floor() as i32, y.floor() as i32, z.floor() as i32];
        let back = (-k).rem_euclid(4);
        let mut i = CUBE_START;
        for vx_ in -RADIUS..=RADIUS {
            for up in -RADIUS..=RADIUS {
                for vz_ in -RADIUS..=RADIUS {
                    let (wx, wz) = rotate_i(vx_, vz_, back);
                    out[i] = self.world.present(feet[0] + wx, feet[1] + up, feet[2] + wz) as i32 as f32;
                    i += 1;
                }
            }
        }
        // _map_layers: height of the top block, whether there is one, whether it has stood there
        let (heights, rest) = out[MAP_START..HISTORY_START].split_at_mut(MAP_CELLS);
        let (theres, beens) = rest.split_at_mut(MAP_CELLS);
        beens.fill(0.0);
        let mut i = 0;
        for mx in -MAP_RADIUS..=MAP_RADIUS {
            for mz in -MAP_RADIUS..=MAP_RADIUS {
                let (wx, wz) = rotate_i(mx, mz, back);
                let mut top = None;
                for dy in (-MAP_BELOW..=MAP_ABOVE).rev() {
                    if self.world.present(feet[0] + wx, feet[1] + dy, feet[2] + wz) {
                        top = Some(dy);
                        break;
                    }
                }
                match top {
                    Some(dy) => {
                        heights[i] = (dy as f32 / 8.0).clamp(-5.0, 5.0);
                        theres[i] = 1.0;
                    }
                    None => {
                        heights[i] = 0.0;
                        theres[i] = 0.0;
                    }
                }
                i += 1;
            }
        }
        for &(cx_, cz_) in &self.visited {
            let (rx, rz) = rotate_i(cx_ - feet[0], cz_ - feet[2], k);
            if rx.abs() <= MAP_RADIUS && rz.abs() <= MAP_RADIUS {
                beens[((rx + MAP_RADIUS) * MAP_SIDE + rz + MAP_RADIUS) as usize] = 1.0;
            }
        }
        let mut i = HISTORY_START;
        for h in &self.history {
            out[i..i + 8].copy_from_slice(h);
            i += 8;
        }
    }

    /// AutoEnv._can_place: a block in hand, aimed at a side in reach, the spot there empty and not where it stands.
    fn can_place(&self) -> bool {
        let p = &self.player;
        let Some(h) = &self.hit else { return false };
        if p.held <= 0 {
            return false;
        }
        let s = FACE_STEP[h.face as usize];
        let t = [h.block[0] + s[0], h.block[1] + s[1], h.block[2] + s[2]];
        let [x, y, z] = p.pos;
        let feet = [x.floor() as i32, y.floor() as i32, z.floor() as i32];
        let cell = [t[0] - feet[0], t[1] - feet[1], t[2] - feet[2]];
        if cell.iter().all(|c| c.abs() <= RADIUS) && self.world.present(t[0], t[1], t[2]) {
            return false;
        }
        let height = if p.keys.shift { 1.5 } else { 1.8 };
        let (tx, ty, tz) = (t[0] as f64, t[1] as f64, t[2] as f64);
        let overlaps = tx < x + 0.3 && x - 0.3 < tx + 1.0 && tz < z + 0.3 && z - 0.3 < tz + 1.0 && ty < y + height && y < ty + 1.0;
        !overlaps
    }
}

/// What the mod's reset leaves: standing still on the ground, as gravity and the ground cancel out each tick.
fn settle(p: &mut Player) {
    p.on_ground = true;
    p.dm = [0.0, -0.0784, 0.0];
    p.fall_distance = 0.0;
    p.server_fall = 0.0;
    p.health = crate::player::MAX_HEALTH;
    p.hurt_time = 0;
}
