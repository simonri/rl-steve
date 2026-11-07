//! The training round, as auto_env.AutoEnv plays it: key limits, points, the end of a round, respawns, and the
//! 4681 values the model sees. Courses come from Python (the same generators as for the real game); everything per
//! tick happens here.
//!
//! Every rule mirrors a line in auto_env.py or skill_env.py; test_sim_env.py checks the two give the same numbers.

use std::collections::HashSet;

use crate::clutch::{self, Cube, Push};
use crate::combat::{self, Swing};
use crate::duel::{self, Bot, View, ACTIONS, EXTRA, HIST_F};
use crate::player::{Hit, Input, Keys, Player, FACE_STEP};
use crate::shaping;
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
pub const MOVE_OBS: usize = HISTORY_START + 8 * HISTORY; // what the movement model saw
pub const N_OBS: usize = MOVE_OBS + EXTRA; // and what it sees with opponents (duel.rs)

pub const TURN_STEPS: [f64; 11] = [-60.0, -15.0, -5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 15.0, 60.0];
const JUMP_WAIT: i32 = 10;
pub const MIN_KEY_TICKS: i32 = 2;

/// SkillEnv._press: each key goes where it is wanted, but changes at most every MIN_KEY_TICKS (a human's limit). The
/// same for the model, the bots and the scripted clutch, in whatever key order they keep.
pub fn limit_keys(keys: &mut [bool; 7], age: &mut [i32; 7], wanted: [bool; 7]) {
    for i in 0..7 {
        age[i] += 1;
        if wanted[i] != keys[i] && age[i] >= MIN_KEY_TICKS {
            keys[i] = wanted[i];
            age[i] = 0;
        }
    }
}

/// The game's keys from the env's order (K_FORWARD ... K_SPRINT).
fn to_game_keys(k: &[bool; 7]) -> Keys {
    Keys { forward: k[K_FORWARD], back: k[K_BACK], left: k[K_LEFT], right: k[K_RIGHT], jump: k[K_JUMP], shift: k[K_SNEAK], sprint: k[K_SPRINT] }
}
const SNEAK_CLICK_TICKS: i32 = 2;
const STALL_TICKS: i32 = 40;
const MOVED_ON: f64 = 1.5; // blocks from where it last was to count as going somewhere
const DETOUR_TICKS: i32 = 200; // longest it may go around something without getting closer: 10 s
const MAX_DEATHS: i32 = 2;
const DAMAGE_SETTLE_TICKS: i32 = 3;
const REWARD_SCALE: f64 = 0.1;

const TICK_COST: f64 = 0.006; // skill_env.TICK_COST
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
const DEATH_HEALTH: f32 = 4.0; // auto_env.DEATH_HEALTH
const CLUTCH_BONUS: f64 = 2.0; // back on a block after a push, without falling
const SAVED_BLOCK_BONUS: f64 = 0.3; // at the goal, per block under the estimated need: as much as a block over it costs

/// The point rules, in the order info["parts"] lists them.
pub const PARTS: [&str; 24] = [
    "time", "click without blocks", "shift without blocks", "looking up", "landed on a block", "progress", "blocks", "closer than ever", "fall damage", "goal",
    "speed", "death", "stuck", "stayed on the start", "clutched", "blocks saved", "damage dealt", "damage taken", "won", "lost", "aim", "missed swing", "long hit", "critical hit",
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
const P_DEALT: usize = 16;
const P_TAKEN: usize = 17;
const P_WON: usize = 18;
const P_LOST: usize = 19;
const P_AIM: usize = 20;
const P_SWING: usize = 21;
const P_LONG_HIT: usize = 22;
const P_CRIT: usize = 23;
const HEALTH_POINTS: f64 = 0.3; // per health point dealt or taken in a fight: a trade of hits is worth nothing
const FIGHT_RESULT: f64 = 10.0; // winning or losing a fight
/// Below the world (its bottom at -64, less 64) a player takes this every tick, whatever its hurt resistance.
const VOID_Y: f64 = -128.0;
const VOID_DAMAGE: f32 = 4.0;

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
    /// 1.8 rules for healing (combat always follows them).
    pub rules18: bool,
    /// A fight instead of a way to a goal.
    pub duel: Option<Duel>,
}

/// A fight: the model's gear and its opponents, each a scripted bot.
#[derive(Clone)]
pub struct Duel {
    pub damage: f32, // attack attribute with the sword
    pub armor: i32,
    pub kb: i32,
    pub centre: [f64; 3], // the middle of the arena: bots that push try to send it the other way
    pub opponents: Vec<Rival>,
}

#[derive(Clone)]
pub struct Rival {
    pub bot: Bot,
    pub pos: [f64; 3],
    pub yaw: f32,
    pub damage: f32,
    pub armor: i32,
    pub kb: i32,
}

/// An opponent in a round: its player, its bot, and the same key limits the model has.
pub struct Opponent {
    pub player: Player,
    pub bot: Bot,
    pub out: bool, // dead or fallen: no longer in the round
    keys: [bool; 7],
    key_age: [i32; 7],
    since_sneak: i32,
    since_hit: i32,
    since_hurt: i32,                   // ticks since the model's hit counted on it (its own fighting state, as the model's)
    history: [[f32; 8]; HISTORY],      // its last inputs, as the model's history
    history2: [[f32; HIST_F]; HISTORY],
    target: Option<[f64; 3]>,          // its goal: the model, where it last stood over the floor (Env.track_target)
    /// An action given from outside for the next tick (a model plays it: self-play), instead of its bot's.
    pub next: Option<[i64; ACTIONS]>,
    /// This tick (fight_tick): its left click (Some(whether it was on the model in reach)) and its hit that counted
    /// (feet distance across, critical). The reward is zero-sum, so these count against the model's (shaping.rs).
    pub swing: Option<bool>,
    pub landed: Option<(f64, bool)>,
    /// Played by a model this tick (self-play), not by its bot.
    played: bool,
    /// A model-played opponent's hit, reaching the server next tick as the model's own does (`queued` in fight_tick):
    /// it stands in for another player's client, whose packet comes a tick after the click. A bot acts on the server
    /// and hits at once. Without this the copy on the opponent's side won 64 % of its own mirror fights.
    queued: Option<Swing>,
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
    pub parts: [f64; PARTS.len()],
    pub report: Report,
    /// The fight's aim potential after the last tick (shaping::potential), for the next tick's γ·Φ(s′) − Φ(s).
    potential: f64,
    /// How much the event bonuses (long hit, critical hit, missed swing) count, from 1 down to 0 as the model gets
    /// stronger against the references (train_auto.Progress; docs/training-design.md#nedtrappning). Never observed.
    pub bonus_weight: f64,
    hit: Option<Hit>,
    k: i32,
    tick_count: i64,
    next_push: usize,
    pushed: bool,        // a push came and it has not landed again yet
    push_airborne: bool, // it has left the ground since
    pushes_given: i32,
    pub opponents: Vec<Opponent>,
    history2: [[f32; HIST_F]; HISTORY],
    queued: Option<(Swing, usize)>, // the model's hit, on its way to the server
    target: Option<[f64; 3]>,       // in a fight, the goal: the nearest opponent where it last stood over the floor
    since_hit: i32,
    since_hurt: i32,
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
        player.rules18 = course.rules18;
        if let Some(d) = &course.duel {
            player.attack_damage = d.damage;
            player.armor = d.armor;
            player.kb_enchant = d.kb;
            player.slot = 1; // a fight starts with the sword in hand
        }
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
            parts: [0.0; PARTS.len()],
            report: Report::default(),
            potential: 0.0,
            bonus_weight: 1.0,
            hit: None,
            k,
            tick_count,
            next_push: 0,
            pushed: false,
            push_airborne: false,
            pushes_given: 0,
            opponents: Vec::new(),
            history2: [[0.0; HIST_F]; HISTORY],
            queued: None,
            target: None,
            since_hit: 99,
            since_hurt: 99,
            course,
        };
        if let Some(d) = env.course.duel.clone() {
            for r in &d.opponents {
                env.opponents.push(Opponent::new(r, env.course.rules18, env.course.blocks));
            }
        }
        env.track_target();
        env.potential = env.fight_potential();
        env.best = env.distance();
        env.ground_distance = env.best;
        env.hit = env.player.pick(&env.world);
        env
    }

    /// shaping::potential for this round: the model against the opponents still in the fight. 0 outside fights.
    fn fight_potential(&self) -> f64 {
        if self.course.duel.is_none() {
            return 0.0;
        }
        let foes: Vec<(&Player, bool)> = self.opponents.iter().map(|o| (&o.player, !o.out)).collect();
        shaping::potential(&self.player, &foes)
    }

    /// AutoEnv._target: in a fight, the goal moves to the nearest opponent still in it that stands over the arena's
    /// floor. One going over the edge leaves it where it last was over the floor, so the goal is never out over the void.
    fn track_target(&mut self) {
        if self.course.duel.is_none() {
            return;
        }
        let me = self.player.pos;
        let world = &self.world;
        let over_floor = |p: &[f64; 3]| world.present(p[0].floor() as i32, Y, p[2].floor() as i32);
        if let Some(t) = self.opponents.iter().filter(|o| !o.gone() && over_floor(&o.player.pos)).map(|o| o.player.pos).min_by(|a, b| duel::dist2(a, &me).total_cmp(&duel::dist2(b, &me))) {
            self.target = Some(t);
        }
        // Each opponent's goal is the model, by the same rule (what it sees when a model plays it).
        if over_floor(&me) {
            for o in self.opponents.iter_mut() {
                o.target = Some(me);
            }
        }
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

    fn press(&mut self, wanted: [bool; 7]) {
        limit_keys(&mut self.keys, &mut self.key_age, wanted);
    }

    fn release_inputs(&mut self) {
        self.keys = [false; 7];
        self.key_age = [MIN_KEY_TICKS; 7];
        self.since_sneak = SNEAK_CLICK_TICKS + 1;
        self.jump_wait = 0;
        self.was_on_ground = true;
    }

    fn game_keys(&self) -> Keys {
        to_game_keys(&self.keys)
    }

    fn game_tick(&mut self, input: &Input) -> crate::player::Events {
        let ev = self.player.tick(&mut self.world, input);
        self.tick_count += 1;
        ev
    }

    /// One decision: AutoEnv.step with one tick. Returns the reward and whether the round is over.
    pub fn step(&mut self, a: [i64; ACTIONS]) -> (f64, bool) {
        let (mv, strafe, sneak, use_, jump, sprint, yaw, pitch) = (a[0], a[1], a[2] != 0, a[3] != 0, a[4] != 0, a[5] != 0, a[6] as usize, a[7] as usize);
        let attack = a[8] != 0;
        self.history2.rotate_left(1);
        self.history2[HISTORY - 1] = [attack as i32 as f32];
        self.history.rotate_left(1);
        self.history[HISTORY - 1] = [mv as f32 / 2.0, strafe as f32 / 2.0, sneak as i32 as f32, use_ as i32 as f32 / 2.0, jump as i32 as f32, sprint as i32 as f32, yaw as f32 / 10.0, pitch as f32 / 10.0];
        let held = self.player.held;
        let health = self.player.health;
        let before = self.game_keys();
        self.press([mv == 1, mv == 2, strafe == 1, strafe == 2, sneak, jump, sprint]);
        self.since_sneak = if self.keys[K_SNEAK] { 0 } else { self.since_sneak + 1 };
        self.sneak_ticks += self.keys[K_SNEAK] as i32;
        let clicked = use_ && self.since_sneak <= SNEAK_CLICK_TICKS;
        let input = Input { keys: self.game_keys(), click: clicked, turn_yaw: TURN_STEPS[yaw], turn_pitch: TURN_STEPS[pitch], attack, slot: None };
        self.maybe_push(before, &input);
        if self.course.duel.is_some() {
            return self.fight_step(&input);
        }
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
            reward += self.closer(target);
        }
        if self.course.blocks > 0 && on_ground {
            // The block it stands on counts the same as one it placed (AutoEnv._tick).
            let p = self.player.pos;
            reward += self.closer([p[0].floor() as i32, p[1].floor() as i32 - 1, p[2].floor() as i32]);
        }
        let success = self.reached();
        let fell = (self.fell() || self.player.health <= DEATH_HEALTH) && !success; // a fall that (nearly) kills is a death too
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

    /// One tick of a fight: every player's left click is worked out before anyone moves, then everyone moves, then the
    /// server handles the hits with the places it knew when the clicks came (combat.rs). AutoEnv._fight_tick.
    fn fight_step(&mut self, input: &Input) -> (f64, bool) {
        let centre = self.course.duel.as_ref().map_or([0.0; 3], |d| d.centre);
        let health_before: Vec<f32> = self.opponents.iter().map(|o| o.player.health).collect();
        let my_health = self.player.shown_health();
        let ev = fight_tick(&mut self.world, &mut self.player, &mut self.opponents, input, centre, self.t, &mut self.since_hit, &mut self.since_hurt, &mut self.queued);
        self.track_target();
        self.tick_count += 1;
        self.t += 1;
        self.hit = self.player.pick(&self.world);
        // Points: what was dealt and taken, the result, and the shaping (shaping.rs: zero-sum, the event bonuses
        // weighted, aim as a potential).
        let mut reward = self.points(P_TIME, -TICK_COST);
        let mine = shaping::events(ev.landed, if input.attack { Some(ev.struck) } else { None });
        let theirs: Vec<[f64; 3]> = self.opponents.iter().map(|o| shaping::events(o.landed, o.swing)).collect();
        let [long, crit, swing] = shaping::zero_sum(mine, &theirs, self.opponents.len(), self.bonus_weight);
        reward += self.points(P_LONG_HIT, long);
        reward += self.points(P_CRIT, crit);
        reward += self.points(P_SWING, swing);
        let dealt: f32 = self.opponents.iter().zip(&health_before).map(|(o, &h)| (h - o.player.health).max(0.0)).sum();
        let taken = (my_health - self.player.shown_health()).max(0.0);
        reward += self.points(P_DEALT, HEALTH_POINTS * dealt as f64);
        reward += self.points(P_TAKEN, -HEALTH_POINTS * taken as f64);
        self.damage += taken as f64;
        let line = self.fall_line();
        for o in self.opponents.iter_mut() {
            if !o.out && (o.player.health <= 0.0 || o.player.pos[1] < line) {
                o.out = true;
            }
        }
        // Won only alive and standing: knocking the last opponent off by falling after it is a loss, however late the
        // fall would count.
        let won = self.opponents.iter().all(|o| o.out) && self.player.on_ground && self.player.pos[1] >= line && self.player.shown_health() > 0.0;
        let lost = !won && (self.player.shown_health() <= 0.0 || self.player.pos[1] < line);
        if won {
            reward += self.points(P_WON, FIGHT_RESULT);
        }
        if lost {
            reward += self.points(P_LOST, -FIGHT_RESULT);
            self.deaths += 1;
        }
        let done = won || lost || self.t >= self.course.limit;
        // Φ is 0 once the round is over, so over a round the aim points add up to about −Φ(start): nothing to farm.
        let now = if done { 0.0 } else { self.fight_potential() };
        reward += self.points(P_AIM, shaping::GAMMA * now - self.potential);
        self.potential = now;
        self.report = Report {
            success: won,
            clean: won,
            deaths: self.deaths,
            damage: self.damage,
            t: self.t,
            pushes: self.pushes_given,
            ..Report::default()
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
        let old = std::mem::replace(&mut self.player, Player::new(h, self.course.home_yaw, self.course.home_pitch, held));
        self.player.keep_gear(&old); // the inventory and the rules stay; the mod's respawn only moves the player
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
        let state = clutch::State { pos: p.pos, vel: p.dm, yaw: p.yaw, pitch: p.pitch, held: p.held, keys: before, sprinting: p.sprinting };
        let goal = [self.course.goal[0] as f64 + 0.5, self.course.goal[1] as f64 + 0.5];
        if let Some(v) = clutch::decide(&cube, &state, input, goal, self.fall_line(), &pushes[self.next_push]) {
            clutch::apply(&mut self.player, v);
            self.next_push += 1;
            self.pushed = true;
            self.push_airborne = false;
            self.pushes_given += 1;
        }
    }

    /// AutoEnv._closer: points for a block closer to the goal than any so far, by how much closer.
    fn closer(&mut self, c: [i32; 3]) -> f64 {
        let reach = climb(((self.course.goal[0] - c[0]) as f64).hypot((self.course.goal[1] - c[2]) as f64), (self.course.dy - c[1] + Y) as f64);
        if reach >= self.closest_block {
            return 0.0;
        }
        let gained = self.closest_block - reach;
        self.closest_block = reach;
        self.points(P_PLACED, PLACE_BONUS * gained)
    }

    fn fall_line(&self) -> f64 {
        ((Y + 1).min(Y + 1 + self.course.dy)) as f64 - self.course.fall_margin
    }

    // ---- What it sees ----

    /// AutoEnv._vec, _with_timer and the history: everything the model gets, into `out`.
    pub fn observe(&self, out: &mut [f32]) {
        let p = &self.player;
        let me = duel::Me {
            slot: p.slot,
            resist: p.resist,
            damage: p.attack_damage,
            armor: p.armor,
            server_sprinting: p.server_sprinting,
            since_hit: self.since_hit,
            since_hurt: self.since_hurt,
            duel: self.course.duel.is_some(),
        };
        let eyes = Eyes {
            p,
            k: self.k,
            target: self.target,
            hit: self.hit,
            keys: &self.keys,
            jump_wait: self.jump_wait,
            visited: &self.visited,
            history: &self.history,
            history2: &self.history2,
            me,
            // as they are: one out still falls, a dead one has no health
            others: self.opponents.iter().map(|o| View::of(&o.player)).collect(),
        };
        self.observe_with(&eyes, out);
    }

    /// What opponent `j` sees, by the model's own rules from its side (self-play: a model plays it): its frame turned
    /// half round (it faces the model as the model faces it), the model as its goal, its own inputs and fighting
    /// state, the model as the one opponent it sees (a second opponent, its teammate, is not shown). Its health is
    /// the server's (it has no client to show damage late); in fights no one's jump wait or stood-on cells are kept.
    pub fn observe_opponent(&self, j: usize, out: &mut [f32]) {
        let o = &self.opponents[j];
        let p = &o.player;
        let me = duel::Me {
            slot: p.slot,
            resist: p.resist,
            damage: p.attack_damage,
            armor: p.armor,
            server_sprinting: p.server_sprinting,
            since_hit: o.since_hit,
            since_hurt: o.since_hurt,
            duel: true,
        };
        let mut model = View::of(&self.player);
        model.health = self.player.health; // the server's, as the model sees its opponents' health
        let no_cells = HashSet::new();
        let eyes = Eyes {
            p,
            k: (self.k + 2).rem_euclid(4),
            target: o.target,
            hit: p.pick(&self.world),
            keys: &o.keys,
            jump_wait: 0,
            visited: &no_cells,
            history: &o.history,
            history2: &o.history2,
            me,
            others: vec![model],
        };
        self.observe_with(&eyes, out);
    }

    /// Everything a player sees, from `e`'s side, into `out` (observe, observe_opponent).
    fn observe_with(&self, e: &Eyes, out: &mut [f32]) {
        debug_assert_eq!(out.len(), N_OBS);
        let p = e.p;
        let c = &self.course;
        let [x, y, z] = p.pos;
        let [vx, vy, vz] = p.dm;
        let k = e.k;
        let turn = (p.yaw as f64).to_radians();
        let (fw, rt) = ((-turn.sin(), turn.cos()), (-turn.cos(), -turn.sin()));
        let ego = |dx: f64, dz: f64| (dx * fw.0 + dz * fw.1, dx * rt.0 + dz * rt.1);
        // The goal: the course's, or in a fight the nearest opponent over the floor (track_target).
        let (gx, gz, goal_up) = match e.target {
            Some(t) => (t[0] - x, t[2] - z, t[1] - y),
            _ => (c.goal[0] as f64 + 0.5 - x, c.goal[1] as f64 + 0.5 - z, (Y + 1 + c.dy) as f64 - y),
        };
        let distance = gx.hypot(gz);
        let (goal_f, goal_r) = ego(gx, gz);
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
        head[15] = p.shown_health() as f64 / 20.0;
        head[16] = p.horizontal_collision as i32 as f64;
        head[17] = p.fall_distance.ln_1p() / 3.0;
        head[18] = p.hurt_time as f64 / 10.0;
        if let Some(h) = &e.hit {
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
        head[33] = can_place(&self.world, p, &e.hit) as i32 as f64;
        for i in 0..HEAD {
            out[i] = (head[i] as f32).clamp(-5.0, 5.0);
        }
        // _with_timer
        let ks = e.keys;
        let known = [
            ks[K_SNEAK] as i32 as f32,
            (1.0 - self.t as f64 / c.limit as f64).max(0.0) as f32,
            ks[K_SPRINT] as i32 as f32,
            ks[K_JUMP] as i32 as f32,
            ks[K_FORWARD] as i32 as f32,
            ks[K_BACK] as i32 as f32,
            ks[K_LEFT] as i32 as f32,
            ks[K_RIGHT] as i32 as f32,
            (e.jump_wait as f64 / JUMP_WAIT as f64) as f32,
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
        for &(cx_, cz_) in e.visited {
            let (rx, rz) = rotate_i(cx_ - feet[0], cz_ - feet[2], k);
            if rx.abs() <= MAP_RADIUS && rz.abs() <= MAP_RADIUS {
                beens[((rx + MAP_RADIUS) * MAP_SIDE + rz + MAP_RADIUS) as usize] = 1.0;
            }
        }
        let mut i = HISTORY_START;
        for h in e.history {
            out[i..i + 8].copy_from_slice(h);
            i += 8;
        }
        // duel.features: the opponents, its own fighting state, the command, its last left clicks.
        duel::features(p, &e.me, &e.others, k, &|| Cube::from_fn(feet, |x, y, z| self.world.present(x, y, z)), e.history2, &mut out[MOVE_OBS..]);
    }
}

/// One player's side of what it sees (Env.observe_with): itself, its frame, its goal, what it aims at, its own keys
/// and inputs, its fighting state and the others as it sees them.
struct Eyes<'a> {
    p: &'a Player,
    k: i32,
    target: Option<[f64; 3]>,
    hit: Option<Hit>,
    keys: &'a [bool; 7],
    jump_wait: i32,
    visited: &'a HashSet<(i32, i32)>,
    history: &'a [[f32; 8]; HISTORY],
    history2: &'a [[f32; HIST_F]; HISTORY],
    me: duel::Me,
    others: Vec<View>,
}

/// AutoEnv._can_place: a block in hand, aimed at a side in reach, the spot there empty and not where it stands.
fn can_place(world: &World, p: &Player, hit: &Option<Hit>) -> bool {
    {
        let Some(h) = hit else { return false };
        if p.held <= 0 {
            return false;
        }
        let s = FACE_STEP[h.face as usize];
        let t = [h.block[0] + s[0], h.block[1] + s[1], h.block[2] + s[2]];
        let [x, y, z] = p.pos;
        let feet = [x.floor() as i32, y.floor() as i32, z.floor() as i32];
        let cell = [t[0] - feet[0], t[1] - feet[1], t[2] - feet[2]];
        if cell.iter().all(|c| c.abs() <= RADIUS) && world.present(t[0], t[1], t[2]) {
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

impl Opponent {
    /// Its action through the same key limits as the model's (Env.press, the shift rule for clicks).
    fn input(&mut self, a: [i64; ACTIONS]) -> Input {
        self.history2.rotate_left(1);
        self.history2[HISTORY - 1] = [(a[8] != 0) as i32 as f32];
        self.history.rotate_left(1);
        self.history[HISTORY - 1] = [a[0] as f32 / 2.0, a[1] as f32 / 2.0, (a[2] != 0) as i32 as f32, (a[3] != 0) as i32 as f32 / 2.0, (a[4] != 0) as i32 as f32,
                                     (a[5] != 0) as i32 as f32, a[6] as f32 / 10.0, a[7] as f32 / 10.0];
        let wanted = [a[0] == 1, a[0] == 2, a[1] == 1, a[1] == 2, a[2] != 0, a[4] != 0, a[5] != 0];
        limit_keys(&mut self.keys, &mut self.key_age, wanted);
        self.since_sneak = if self.keys[K_SNEAK] { 0 } else { self.since_sneak + 1 };
        Input {
            keys: to_game_keys(&self.keys),
            click: a[3] != 0 && self.since_sneak <= SNEAK_CLICK_TICKS,
            turn_yaw: TURN_STEPS[a[6] as usize],
            turn_pitch: TURN_STEPS[a[7] as usize],
            attack: a[8] != 0,
            slot: None, // a bot keeps the sword it starts with
        }
    }
}

/// One tick of the game with opponents, in the order the game has them (checked against it, duel_real_vs_sim):
/// bots choose from where everyone is now; the model's hit from last tick's click reaches the server (its packet comes a
/// tick after the click), so an opponent it knocks flies off this tick; every left click is worked out before anyone
/// moves; everyone moves; the opponents' hits land, their motion reaching the model before its next tick; counters
/// count up last, so a hit's own tick shows 1, as the server's do. `queued` carries the model's hit to the next tick.
#[allow(clippy::too_many_arguments)]
pub fn fight_tick(world: &mut World, player: &mut Player, opponents: &mut [Opponent], input: &Input, centre: [f64; 3], t: i32, since_hit: &mut i32, since_hurt: &mut i32,
                  queued: &mut Option<(Swing, usize)>) -> crate::player::Events {
    player.fighting = true;
    let me = View::of(player);
    let mut inputs = Vec::with_capacity(opponents.len());
    for o in opponents.iter_mut() {
        let given = o.next.take(); // self-play: a model's action for this opponent
        o.swing = None;
        o.landed = None;
        o.played = given.is_some();
        let a = if o.out { [0, 0, 0, 0, 0, 0, 5, 5, 0] } else if let Some(a) = given { a } else { duel::action(o.bot, &View::of(&o.player), &me, centre, t, o.since_hit, &|x, y, z| world.present(x, y, z)) };
        inputs.push(o.input(a));
    }
    let mut landed = None;
    if let Some((s, j)) = queued.take() {
        if !opponents[j].dead() {
            let at = opponents[j].player.pos;
            let (counted, broke) = combat::hit(&s, &mut opponents[j].player, at);
            if counted {
                *since_hit = 0;
                opponents[j].since_hurt = 0;
                landed = Some(((s.pos[0] - at[0]).hypot(s.pos[2] - at[2]), s.falling));
            }
            if broke {
                player.server_sprinting = false;
            }
        }
    }
    // Model-played opponents' hits from last tick's clicks reach the server now, as the model's do.
    let at_start = player.health;
    for o in opponents.iter_mut() {
        if let Some(s) = o.queued.take() {
            if !o.dead() {
                let at = player.pos;
                let (counted, broke) = combat::hit(&s, player, at);
                if counted {
                    *since_hurt = 0;
                    o.since_hit = 0;
                    o.landed = Some(((s.pos[0] - at[0]).hypot(s.pos[2] - at[2]), s.falling));
                }
                if broke {
                    o.player.server_sprinting = false;
                }
            }
        }
    }
    let early = at_start - player.health; // shown a tick late, as the other hits are
    player.unseen = 0.0;
    player.begin(input);
    for (o, i) in opponents.iter_mut().zip(&inputs) {
        o.player.begin(i);
    }
    let boxes: Vec<Option<_>> = opponents.iter().map(|o| if o.dead() { None } else { Some(combat::body(&o.player)) }).collect();
    let my_box = Some(combat::body(player));
    let target = if input.attack { combat::click(player, world, &boxes) } else { None };
    let struck = target.is_some();
    if let Some(j) = target {
        *queued = Some((Swing::of(player), j));
        combat::recoil(player);
    }
    let my_pos = player.pos;
    let mut their_hits = Vec::new();
    for (j, (o, i)) in opponents.iter_mut().zip(&inputs).enumerate() {
        if i.attack && !o.dead() {
            let on = combat::click(&mut o.player, world, &[my_box]).is_some();
            o.swing = Some(on);
            if on {
                if o.played {
                    o.queued = Some(Swing::of(&o.player));
                } else {
                    their_hits.push((j, Swing::of(&o.player)));
                }
                combat::recoil(&mut o.player);
            }
        }
    }
    let mut ev = player.rest(world, input);
    ev.landed = landed;
    ev.struck = struck;
    // Players do not push each other: Hypixel's teams have collisionRule never, and so do the mod's (Fight18.gear).
    for (o, i) in opponents.iter_mut().zip(&inputs) {
        if !o.dead() {
            o.player.rest(world, i); // out of the fight it still falls; the mod's drive() leaves only the dead where they are
            if o.player.pos[1] < VOID_Y {
                o.player.health = (o.player.health - VOID_DAMAGE).max(0.0); // Entity.onBelowWorld: the void kills
            }
        }
    }
    let before_hits = player.health;
    for (j, s) in their_hits {
        let (counted, broke) = combat::hit(&s, player, my_pos);
        if counted {
            *since_hurt = 0;
            opponents[j].since_hit = 0;
            opponents[j].landed = Some(((s.pos[0] - my_pos[0]).hypot(s.pos[2] - my_pos[2]), s.falling));
        }
        if broke {
            opponents[j].player.server_sprinting = false;
        }
    }
    player.unseen = early + before_hits - player.health;
    player.count_down(); // Fight18.serverTick: hurt resistance and healing after the bots' hits
    *since_hit += 1;
    *since_hurt += 1;
    for o in opponents.iter_mut() {
        o.since_hit += 1;
        o.since_hurt += 1;
    }
    ev
}

impl Opponent {
    /// Out of the fight, or killed this tick by the model's hit (which lands before it would act).
    pub fn gone(&self) -> bool {
        self.out || self.dead()
    }

    /// Dead: no longer in the world as a player (it still is while it falls out of the fight).
    pub fn dead(&self) -> bool {
        self.player.health <= 0.0
    }

    /// An opponent standing where a round puts it, as after the mod's reset.
    /// `blocks`: as many as the model has. A scripted bot never places one (duel::action never right-clicks), but a
    /// model playing it (self-play) has learned to save itself with them: without, a copy of the model fell off three
    /// times as often from the opponent's side as from its own, and lost 73 % of the fights against itself.
    pub fn new(r: &Rival, rules18: bool, blocks: i32) -> Opponent {
        let mut p = Player::new(r.pos, r.yaw, 0.0, blocks);
        settle(&mut p);
        p.rules18 = rules18;
        p.attack_damage = r.damage;
        p.armor = r.armor;
        p.kb_enchant = r.kb;
        p.slot = 1;
        p.direct_turn = true; // Fight18.drive adds the turn step itself
        Opponent {
            player: p,
            bot: r.bot,
            out: false,
            keys: [false; 7],
            key_age: [MIN_KEY_TICKS; 7],
            since_sneak: SNEAK_CLICK_TICKS + 1,
            since_hit: 99,
            since_hurt: 99,
            history: [[0.0; 8]; HISTORY],
            history2: [[0.0; HIST_F]; HISTORY],
            target: None,
            next: None,
            swing: None,
            landed: None,
            played: false,
            queued: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::FULL;

    /// A floor at y 99 and the model at (0.5, 100, 0.5) facing +z; one opponent `ahead` blocks in front, facing it.
    fn arena(bot: Bot, ahead: f64) -> (World, Player, Vec<Opponent>) {
        let mut w = World::new([-10, 90, -10], [10, 110, 10]);
        w.fill([-10, 99, -10], [10, 99, 10], FULL);
        let mut p = Player::new([0.5, 100.0, 0.5], 0.0, 0.0, 0);
        settle(&mut p);
        p.rules18 = true;
        let o = Opponent::new(&Rival { bot, pos: [0.5, 100.0, 0.5 + ahead], yaw: 180.0, damage: 1.0, armor: 0, kb: 0 }, true, 0);
        (w, p, vec![o])
    }

    fn tick(w: &mut World, p: &mut Player, o: &mut [Opponent], t: i32, queued: &mut Option<(Swing, usize)>) {
        let (mut since_hit, mut since_hurt) = (99, 99);
        fight_tick(w, p, o, &Input::default(), [0.5, 100.0, 0.5], t, &mut since_hit, &mut since_hurt, queued);
    }

    /// A fight laid out mirror-symmetric: a platform round the middle (z = 2.5), the model at z 0.5 facing +z, its
    /// opponent at z 4.5 facing it.
    fn mirror_duel() -> Env {
        let mut w = World::new([-12, 90, -12], [12, 110, 16]);
        w.fill([-8, Y, -8], [8, Y, 12], FULL);
        let feet = (Y + 1) as f64;
        let course = Course {
            hop: false,
            goal: [0, 4],
            dy: 0,
            goal_half: 0,
            fall_margin: 3.0,
            landing: vec![],
            par: 0,
            limit: 600,
            no_blocks: false,
            field: None,
            home: [0.5, feet, 0.5],
            home_yaw: 0.0,
            home_pitch: 0.0,
            blocks: 0, // as the opponent: the same gear on both sides
            world: w,
            settle: false,
            pushes: vec![],
            rules18: true,
            duel: Some(Duel {
                damage: 5.0,
                armor: 7,
                kb: 0,
                centre: [0.5, feet, 2.5],
                opponents: vec![Rival { bot: Bot::Dummy, pos: [0.5, feet, 4.5], yaw: 180.0, damage: 5.0, armor: 7, kb: 0 }],
            }),
        };
        Env::new(course, 0)
    }

    fn assert_same(a: &[f32], b: &[f32], what: &str) {
        let worst = a.iter().zip(b).enumerate().map(|(i, (x, y))| ((x - y).abs(), i)).fold((0.0f32, 0), |m, d| if d.0 > m.0 { d } else { m });
        assert!(worst.0 < 1e-5, "{what}: input {} differs by {}", worst.1, worst.0);
    }

    /// An opponent played by a model sees by exactly the model's rules: mirrored, the two see the same numbers (the
    /// proof that its inputs are as true as the model's). Also after moving the same way.
    #[test]
    fn an_opponent_sees_as_the_model_does() {
        let mut env = mirror_duel();
        let (mut mine, mut theirs) = (vec![0.0; N_OBS], vec![0.0; N_OBS]);
        env.observe(&mut mine);
        env.observe_opponent(0, &mut theirs);
        assert_same(&mine, &theirs, "at the start");
        let step = [1, 1, 0, 0, 0, 1, 5, 5, 0]; // forward, left, sprint: no turn (the model turns through the mouse)
        for _ in 0..6 {
            env.opponents[0].next = Some(step);
            env.step(step);
        }
        env.observe(&mut mine);
        env.observe_opponent(0, &mut theirs);
        assert_same(&mine, &theirs, "after moving the same way");
    }

    /// Out of the fight an opponent is still a player: it keeps falling, and below the world the void kills it.
    #[test]
    fn out_still_falls_until_the_void_kills_it() {
        let (mut w, mut p, mut o) = arena(Bot::Walker, 2.0);
        o[0].out = true;
        o[0].player.pos = [20.5, -125.0, 0.5];
        o[0].player.on_ground = false;
        let start = o[0].player.pos[1];
        let (mut since_hit, mut since_hurt) = (99, 99);
        for t in 0..3 {
            fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], t, &mut since_hit, &mut since_hurt, &mut None);
        }
        assert!(o[0].player.pos[1] < start, "it falls");
        for t in 3..20 {
            fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], t, &mut since_hit, &mut since_hurt, &mut None);
        }
        assert!(o[0].dead(), "the void killed it: {}", o[0].player.health);
    }

    /// The model's queued hit lands first: an opponent it kills neither hits back nor moves that tick (Fight18.drive
    /// skips the dead).
    #[test]
    fn killed_by_the_queued_hit_does_nothing_more() {
        let (mut w, mut p, mut o) = arena(Bot::Walker, 2.0);
        o[0].player.health = 1.0;
        let at = o[0].player.pos;
        let mut queued = Some((Swing::of(&p), 0));
        tick(&mut w, &mut p, &mut o, 0, &mut queued);
        assert_eq!(o[0].player.health, 0.0);
        assert!(o[0].gone());
        assert_eq!(p.health, 20.0, "a dead bot does not hit");
        assert_eq!(o[0].player.pos, at, "nor move");
    }

    /// A bot's hit: the model's server counts resistance down after it (19 at the end of the tick, as the mod's
    /// tickState), and its client shows the damage a tick later.
    #[test]
    fn a_bot_hit_counts_and_shows_late() {
        let (mut w, mut p, mut o) = arena(Bot::Walker, 2.0);
        let mut queued = None;
        tick(&mut w, &mut p, &mut o, 0, &mut queued);
        assert_eq!(p.health, 19.0, "the walker's first click hits");
        assert_eq!(p.resist, 19);
        assert_eq!(p.hurt_time, 10);
        assert_eq!(p.shown_health(), 20.0, "shown next tick");
        tick(&mut w, &mut p, &mut o, 1, &mut queued);
        assert_eq!(p.shown_health(), 19.0);
        assert_eq!(p.resist, 18);
    }

    /// The model's hit that counts is reported with the distance across (where the server had both) and whether it
    /// was critical: what the long-hit and critical-hit points are worked out from.
    #[test]
    fn a_landed_hit_tells_how_far_and_how() {
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 2.5);
        let mut queued = Some((Swing::of(&p), 0));
        let (mut since_hit, mut since_hurt) = (99, 99);
        let ev = fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], 0, &mut since_hit, &mut since_hurt, &mut queued);
        let (across, critical) = ev.landed.expect("the hit counts");
        assert!((across - 2.5).abs() < 1e-9 && !critical, "{across} {critical}");
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 2.5);
        p.fall_distance = 0.5;
        p.on_ground = false;
        let mut queued = Some((Swing::of(&p), 0));
        let ev = fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], 0, &mut since_hit, &mut since_hurt, &mut queued);
        assert!(ev.landed.is_some_and(|(_, c)| c), "a hit while falling is critical");
        let ev = fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], 1, &mut since_hit, &mut since_hurt, &mut None);
        assert!(ev.landed.is_none(), "nothing landed this tick");
    }

    /// A left click on the opponent in front is a strike; one turned away from it is a miss (what the missed-swing cost
    /// goes by).
    #[test]
    fn a_click_strikes_or_misses() {
        let click = Input { attack: true, ..Input::default() };
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 2.0);
        let (mut since_hit, mut since_hurt) = (99, 99);
        let ev = fight_tick(&mut w, &mut p, &mut o, &click, [0.5, 100.0, 0.5], 0, &mut since_hit, &mut since_hurt, &mut None);
        assert!(ev.struck, "straight ahead, 2 blocks");
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 2.0);
        p.yaw = 90.0;
        let ev = fight_tick(&mut w, &mut p, &mut o, &click, [0.5, 100.0, 0.5], 0, &mut since_hit, &mut since_hurt, &mut None);
        assert!(!ev.struck, "looking the other way");
    }

    /// A bot's hit lands the tick it clicks (it is on the server); a model playing the opponent hits a tick later, as
    /// the model does (its client's packet), so self-play is fair to both sides.
    #[test]
    fn a_model_played_opponent_hits_a_tick_late() {
        let swing = [0, 0, 0, 0, 0, 0, 5, 5, 1];
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 2.0);
        let (mut since_hit, mut since_hurt) = (99, 99);
        o[0].next = Some(swing);
        let full = p.health;
        fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], 0, &mut since_hit, &mut since_hurt, &mut None);
        assert_eq!(o[0].swing, Some(true));
        assert_eq!(p.health, full, "not yet");
        fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], 1, &mut since_hit, &mut since_hurt, &mut None);
        assert!(p.health < full && o[0].landed.is_some(), "the next tick");
        let (mut w, mut p, mut o) = arena(Bot::Walker, 2.0);
        let mut hurt_at = None;
        for t in 0..20 {
            fight_tick(&mut w, &mut p, &mut o, &Input::default(), [0.5, 100.0, 0.5], t, &mut since_hit, &mut since_hurt, &mut None);
            if o[0].swing == Some(true) {
                hurt_at = Some(p.health < full);
                break;
            }
        }
        assert_eq!(hurt_at, Some(true), "a bot's hit lands the tick it clicks");
    }

    /// Bots turn by the step itself, as Fight18.drive adds it; the model turns through the mouse (multiples of 0.15).
    #[test]
    fn bots_turn_directly() {
        let (_, mut p, mut o) = arena(Bot::Dummy, 3.0);
        let yaw = o[0].player.yaw;
        o[0].player.turn(15.0, -60.0);
        assert_eq!(o[0].player.yaw, yaw + 15.0);
        assert_eq!(o[0].player.pitch, -60.0);
        let before = p.yaw;
        p.turn(0.5, 0.0);
        assert_eq!(p.yaw, before + ((0.5f64 / 0.15) as f32 * 0.15f32));
    }

    /// The model's server counters wait in a fight (fight_tick counts them) and run with the move outside one.
    #[test]
    fn counters_once_a_tick() {
        let (mut w, mut p, _) = arena(Bot::Dummy, 3.0);
        p.resist = 5;
        p.tick(&mut w, &Input::default());
        assert_eq!(p.resist, 4, "outside a fight: in the server tick");
        let (mut w, mut p, mut o) = arena(Bot::Dummy, 3.0);
        p.resist = 5;
        tick(&mut w, &mut p, &mut o, 0, &mut None);
        assert_eq!(p.resist, 4, "in a fight: once, after the hits");
    }
}
