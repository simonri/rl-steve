//! The Python side: one player in one world, for checking against the real game tick by tick.

use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::conversion::FromPyObjectOwned;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::clutch::{self, Cube, Push};
use crate::duel::{self, Bot, View, ACTIONS, EXTRA, HIST_F};
use crate::env::{fight_tick, Course, Duel, Env, Field, Opponent, Rival, HISTORY, MOVE_OBS, N_OBS, PARTS};
use crate::player::{Input, Keys, Player, FACE_NAMES};
use crate::world::{kind_for, World};

#[pyclass(unsendable)]
pub struct Single {
    world: World,
    player: Player,
    opponents: Vec<Opponent>,
    centre: [f64; 3],
    t: i32,
    since_hit: i32,
    since_hurt: i32,
    fall_line: f64, // below this an opponent is out, as in Env.fight_step
}

#[pymethods]
impl Single {
    /// boxes: filled boxes [x0, y0, z0, x1, y1, z1] of plain blocks. cells: single blocks (x, y, z, state name).
    #[new]
    #[pyo3(signature = (boxes, cells, min, max))]
    fn new(boxes: Vec<[i32; 6]>, cells: Vec<(i32, i32, i32, String)>, min: [i32; 3], max: [i32; 3]) -> Self {
        let mut world = World::new(min, max);
        for b in boxes {
            world.fill([b[0], b[1], b[2]], [b[3], b[4], b[5]], crate::world::FULL);
        }
        for (x, y, z, name) in cells {
            world.set(x, y, z, kind_for(&name));
        }
        Single { world, player: Player::new([0.0; 3], 0.0, 0.0, 0), opponents: Vec::new(), centre: [0.0; 3], t: 0, since_hit: 99, since_hurt: 99, fall_line: f64::MIN }
    }

    /// A new player standing at a spot, as after the mod's reset: nothing kept from before.
    fn spawn(&mut self, pos: [f64; 3], yaw: f32, pitch: f32, held: i32) {
        let mut p = Player::new(pos, yaw, pitch, held);
        p.on_ground = true;
        p.dm = [0.0, -0.0784, 0.0];
        p.keep_gear(&self.player); // as Env.respawn: the inventory and the rules stay
        self.player = p;
    }

    /// A hit's knockback, as clutch.rs gives it.
    fn push(&mut self, vel: [f64; 3]) {
        clutch::apply(&mut self.player, vel);
    }

    /// Whether the scripted clutch saves this push in the whole world, not just the cube the model sees: the check
    /// that a push the cube allowed really can be saved.
    #[pyo3(signature = (vel, before, keys, click, turn_yaw, turn_pitch, fall_line))]
    #[allow(clippy::too_many_arguments)]
    fn can_save(&self, vel: [f64; 3], before: [bool; 7], keys: [bool; 7], click: bool, turn_yaw: f64, turn_pitch: f64, fall_line: f64) -> bool {
        let p = &self.player;
        let feet = [p.pos[0].floor() as i32, p.pos[1].floor() as i32, p.pos[2].floor() as i32];
        let state = clutch::State { pos: p.pos, vel: p.dm, yaw: p.yaw, pitch: p.pitch, held: p.held, keys: to_keys(before), sprinting: p.sprinting };
        clutch::can_save(&self.world, feet, &state, vel, &Input { keys: to_keys(keys), click, turn_yaw, turn_pitch, ..Input::default() }, fall_line)
    }

    /// The player's gear and hand: attack attribute (1 plus the sword), armor points, Knockback level, slot (0 blocks,
    /// 1 sword), and whether healing follows 1.8.
    #[pyo3(signature = (damage, armor, kb, slot, rules18))]
    fn set_gear(&mut self, damage: f32, armor: i32, kb: i32, slot: u8, rules18: bool) {
        let p = &mut self.player;
        (p.attack_damage, p.armor, p.kb_enchant, p.slot, p.rules18) = (damage, armor, kb, slot, rules18);
    }

    /// An opponent played by a scripted bot (duel.Bot by index), as the round's reset puts it.
    #[pyo3(signature = (bot, pos, yaw, damage, armor, kb, rules18, centre, fall_line))]
    #[allow(clippy::too_many_arguments)]
    fn add_opponent(&mut self, bot: i32, pos: [f64; 3], yaw: f32, damage: f32, armor: i32, kb: i32, rules18: bool, centre: [f64; 3], fall_line: f64) {
        self.fall_line = fall_line;
        self.opponents.push(Opponent::new(&Rival { bot: Bot::from_index(bot), pos, yaw, damage, armor, kb }, rules18));
        self.centre = centre;
        self.t = 0;
        self.since_hit = 99;
        self.since_hurt = 99;
    }

    fn set_block(&mut self, x: i32, y: i32, z: i32, name: &str) {
        self.world.set(x, y, z, kind_for(name));
    }

    /// Sets what the game reports, plus what it keeps inside: the keys held last tick and the pose.
    #[pyo3(signature = (pos, vel, yaw, pitch, on_ground, held, health=20.0, fall_distance=0.0, sprinting=false, hurt_time=0, keys=None, crouch_pose=None, collision=false))]
    #[allow(clippy::too_many_arguments)]
    fn set_state(&mut self, pos: [f64; 3], vel: [f64; 3], yaw: f32, pitch: f32, on_ground: bool, held: i32, health: f32, fall_distance: f64, sprinting: bool, hurt_time: i32,
                 keys: Option<[bool; 7]>, crouch_pose: Option<bool>, collision: bool) {
        let p = &mut self.player;
        p.pos = pos;
        p.dm = vel;
        p.yaw = yaw;
        p.pitch = pitch;
        p.on_ground = on_ground;
        p.held = held;
        p.health = health;
        p.fall_distance = fall_distance;
        p.server_fall = fall_distance;
        p.sprinting = sprinting;
        p.hurt_time = hurt_time;
        p.horizontal_collision = collision;
        if let Some(k) = keys {
            p.keys = to_keys(k);
            let left = (k[2] as i32 - k[3] as i32) as f32;
            let fwd = (k[0] as i32 - k[1] as i32) as f32;
            let len = (left * left + fwd * fwd).sqrt();
            p.move_vec = if len < 1.0e-4 { [0.0, 0.0] } else { [left / len, fwd / len] };
        }
        if let Some(c) = crouch_pose {
            p.crouch_pose = c;
        }
    }

    /// One tick: keys (forward, back, left, right, jump, sneak, sprint), a click, and the turn in degrees.
    #[pyo3(signature = (keys, click, yaw, pitch, attack=false, slot=None))]
    fn tick<'py>(&mut self, py: Python<'py>, keys: [bool; 7], click: bool, yaw: f64, pitch: f64, attack: bool, slot: Option<u8>) -> PyResult<Bound<'py, PyDict>> {
        let input = Input { keys: to_keys(keys), click, turn_yaw: yaw, turn_pitch: pitch, attack, slot };
        let ev = if self.opponents.is_empty() {
            self.player.tick(&mut self.world, &input)
        } else {
            let ev = fight_tick(&mut self.world, &mut self.player, &mut self.opponents, &input, self.centre, self.t, &mut self.since_hit, &mut self.since_hurt);
            self.t += 1;
            for o in self.opponents.iter_mut() {
                if !o.out && (o.player.health <= 0.0 || o.player.pos[1] < self.fall_line) {
                    o.out = true;
                }
            }
            ev
        };
        let d = self.state(py)?;
        if let Some((t, s)) = ev.placed {
            d.set_item("placed", (t, s))?;
        }
        d.set_item("damage", ev.damage)?;
        Ok(d)
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.player;
        let d = PyDict::new(py);
        d.set_item("pos", p.pos)?;
        d.set_item("vel", p.dm)?;
        d.set_item("yaw", p.yaw)?;
        d.set_item("pitch", p.pitch)?;
        d.set_item("on_ground", p.on_ground)?;
        d.set_item("sneaking", p.keys.shift)?;
        d.set_item("sprinting", p.sprinting)?;
        d.set_item("collision", p.horizontal_collision)?;
        d.set_item("fall_distance", p.fall_distance)?;
        d.set_item("held_count", p.held)?;
        d.set_item("health", p.health)?;
        d.set_item("hurt_time", p.hurt_time)?;
        d.set_item("crouch_pose", p.crouch_pose)?;
        d.set_item("slot", p.slot)?;
        d.set_item("miss_lock", p.miss_lock)?;
        d.set_item("resist", p.resist)?;
        d.set_item("attack_damage", p.attack_damage)?;
        d.set_item("armor", p.armor)?;
        d.set_item("server_sprinting", p.server_sprinting)?;
        d.set_item("since_hit", self.since_hit)?;
        d.set_item("since_hurt", self.since_hurt)?;
        let players = pyo3::types::PyList::empty(py);
        for o in &self.opponents {
            let q = &o.player;
            let e = PyDict::new(py);
            e.set_item("pos", q.pos)?;
            e.set_item("vel", q.dm)?;
            e.set_item("yaw", q.yaw)?;
            e.set_item("pitch", q.pitch)?;
            e.set_item("health", q.health)?;
            e.set_item("armor", q.armor)?;
            e.set_item("damage", View::of(q).damage)?;
            e.set_item("hurt_time", q.hurt_time)?;
            e.set_item("on_ground", q.on_ground)?;
            e.set_item("sprinting", q.sprinting)?;
            e.set_item("crouching", q.crouch_pose)?;
            e.set_item("out", o.out)?;
            players.append(e)?;
        }
        d.set_item("players", players)?;
        if let Some(h) = p.pick(&self.world) {
            let hit = PyDict::new(py);
            hit.set_item("type", "block")?;
            hit.set_item("block", h.block)?;
            hit.set_item("face", FACE_NAMES[h.face as usize])?;
            hit.set_item("loc", h.loc)?;
            d.set_item("hit", hit)?;
        }
        Ok(d)
    }
}

#[pymethods]
impl Single {
    /// The mod's solidAround: '1' where something is, x then y then z, around the block of the feet.
    fn solid_around(&self, r: i32) -> String {
        let p = &self.player.pos;
        let f = [p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32];
        let mut s = String::with_capacity(((2 * r + 1) * (2 * r + 1) * (2 * r + 1)) as usize);
        for x in -r..=r {
            for y in -r..=r {
                for z in -r..=r {
                    s.push(if self.world.present(f[0] + x, f[1] + y, f[2] + z) { '1' } else { '0' });
                }
            }
        }
        s
    }

    /// The mod's surfaceAround: per column the top block from 8 above to 16 below the feet, 'A' at 16 below.
    fn surface_around(&self, r: i32) -> String {
        let p = &self.player.pos;
        let f = [p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32];
        let mut s = String::new();
        for x in -r..=r {
            for z in -r..=r {
                let mut c = '.';
                for y in (-16..=8).rev() {
                    if self.world.present(f[0] + x, f[1] + y, f[2] + z) {
                        c = char::from(b'A' + (y + 16) as u8);
                        break;
                    }
                }
                s.push(c);
            }
        }
        s
    }
}

/// Many training rounds side by side, stepped together on every core.
#[pyclass]
pub struct Batch {
    envs: Vec<Option<Env>>,
    tick_count: i64,
}

fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?.extract::<T>().map_err(Into::into)
}

#[pymethods]
impl Batch {
    #[new]
    fn new(n: usize) -> Self {
        Batch { envs: (0..n).map(|_| None).collect(), tick_count: 0 }
    }

    #[getter]
    fn n_obs(&self) -> usize {
        N_OBS
    }

    /// Starts round `i` from a course drawn in Python (sim_env.CourseMaker.draw).
    fn set_course(&mut self, i: usize, spec: &Bound<'_, PyDict>) -> PyResult<()> {
        let lo: [i32; 3] = get(spec, "lo")?;
        let hi: [i32; 3] = get(spec, "hi")?;
        let mut world = World::new(lo, hi);
        let boxes: Vec<[i32; 6]> = get(spec, "boxes")?;
        for b in boxes {
            world.fill([b[0], b[1], b[2]], [b[3], b[4], b[5]], crate::world::FULL);
        }
        if let Some(cells) = spec.get_item("cells")? {
            if !cells.is_none() {
                let cells: PyReadonlyArray2<i32> = cells.extract()?;
                let c = cells.as_array();
                for row in c.rows() {
                    world.set(row[0], row[1], row[2], row[3] as u8);
                }
            }
        }
        let field = match spec.get_item("field")? {
            Some(f) if !f.is_none() => {
                let f = f.cast::<PyDict>()?;
                let dist: PyReadonlyArray2<f64> = get(f, "dist")?;
                let blocked: PyReadonlyArray2<bool> = get(f, "blocked")?;
                let d = dist.as_array();
                let (nx, nz) = (d.shape()[0] as i32, d.shape()[1] as i32);
                let dist: Vec<f64> = d.iter().copied().collect();
                let max = dist.iter().copied().fold(f64::MIN, f64::max);
                Some(Field { x0: get(f, "x0")?, z0: get(f, "z0")?, nx, nz, dist, blocked: blocked.as_array().iter().copied().collect(), max })
            }
            _ => None,
        };
        let goal: [i32; 2] = get(spec, "goal")?;
        let course = Course {
            hop: get(spec, "hop")?,
            goal,
            dy: get(spec, "dy")?,
            goal_half: get(spec, "goal_half")?,
            fall_margin: get(spec, "fall_margin")?,
            landing: get(spec, "landing")?,
            par: get(spec, "par")?,
            limit: get(spec, "limit")?,
            no_blocks: get(spec, "no_blocks")?,
            field,
            home: get(spec, "home")?,
            home_yaw: get(spec, "yaw")?,
            home_pitch: get(spec, "pitch")?,
            blocks: get(spec, "blocks")?,
            world,
            settle: spec.get_item("settle")?.map(|v| v.extract::<bool>()).transpose()?.unwrap_or(false),
            rules18: spec.get_item("rules18")?.map(|v| v.extract::<bool>()).transpose()?.unwrap_or(false),
            duel: match spec.get_item("duel")? {
                Some(d) if !d.is_none() => {
                    let d = d.cast::<PyDict>()?;
                    let opponents: Vec<(i32, [f64; 3], f32, f32, i32, i32)> = get(d, "opponents")?;
                    Some(Duel {
                        damage: get(d, "damage")?,
                        armor: get(d, "armor")?,
                        kb: get(d, "kb")?,
                        centre: get(d, "centre")?,
                        opponents: opponents.into_iter().map(|(b, pos, yaw, damage, armor, kb)| Rival { bot: Bot::from_index(b), pos, yaw, damage, armor, kb }).collect(),
                    })
                }
                _ => None,
            },
            pushes: match spec.get_item("pushes")? {
                Some(p) if !p.is_none() => p.extract::<Vec<(i32, i32, usize, f64, f64)>>()?.into_iter().map(to_push).collect(),
                _ => Vec::new(),
            },
        };
        self.envs[i] = Some(Env::new(course, self.tick_count));
        Ok(())
    }

    /// Writes what round `i` sees now into `out`.
    fn observe(&self, i: usize, mut out: PyReadwriteArray1<f32>) -> PyResult<()> {
        let env = self.envs[i].as_ref().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no course"))?;
        env.observe(out.as_slice_mut()?);
        Ok(())
    }

    /// One decision in every round at once. Writes what each sees after it, its reward and whether it ended.
    fn step(&mut self, py: Python<'_>, actions: PyReadonlyArray2<i64>, mut obs: PyReadwriteArray2<f32>, mut rewards: PyReadwriteArray1<f32>, mut dones: PyReadwriteArray1<bool>) -> PyResult<()> {
        let acts = actions.as_array();
        let acts: Vec<[i64; ACTIONS]> = acts.rows().into_iter().map(|r| std::array::from_fn(|i| r[i])).collect();
        let obs = obs.as_slice_mut()?;
        let rewards = rewards.as_slice_mut()?;
        let dones = dones.as_slice_mut()?;
        let envs = &mut self.envs;
        py.detach(|| {
            envs.par_iter_mut()
                .zip(obs.par_chunks_mut(N_OBS))
                .zip(rewards.par_iter_mut().zip(dones.par_iter_mut()))
                .zip(acts.par_iter())
                .for_each(|(((env, o), (r, d)), a)| {
                    if let Some(env) = env {
                        let (reward, done) = env.step(*a);
                        env.observe(o);
                        *r = reward as f32;
                        *d = done;
                    }
                });
        });
        self.tick_count += 1;
        Ok(())
    }

    /// How round `i` went: the info a finished round reports.
    fn report<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let env = self.envs[i].as_ref().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no course"))?;
        let r = &env.report;
        let d = PyDict::new(py);
        d.set_item("success", r.success)?;
        d.set_item("clean", r.clean)?;
        d.set_item("fell", r.deaths > 0)?;
        d.set_item("deaths", r.deaths)?;
        d.set_item("respawned", r.respawned)?;
        d.set_item("stalled", r.stalled)?;
        d.set_item("damage", r.damage)?;
        d.set_item("blocks_used", r.used)?;
        d.set_item("blocks_needed", env.course.par)?;
        d.set_item("landed", r.landed)?;
        d.set_item("shift_share", r.shift_share)?;
        d.set_item("ticks", r.t)?;
        d.set_item("pushes", r.pushes)?;
        if r.success {
            d.set_item("blocks_per_s", r.blocks_per_s)?;
        }
        let parts = PyDict::new(py);
        for (name, v) in PARTS.iter().zip(env.parts.iter()) {
            if *v != 0.0 {
                parts.set_item(*name, *v)?;
            }
        }
        d.set_item("parts", parts)?;
        Ok(d)
    }

    /// The state of round `i`'s player, for checks and for watching.
    fn player<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let env = self.envs[i].as_ref().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no course"))?;
        let p = &env.player;
        let d = PyDict::new(py);
        d.set_item("pos", p.pos)?;
        d.set_item("vel", p.dm)?;
        d.set_item("yaw", p.yaw)?;
        d.set_item("pitch", p.pitch)?;
        d.set_item("on_ground", p.on_ground)?;
        d.set_item("held_count", p.held)?;
        d.set_item("health", p.health)?;
        d.set_item("t", env.t)?;
        Ok(d)
    }
}

fn to_keys(k: [bool; 7]) -> Keys {
    Keys { forward: k[0], back: k[1], left: k[2], right: k[3], jump: k[4], shift: k[5], sprint: k[6] }
}

#[pyfunction]
fn kind_of(name: &str) -> u8 {
    kind_for(name)
}

fn to_push(p: (i32, i32, usize, f64, f64)) -> Push {
    Push { from: p.0, until: p.1, level: p.2, turn: p.3, jitter: p.4 }
}

/// clutch.decide for the real game: the push to give now, from what the mod reports, or None.
/// keys are (forward, back, left, right, jump, sneak, sprint): `before` as held last tick, `keys` for this one.
#[pyfunction]
#[pyo3(signature = (solid, feet, pos, vel, yaw, pitch, held, before, keys, click, turn_yaw, turn_pitch, goal, fall_line, push, sprinting=false))]
#[allow(clippy::too_many_arguments)]
fn push_for(solid: &str, feet: [i32; 3], pos: [f64; 3], vel: [f64; 3], yaw: f32, pitch: f32, held: i32, before: [bool; 7], keys: [bool; 7], click: bool, turn_yaw: f64,
            turn_pitch: f64, goal: [f64; 2], fall_line: f64, push: (i32, i32, usize, f64, f64), sprinting: bool) -> Option<[f64; 3]> {
    let cube = Cube::from_text(feet, solid);
    let state = clutch::State { pos, vel, yaw, pitch, held, keys: to_keys(before), sprinting };
    let first = Input { keys: to_keys(keys), click, turn_yaw, turn_pitch, ..Input::default() };
    clutch::decide(&cube, &state, &first, goal, fall_line, &to_push(push))
}

#[pymodule]
fn mcsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Single>()?;
    m.add_class::<Batch>()?;
    m.add("N_OBS", N_OBS)?;
    m.add_function(wrap_pyfunction!(kind_of, m)?)?;
    m.add_function(wrap_pyfunction!(push_for, m)?)?;
    m.add_function(wrap_pyfunction!(combat_features, m)?)?;
    m.add_function(wrap_pyfunction!(bot_action, m)?)?;
    m.add("MOVE_OBS", MOVE_OBS)?;
    m.add("EXTRA", EXTRA)?;
    m.add("ACTIONS", ACTIONS)?;
    Ok(())
}

/// A player as Python has it: (pos, vel, yaw, pitch, health, armor, damage, hurt_time, on_ground, sprinting, crouching).
type ViewTuple = ([f64; 3], [f64; 3], f32, f32, f32, i32, f32, i32, bool, bool, bool);

fn to_view(v: &ViewTuple) -> View {
    View { pos: v.0, vel: v.1, yaw: v.2, pitch: v.3, health: v.4, armor: v.5, damage: v.6, hurt_time: v.7, on_ground: v.8, sprinting: v.9, crouching: v.10 }
}

/// duel.features for the real game (and AutoEnv): the values after the movement model's, from what the game reports.
/// me: (slot, miss_lock, resist, damage, armor, server_sprinting, since_hit, since_hurt, duel). history: the last
/// HISTORY (attack, hotbar / 2) pairs, oldest first.
#[pyfunction]
#[pyo3(signature = (pos, yaw, pitch, crouching, me, enemies, k, solid, feet, history))]
#[allow(clippy::too_many_arguments)]
fn combat_features(pos: [f64; 3], yaw: f32, pitch: f32, crouching: bool, me: (u8, i32, i32, f32, i32, bool, i32, i32, bool), enemies: Vec<ViewTuple>, k: i32,
                   solid: &str, feet: [i32; 3], history: Vec<[f32; HIST_F]>) -> Vec<f32> {
    let mut p = Player::new(pos, yaw, pitch, 0);
    p.crouch_pose = crouching;
    let m = duel::Me { slot: me.0, miss_lock: me.1, resist: me.2, damage: me.3, armor: me.4, server_sprinting: me.5, since_hit: me.6, since_hurt: me.7, duel: me.8 };
    let views: Vec<View> = enemies.iter().map(to_view).collect();
    let mut h = [[0.0; HIST_F]; HISTORY];
    for (i, v) in history.iter().rev().take(HISTORY).enumerate() {
        h[HISTORY - 1 - i] = *v;
    }
    let mut out = vec![0.0; EXTRA];
    duel::features(&p, &m, &views, k, &Cube::from_text(feet, solid), &h, &mut out);
    out
}

/// duel.action for the real game: a scripted opponent's inputs this tick, in the model's action space.
#[pyfunction]
#[pyo3(signature = (bot, me, target, centre, t, since_hit))]
fn bot_action(bot: i32, me: ViewTuple, target: ViewTuple, centre: [f64; 3], t: i32, since_hit: i32) -> Vec<i64> {
    duel::action(Bot::from_index(bot), &to_view(&me), &to_view(&target), centre, t, since_hit).to_vec()
}
