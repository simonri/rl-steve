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
    queued: Option<(crate::combat::Swing, usize)>,
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
        Single { world, player: Player::new([0.0; 3], 0.0, 0.0, 0), opponents: Vec::new(), centre: [0.0; 3], t: 0, since_hit: 99, since_hurt: 99, fall_line: f64::MIN, queued: None }
    }

    /// A player standing at a spot, as after the mod's reset or respawn: its gear, hotbar and slot kept (set_hotbar gives
    /// a new round's). `held` only matters for a player that had nothing before.
    fn spawn(&mut self, pos: [f64; 3], yaw: f32, pitch: f32, held: i32) {
        let mut p = Player::new(pos, yaw, pitch, held);
        p.on_ground = true;
        p.dm = [0.0, -0.0784, 0.0];
        p.keep_gear(&self.player); // as Env.respawn: the inventory and the rules stay
        self.player = p;
        self.queued = None; // a hit on its way belongs to the player that was
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
        let state = clutch::State { pos: p.pos, vel: p.dm, yaw: p.yaw, pitch: p.pitch, held: p.held(), keys: to_keys(before), sprinting: p.sprinting };
        clutch::can_save(&self.world, feet, &state, vel, &Input { keys: to_keys(keys), click, turn_yaw, turn_pitch, ..Input::default() }, fall_line)
    }

    /// The player's gear and hand: attack attribute (1 plus the sword), armor points, Knockback level, the selected
    /// hotbar slot (0-8), and whether healing follows 1.8.
    #[pyo3(signature = (damage, armor, kb, slot, rules18))]
    fn set_gear(&mut self, damage: f32, armor: i32, kb: i32, slot: u8, rules18: bool) {
        let p = &mut self.player;
        (p.attack_damage, p.armor, p.kb_enchant, p.slot, p.rules18) = (damage, armor, kb, slot, rules18);
    }

    /// The hotbar: (item, count) per slot (items.rs), and the selected slot, as the round's reset gives them.
    fn set_hotbar(&mut self, slots: Vec<(u8, i32)>, slot: u8) -> PyResult<()> {
        self.player.hotbar = to_hotbar(slots);
        self.player.slot = to_slot(slot)?;
        Ok(())
    }

    /// An opponent played by a scripted bot (duel.Bot by index), as the round's reset puts it.
    #[pyo3(signature = (bot, pos, yaw, damage, armor, kb, rules18, centre, fall_line))]
    #[allow(clippy::too_many_arguments)]
    fn add_opponent(&mut self, bot: i32, pos: [f64; 3], yaw: f32, damage: f32, armor: i32, kb: i32, rules18: bool, centre: [f64; 3], fall_line: f64) {
        self.fall_line = fall_line;
        let (hotbar, slot) = crate::env::rival_hotbar(&self.player.hotbar, damage); // the model's layout, with its own sword
        self.opponents.push(Opponent::new(&Rival { bot: Bot::from_index(bot), pos, yaw, damage, armor, kb }, rules18, hotbar, slot));
        self.centre = centre;
        self.t = 0;
        self.since_hit = 99;
        self.since_hurt = 99;
        self.queued = None;
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
        p.set_held(held);
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
            let ev = fight_tick(&mut self.world, &mut self.player, &mut self.opponents, &input, self.centre, self.t, &mut self.since_hit, &mut self.since_hurt, &mut self.queued);
            self.t += 1;
            for o in self.opponents.iter_mut() {
                if !o.out && (o.player.health <= 0.0 || o.player.pos[1] < self.fall_line) {
                    o.out = true;
                }
            }
            ev
        };
        let d = self.state(py)?;
        if let Some((across, critical, sword)) = ev.landed {
            d.set_item("landed", (across, critical, sword))?; // as the mod reports the model's counted hit
        }
        if ev.struck {
            d.set_item("struck", true)?; // its click was on an opponent, as the mod reports it
        }
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
        d.set_item("held_count", p.held())?;
        d.set_item("hotbar", p.hotbar.iter().map(|h| (h.item, h.count)).collect::<Vec<_>>())?;
        d.set_item("slot", p.slot)?;
        d.set_item("health", p.shown_health())?; // what its client shows (a hit's damage from the next tick)
        d.set_item("hurt_time", p.hurt_time)?;
        d.set_item("crouch_pose", p.crouch_pose)?;
        d.set_item("slot", p.slot)?;
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
            if let Some(s) = o.swing {
                e.set_item("swing", s)?; // as the mod reports an opponent's click and its counted hit
            }
            if let Some(l) = o.landed {
                e.set_item("landed", l)?;
            }
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
    bonus_weight: f64, // Env.bonus_weight for every round, set from training (train_auto.Progress)
}

impl Batch {
    /// Round `i`, or a Python error when there is no such round or it has no course yet.
    fn env(&self, i: usize) -> PyResult<&Env> {
        self.envs
            .get(i)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(format!("round {i} of {}", self.envs.len())))?
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no course"))
    }
}

fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?.extract::<T>().map_err(Into::into)
}

#[pymethods]
impl Batch {
    #[new]
    fn new(n: usize) -> Self {
        Batch { envs: (0..n).map(|_| None).collect(), tick_count: 0, bonus_weight: 1.0 }
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
            hotbar: match spec.get_item("hotbar")? {
                Some(h) if !h.is_none() => to_hotbar(h.extract::<Vec<(u8, i32)>>()?),
                _ => crate::player::blocks_hotbar(get(spec, "blocks")?), // the blocks in the first slot, selected
            },
            slot: to_slot(spec.get_item("slot")?.map(|v| v.extract::<u8>()).transpose()?.unwrap_or(0))?,
            needs_blocks: spec.get_item("needs_blocks")?.map(|v| v.extract::<bool>()).transpose()?.unwrap_or(false),
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
        let n = self.envs.len();
        let mut env = Env::new(course, self.tick_count);
        env.bonus_weight = self.bonus_weight;
        *self.envs.get_mut(i).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(format!("round {i} of {n}")))? = Some(env);
        Ok(())
    }

    /// How much the fight's event bonuses count, for every round from now on, the running ones too (shaping.rs).
    fn set_bonus_weight(&mut self, w: f64) {
        self.bonus_weight = w;
        for e in self.envs.iter_mut().flatten() {
            e.bonus_weight = w;
        }
    }

    /// Writes what round `i` sees now into `out`.
    fn observe(&self, i: usize, mut out: PyReadwriteArray1<f32>) -> PyResult<()> {
        let env = self.env(i)?;
        env.observe(out.as_slice_mut()?);
        Ok(())
    }

    /// Self-play: the next tick's action for the first opponent of each round, played by a model instead of its bot.
    /// A row starting with -1 leaves that round's opponent to its bot. Taken by the next `step`.
    fn set_opponent_actions(&mut self, actions: PyReadonlyArray2<i64>) -> PyResult<()> {
        let acts = actions.as_array();
        let n = self.envs.len();
        let bad = |what: String| pyo3::exceptions::PyValueError::new_err(what);
        if acts.nrows() != n || acts.ncols() != ACTIONS {
            return Err(bad(format!("opponent actions are {:?}, not ({n}, {ACTIONS})", acts.shape())));
        }
        for (env, row) in self.envs.iter_mut().zip(acts.rows()) {
            let Some(env) = env else { continue };
            if row[0] < 0 {
                continue;
            }
            if let Some((j, v)) = row.iter().enumerate().find(|&(j, &v)| v < 0 || v >= crate::duel::CHOICES[j]) {
                return Err(bad(format!("opponent action {v} at {j} is outside 0..{}", crate::duel::CHOICES[j])));
            }
            if let Some(o) = env.opponents.first_mut() {
                o.next = Some(std::array::from_fn(|i| row[i]));
            }
        }
        Ok(())
    }

    /// Self-play: what the first opponent of every round sees (Env.observe_opponent), into `out` (rounds x N_OBS).
    /// Returns which rounds have one; the others' rows are left as they are.
    fn observe_opponents(&self, py: Python<'_>, mut out: PyReadwriteArray2<f32>) -> PyResult<Vec<bool>> {
        let out = out.as_slice_mut()?;
        if out.len() != self.envs.len() * N_OBS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("a buffer for {} rounds, not {}", out.len() / N_OBS, self.envs.len())));
        }
        let envs = &self.envs;
        Ok(py.detach(|| {
            envs.par_iter()
                .zip(out.par_chunks_mut(N_OBS))
                .map(|(env, o)| match env {
                    Some(env) if !env.opponents.is_empty() => {
                        env.observe_opponent(0, o);
                        true
                    }
                    _ => false,
                })
                .collect()
        }))
    }

    /// One decision in every round at once. Writes what each sees after it, its reward and whether it ended.
    fn step(&mut self, py: Python<'_>, actions: PyReadonlyArray2<i64>, mut obs: PyReadwriteArray2<f32>, mut rewards: PyReadwriteArray1<f32>, mut dones: PyReadwriteArray1<bool>) -> PyResult<()> {
        let acts = actions.as_array();
        let n = self.envs.len();
        let bad = |what: String| pyo3::exceptions::PyValueError::new_err(what);
        if acts.nrows() != n || acts.ncols() != ACTIONS {
            return Err(bad(format!("actions are {:?}, not ({n}, {ACTIONS})", acts.shape())));
        }
        if let Some((i, v)) = acts.indexed_iter().find(|((_, j), &v)| v < 0 || v >= crate::duel::CHOICES[*j]) {
            return Err(bad(format!("action {v} at {i:?} is outside 0..{}", crate::duel::CHOICES[i.1])));
        }
        let acts: Vec<[i64; ACTIONS]> = acts.rows().into_iter().map(|r| std::array::from_fn(|i| r[i])).collect();
        let obs = obs.as_slice_mut()?;
        let rewards = rewards.as_slice_mut()?;
        let dones = dones.as_slice_mut()?;
        if obs.len() != n * N_OBS || rewards.len() != n || dones.len() != n {
            return Err(bad(format!("buffers for {} rounds, {} rewards, {} dones, not {n}", obs.len() / N_OBS, rewards.len(), dones.len())));
        }
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
        let env = self.env(i)?;
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
        let env = self.env(i)?;
        let p = &env.player;
        let d = PyDict::new(py);
        d.set_item("pos", p.pos)?;
        d.set_item("vel", p.dm)?;
        d.set_item("yaw", p.yaw)?;
        d.set_item("pitch", p.pitch)?;
        d.set_item("on_ground", p.on_ground)?;
        d.set_item("held_count", p.held())?;
        d.set_item("hotbar", p.hotbar.iter().map(|h| (h.item, h.count)).collect::<Vec<_>>())?;
        d.set_item("slot", p.slot)?;
        d.set_item("health", p.shown_health())?;
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
    m.add_function(wrap_pyfunction!(fight_potential, m)?)?;
    m.add_function(wrap_pyfunction!(fight_events, m)?)?;
    m.add("GAMMA", crate::shaping::GAMMA)?;
    m.add("NON_SWORD", crate::shaping::NON_SWORD)?;
    m.add_function(wrap_pyfunction!(bot_action, m)?)?;
    m.add("MOVE_OBS", MOVE_OBS)?;
    m.add("EXTRA", EXTRA)?;
    m.add("ENEMY_F", crate::duel::ENEMY_F)?;
    m.add("ACTIONS", ACTIONS)?;
    Ok(())
}

/// A player as Python has it: (pos, vel, yaw, pitch, health, armor, damage, hurt_time, on_ground, sprinting, crouching).
type ViewTuple = ([f64; 3], [f64; 3], f32, f32, f32, i32, f32, i32, bool, bool, bool);

fn to_view(v: &ViewTuple) -> View {
    View { pos: v.0, vel: v.1, yaw: v.2, pitch: v.3, health: v.4, armor: v.5, damage: v.6, hurt_time: v.7, on_ground: v.8, sprinting: v.9, crouching: v.10 }
}

/// duel.features for the real game (and AutoEnv): the values after the movement model's, from what the game reports.
/// me: (selected slot 0-8, resist, damage, armor, server_sprinting, since_hit, since_hurt, duel). history: the last
/// HISTORY left clicks, oldest first, each as a one-value list. hotbar: (item, count) per slot, as the game shows it.
#[pyfunction]
#[pyo3(signature = (pos, yaw, pitch, crouching, me, enemies, k, solid, feet, history, hotbar))]
#[allow(clippy::too_many_arguments)]
fn combat_features(pos: [f64; 3], yaw: f32, pitch: f32, crouching: bool, me: (u8, i32, f32, i32, bool, i32, i32, bool), enemies: Vec<ViewTuple>, k: i32,
                   solid: &str, feet: [i32; 3], history: Vec<[f32; HIST_F]>, hotbar: Vec<(u8, i32)>) -> Vec<f32> {
    let mut p = Player::new(pos, yaw, pitch, 0);
    p.crouch_pose = crouching;
    p.hotbar = to_hotbar(hotbar);
    p.slot = me.0;
    let m = duel::Me { resist: me.1, damage: me.2, armor: me.3, server_sprinting: me.4, since_hit: me.5, since_hurt: me.6, duel: me.7 };
    let views: Vec<View> = enemies.iter().map(to_view).collect();
    let mut h = [[0.0; HIST_F]; HISTORY];
    for (i, v) in history.iter().rev().take(HISTORY).enumerate() {
        h[HISTORY - 1 - i] = *v;
    }
    let mut out = vec![0.0; EXTRA];
    duel::features(&p, &m, &views, k, &|| Cube::from_text(feet, &solid), &h, &mut out);
    out
}

/// shaping::potential for AutoEnv, from what the game reports: the model's aim at the opponents still in the fight
/// (`alive`), minus theirs at it.
#[pyfunction]
#[pyo3(signature = (pos, yaw, pitch, crouching, enemies, alive))]
fn fight_potential(pos: [f64; 3], yaw: f32, pitch: f32, crouching: bool, enemies: Vec<ViewTuple>, alive: Vec<bool>) -> f64 {
    let mut me = Player::new(pos, yaw, pitch, 0);
    me.crouch_pose = crouching;
    let players: Vec<Player> = enemies
        .iter()
        .map(|e| {
            let mut p = Player::new(e.0, e.2, e.3, 0);
            p.crouch_pose = e.10;
            p
        })
        .collect();
    let foes: Vec<(&Player, bool)> = players.iter().zip(alive).collect();
    crate::shaping::potential(&me, &foes)
}

/// shaping::events and zero_sum for AutoEnv: [long hit, critical hit, missed swing] for the model this tick. `foes` is
/// each opponent's (landed, swing), as the game reports them.
#[pyfunction]
#[pyo3(signature = (landed, swing, foes, weight))]
fn fight_events(landed: Option<(f64, bool, bool)>, swing: Option<bool>, foes: Vec<(Option<(f64, bool, bool)>, Option<bool>)>, weight: f64) -> [f64; 3] {
    let theirs: Vec<[f64; 3]> = foes.iter().map(|(l, s)| crate::shaping::events(*l, *s)).collect();
    crate::shaping::zero_sum(crate::shaping::events(landed, swing), &theirs, foes.len(), weight)
}

/// duel.action for the real game: a scripted opponent's inputs this tick, in the model's action space.
#[pyfunction]
#[pyo3(signature = (bot, me, target, centre, t, since_hit, floor))]
fn bot_action(bot: i32, me: ViewTuple, target: ViewTuple, centre: [f64; 3], t: i32, since_hit: i32, floor: Vec<[i32; 6]>) -> Vec<i64> {
    // The arena's blocks as boxes (x0, y0, z0, x1, y1, z1): the game's world for the bot's step ahead.
    let ground = |x: i32, y: i32, z: i32| floor.iter().any(|b| b[0] <= x && x <= b[3] && b[1] <= y && y <= b[4] && b[2] <= z && z <= b[5]);
    duel::action(Bot::from_index(bot), &to_view(&me), &to_view(&target), centre, t, since_hit, &ground).to_vec()
}

/// A hotbar from Python: (item, count) per slot, items.rs's numbering.
fn to_hotbar(slots: Vec<(u8, i32)>) -> [crate::items::Stack; crate::items::SLOTS] {
    let mut h = [crate::items::Stack::default(); crate::items::SLOTS];
    for (i, (item, count)) in slots.into_iter().take(crate::items::SLOTS).enumerate() {
        // As the game has it: nothing in a slot with no items, and anything off the list is "other".
        h[i] = if count <= 0 || item == crate::items::EMPTY {
            crate::items::Stack::default()
        } else {
            crate::items::Stack { item: if (item as usize) < crate::items::KINDS { item } else { crate::items::OTHER }, count }
        };
    }
    h
}

/// A hotbar slot from Python: 0-8, or a ValueError (a slot past the hotbar would be shown as the last one).
fn to_slot(slot: u8) -> PyResult<u8> {
    if (slot as usize) < crate::items::SLOTS {
        Ok(slot)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!("hotbar slot {slot}: 0-8")))
    }
}
