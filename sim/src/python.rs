//! The Python side: one player in one world, for checking against the real game tick by tick.

use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::conversion::FromPyObjectOwned;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::env::{Course, Env, Field, N_OBS, PARTS};
use crate::player::{Input, Keys, Player, FACE_NAMES};
use crate::world::{kind_for, World};

#[pyclass(unsendable)]
pub struct Single {
    world: World,
    player: Player,
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
        Single { world, player: Player::new([0.0; 3], 0.0, 0.0, 0) }
    }

    /// A new player standing at a spot, as after the mod's reset: nothing kept from before.
    fn spawn(&mut self, pos: [f64; 3], yaw: f32, pitch: f32, held: i32) {
        let mut p = Player::new(pos, yaw, pitch, held);
        p.on_ground = true;
        p.dm = [0.0, -0.0784, 0.0];
        self.player = p;
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
    fn tick<'py>(&mut self, py: Python<'py>, keys: [bool; 7], click: bool, yaw: f64, pitch: f64) -> PyResult<Bound<'py, PyDict>> {
        let ev = self.player.tick(&mut self.world, &Input { keys: to_keys(keys), click, turn_yaw: yaw, turn_pitch: pitch });
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
        let acts: Vec<[i64; 8]> = acts.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]).collect();
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

#[pymodule]
fn mcsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Single>()?;
    m.add_class::<Batch>()?;
    m.add("N_OBS", N_OBS)?;
    m.add_function(wrap_pyfunction!(kind_of, m)?)?;
    Ok(())
}
