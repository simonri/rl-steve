//! Blocks: a dense grid over the arena. Each cell holds a block kind, and each kind has its collision boxes.
//!
//! New kinds of block only need an entry in `kinds()` and a rule in `kind_for`. Physics works on the boxes alone.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub const fn new(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> Self {
        Aabb { min: [x0, y0, z0], max: [x1, y1, z1] }
    }

    #[inline]
    pub fn moved(&self, d: [f64; 3]) -> Aabb {
        Aabb { min: [self.min[0] + d[0], self.min[1] + d[1], self.min[2] + d[2]], max: [self.max[0] + d[0], self.max[1] + d[1], self.max[2] + d[2]] }
    }

    /// AABB.expandTowards: grows the box on the side the movement goes.
    #[inline]
    pub fn expand_towards(&self, d: [f64; 3]) -> Aabb {
        let mut b = *self;
        for i in 0..3 {
            if d[i] < 0.0 {
                b.min[i] += d[i];
            } else if d[i] > 0.0 {
                b.max[i] += d[i];
            }
        }
        b
    }

    #[inline]
    pub fn deflate(&self, e: f64) -> Aabb {
        Aabb::new(self.min[0] + e, self.min[1] + e, self.min[2] + e, self.max[0] - e, self.max[1] - e, self.max[2] - e)
    }

    /// Overlap with room inside, as Shapes.joinIsNotEmpty: touching faces do not count.
    #[inline]
    pub fn intersects(&self, o: &Aabb) -> bool {
        self.min[0] < o.max[0] && self.max[0] > o.min[0] && self.min[1] < o.max[1] && self.max[1] > o.min[1] && self.min[2] < o.max[2] && self.max[2] > o.min[2]
    }
}

/// What a block is to the player: the boxes it bumps into, whether the eyes count it as a block at all, and whether
/// placing a block into it replaces it (tall grass) or needs an empty spot.
pub struct Kind {
    pub boxes: Vec<Aabb>,
    pub air: bool,
    pub replaceable: bool,
    pub suffocating: bool, // a player inside it gets pushed out (stone yes, glass and leaves no)
}

pub const AIR: u8 = 0;
pub const FULL: u8 = 1;
pub const PASSABLE: u8 = 2; // flowers, torches: something there, nothing to stand on
pub const PLANT: u8 = 3; // tall grass, ferns: like PASSABLE, and a placed block replaces it
pub const SLAB_BOTTOM: u8 = 4;
pub const SLAB_TOP: u8 = 5;
pub const STAIRS: u8 = 6; // + facing (n, s, e, w), + 4 when upside down
pub const BED: u8 = 14;
pub const CARPET: u8 = 15;
pub const FENCE: u8 = 16; // + bit mask of arms: 1 north, 2 south, 4 east, 8 west
pub const PANE: u8 = 32; // the same for glass panes and bars
pub const WATER: u8 = 48; // no collision here: swimming is not simulated yet
pub const CLEAR: u8 = 49; // a full block that does not push a player out of it: glass, leaves

fn full() -> Aabb {
    Aabb::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
}

fn cross(arms: u8, half: f64, height: f64) -> Vec<Aabb> {
    let (a, b) = (0.5 - half, 0.5 + half);
    let mut v = vec![Aabb::new(a, 0.0, a, b, height, b)];
    if arms & 1 != 0 {
        v.push(Aabb::new(a, 0.0, 0.0, b, height, a));
    }
    if arms & 2 != 0 {
        v.push(Aabb::new(a, 0.0, b, b, height, 1.0));
    }
    if arms & 4 != 0 {
        v.push(Aabb::new(b, 0.0, a, 1.0, height, b));
    }
    if arms & 8 != 0 {
        v.push(Aabb::new(0.0, 0.0, a, a, height, b));
    }
    v
}

/// Every kind there is, by number.
pub fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let mut k: Vec<Kind> = (0..=255).map(|_| Kind { boxes: vec![full()], air: false, replaceable: false, suffocating: false }).collect();
        k[FULL as usize].suffocating = true;
        k[AIR as usize] = Kind { boxes: vec![], air: true, replaceable: true, suffocating: false };
        k[PASSABLE as usize] = Kind { boxes: vec![], air: false, replaceable: false, suffocating: false };
        k[PLANT as usize] = Kind { boxes: vec![], air: false, replaceable: true, suffocating: false };
        k[WATER as usize] = Kind { boxes: vec![], air: false, replaceable: true, suffocating: false };
        k[SLAB_BOTTOM as usize].boxes = vec![Aabb::new(0.0, 0.0, 0.0, 1.0, 0.5, 1.0)];
        k[SLAB_TOP as usize].boxes = vec![Aabb::new(0.0, 0.5, 0.0, 1.0, 1.0, 1.0)];
        for facing in 0..4u8 {
            // The tall part of a stair is on the side it faces.
            let quarter = |y0: f64, y1: f64| match facing {
                0 => Aabb::new(0.0, y0, 0.0, 1.0, y1, 0.5),
                1 => Aabb::new(0.0, y0, 0.5, 1.0, y1, 1.0),
                2 => Aabb::new(0.5, y0, 0.0, 1.0, y1, 1.0),
                _ => Aabb::new(0.0, y0, 0.0, 0.5, y1, 1.0),
            };
            k[(STAIRS + facing) as usize].boxes = vec![Aabb::new(0.0, 0.0, 0.0, 1.0, 0.5, 1.0), quarter(0.5, 1.0)];
            k[(STAIRS + 4 + facing) as usize].boxes = vec![Aabb::new(0.0, 0.5, 0.0, 1.0, 1.0, 1.0), quarter(0.0, 0.5)];
        }
        k[BED as usize].boxes = vec![Aabb::new(0.0, 0.0, 0.0, 1.0, 0.5625, 1.0)];
        k[CARPET as usize].boxes = vec![Aabb::new(0.0, 0.0, 0.0, 1.0, 0.0625, 1.0)];
        for arms in 0..16u8 {
            k[(FENCE + arms) as usize].boxes = cross(arms, 0.125, 1.5);
            k[(PANE + arms) as usize].boxes = cross(arms, 0.0625, 1.0);
        }
        k
    })
}

/// The kind for a block state as the game names it, such as "minecraft:oak_slab[type=bottom,waterlogged=false]".
pub fn kind_for(state: &str) -> u8 {
    let (name, props) = match state.find('[') {
        Some(i) => (&state[..i], &state[i + 1..state.len() - 1]),
        None => (state, ""),
    };
    let name = name.trim_start_matches("minecraft:");
    let prop = |key: &str| props.split(',').find_map(|p| p.strip_prefix(key).and_then(|v| v.strip_prefix('=')));
    if name == "air" || name == "cave_air" || name == "void_air" {
        return AIR;
    }
    if name.ends_with("_slab") {
        return match prop("type") {
            Some("top") => SLAB_TOP,
            Some("double") => FULL,
            _ => SLAB_BOTTOM,
        };
    }
    if name.ends_with("_stairs") {
        let facing = match prop("facing") {
            Some("north") => 0,
            Some("south") => 1,
            Some("east") => 2,
            _ => 3,
        };
        return STAIRS + facing + if prop("half") == Some("top") { 4 } else { 0 };
    }
    let arms = || ["north", "south", "east", "west"].iter().enumerate().fold(0u8, |m, (i, d)| if prop(d) == Some("true") { m | (1 << i) } else { m });
    if name.ends_with("_fence") || name.ends_with("_wall") {
        return FENCE + arms();
    }
    if name.ends_with("_pane") || name == "iron_bars" {
        return PANE + arms();
    }
    if name.ends_with("_bed") {
        return BED;
    }
    if name.ends_with("_carpet") {
        return CARPET;
    }
    if name == "water" || name == "lava" || name == "bubble_column" {
        return WATER;
    }
    if name.ends_with("glass") || name.ends_with("_leaves") || name == "barrier" || name == "glowstone" || name.ends_with("ice") {
        return CLEAR;
    }
    if matches!(name, "short_grass" | "grass" | "tall_grass" | "fern" | "large_fern" | "vine" | "dead_bush" | "snow") {
        return PLANT;
    }
    if name.ends_with("_flower")
        || name.ends_with("torch")
        || name.ends_with("_sapling")
        || name.ends_with("_tulip")
        || name.ends_with("_button")
        || name.ends_with("_sign")
        || name.ends_with("_banner")
        || name.ends_with("_pressure_plate")
        || matches!(name, "poppy" | "dandelion" | "allium" | "azure_bluet" | "oxeye_daisy" | "cornflower" | "lily_of_the_valley" | "rose_bush"
            | "peony" | "lilac" | "sunflower" | "lever" | "redstone_wire" | "light" | "structure_void" | "rail" | "tripwire")
    {
        return PASSABLE;
    }
    FULL
}

/// A box of cells. Outside it everything is air. Placing a block outside makes it bigger.
#[derive(Clone)]
pub struct World {
    pub origin: [i32; 3],
    pub size: [i32; 3],
    cells: Vec<u8>,
}

impl World {
    pub fn new(min: [i32; 3], max: [i32; 3]) -> World {
        let size = [(max[0] - min[0] + 1).max(1), (max[1] - min[1] + 1).max(1), (max[2] - min[2] + 1).max(1)];
        World { origin: min, size, cells: vec![AIR; (size[0] * size[1] * size[2]) as usize] }
    }

    #[inline]
    fn index(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        let (i, j, k) = (x - self.origin[0], y - self.origin[1], z - self.origin[2]);
        if i < 0 || j < 0 || k < 0 || i >= self.size[0] || j >= self.size[1] || k >= self.size[2] {
            return None;
        }
        Some(((i * self.size[1] + j) * self.size[2] + k) as usize)
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> u8 {
        self.index(x, y, z).map_or(AIR, |i| self.cells[i])
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, kind: u8) {
        if self.index(x, y, z).is_none() {
            if kind == AIR {
                return;
            }
            self.grow(x, y, z);
        }
        let i = self.index(x, y, z).unwrap();
        self.cells[i] = kind;
    }

    fn grow(&mut self, x: i32, y: i32, z: i32) {
        const MARGIN: i32 = 16;
        let p = [x, y, z];
        let mut min = self.origin;
        let mut max = [self.origin[0] + self.size[0] - 1, self.origin[1] + self.size[1] - 1, self.origin[2] + self.size[2] - 1];
        for a in 0..3 {
            if p[a] < min[a] {
                min[a] = p[a] - MARGIN;
            }
            if p[a] > max[a] {
                max[a] = p[a] + MARGIN;
            }
        }
        let mut bigger = World::new(min, max);
        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                for k in 0..self.size[2] {
                    let v = self.cells[((i * self.size[1] + j) * self.size[2] + k) as usize];
                    if v != AIR {
                        let at = bigger.index(self.origin[0] + i, self.origin[1] + j, self.origin[2] + k).unwrap();
                        bigger.cells[at] = v;
                    }
                }
            }
        }
        *self = bigger;
    }

    pub fn fill(&mut self, min: [i32; 3], max: [i32; 3], kind: u8) {
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    self.set(x, y, z, kind);
                }
            }
        }
    }

    /// Something there, for the eyes: anything that is not air.
    #[inline]
    pub fn present(&self, x: i32, y: i32, z: i32) -> bool {
        !kinds()[self.get(x, y, z) as usize].air
    }

    /// Every collision box of the blocks near `area`, in world coordinates, as BlockCollisions collects them: one
    /// block further out on every side, which catches fences reaching up from the row below.
    pub fn boxes_near(&self, area: &Aabb, out: &mut Vec<Aabb>) {
        out.clear();
        let lo = [(area.min[0] - 1.0e-7).floor() as i32 - 1, (area.min[1] - 1.0e-7).floor() as i32 - 1, (area.min[2] - 1.0e-7).floor() as i32 - 1];
        let hi = [(area.max[0] + 1.0e-7).floor() as i32 + 1, (area.max[1] + 1.0e-7).floor() as i32 + 1, (area.max[2] + 1.0e-7).floor() as i32 + 1];
        let all = kinds();
        for x in lo[0]..=hi[0] {
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    let id = self.get(x, y, z);
                    if id == AIR {
                        continue;
                    }
                    for b in &all[id as usize].boxes {
                        out.push(b.moved([x as f64, y as f64, z as f64]));
                    }
                }
            }
        }
    }

    /// Level.collidesWithSuffocatingBlock: a block the player gets pushed out of has a box with room in the area.
    pub fn suffocates(&self, area: &Aabb) -> bool {
        let all = kinds();
        for x in area.min[0].floor() as i32..=area.max[0].floor() as i32 {
            for z in area.min[2].floor() as i32..=area.max[2].floor() as i32 {
                for y in area.min[1].floor() as i32..=area.max[1].floor() as i32 {
                    let k = &all[self.get(x, y, z) as usize];
                    if k.suffocating && k.boxes.iter().any(|b| b.moved([x as f64, y as f64, z as f64]).intersects(area)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Level.noCollision for blocks: no box has room inside the area.
    pub fn no_collision(&self, area: &Aabb) -> bool {
        let all = kinds();
        let lo = [area.min[0].floor() as i32 - 1, area.min[1].floor() as i32 - 1, area.min[2].floor() as i32 - 1];
        let hi = [area.max[0].floor() as i32 + 1, area.max[1].floor() as i32 + 1, area.max[2].floor() as i32 + 1];
        for x in lo[0]..=hi[0] {
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    let id = self.get(x, y, z);
                    if id == AIR {
                        continue;
                    }
                    for b in &all[id as usize].boxes {
                        if b.moved([x as f64, y as f64, z as f64]).intersects(area) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}
