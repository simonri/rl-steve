//! One player, one tick at a time, as Minecraft 26.3 runs it for the local player in our lockstep mod.
//!
//! Order within a tick (mod: action at START_CLIENT_TICK, server tick and its packets at END_CLIENT_TICK):
//!   1. keys are set, a click is queued, the view turns (Entity.turn)
//!   2. Minecraft.tick: pick (what the crosshair is on), then the click places a block (BlockItem.place)
//!   3. LocalPlayer.tick -> LivingEntity.baseTick (hurt time) -> LocalPlayer.aiStep -> LivingEntity.aiStep -> travel
//!   4. Player.updatePlayerPose
//!   5. server tick: fall damage from the landing, its packets come back before the reply (health, motion, hurt)
//! Names in comments are the game's methods this follows.

use crate::mth;
use crate::world::{kinds, Aabb, World, AIR, FULL};

pub const WIDTH: f64 = 0.6;
pub const STAND_HEIGHT: f64 = 1.8;
pub const CROUCH_HEIGHT: f64 = 1.5;
pub const STAND_EYES: f64 = 1.62;
pub const CROUCH_EYES: f64 = 1.27;
pub const STEP_HEIGHT: f32 = 0.6;
pub const GRAVITY: f64 = 0.08;
pub const JUMP_POWER: f32 = 0.42;
pub const REACH: f64 = 4.5;
pub const SNEAK_SPEED: f32 = 0.3;
pub const SAFE_FALL: f64 = 3.0;
pub const SPRINT_WINDOW: i32 = 7; // options.sprintWindow: ticks to tap forward again for a sprint
pub const MAX_HEALTH: f32 = 20.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Keys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub shift: bool,
    pub sprint: bool,
}

/// What the crosshair is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub block: [i32; 3],
    pub face: u8, // 0 down, 1 up, 2 north, 3 south, 4 west, 5 east, as Direction
    pub loc: [f64; 3],
}

pub const FACE_NAMES: [&str; 6] = ["down", "up", "north", "south", "west", "east"];
pub const FACE_STEP: [[i32; 3]; 6] = [[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0]];

/// One tick's input, as the mod applies it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    pub keys: Keys,
    pub click: bool,
    pub attack: bool,     // left click: hit the player in the crosshair (combat.rs works out whom)
    pub slot: Option<u8>, // a hotbar key: 0 blocks, 1 sword
    pub turn_yaw: f64,   // degrees, sent to Entity.turn as yaw / 0.15
    pub turn_pitch: f64, // degrees
}

/// What happened this tick besides the new state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Events {
    pub placed: Option<([i32; 3], [i32; 3])>, // the new block and the block it was placed against
    pub clicked_block: bool,                   // the click was on a block, placed or not (UseBlockCallback)
    pub damage: f32,
    pub landed: Option<(f64, bool)>, // the model's own hit that counted this tick: feet distance across, critical (fight_tick)
}

#[derive(Clone, Debug)]
pub struct Player {
    pub pos: [f64; 3],
    pub dm: [f64; 3], // deltaMovement
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub horizontal_collision: bool,
    pub minor_horizontal_collision: bool,
    pub fall_distance: f64,
    pub server_fall: f64, // the server's own count, from the moves the client reports
    pub sprinting: bool,
    pub crouching: bool,    // LocalPlayer.crouching: moving slowly
    pub crouch_pose: bool,  // Pose.CROUCHING: the smaller box and lower eyes
    pub keys: Keys,         // input.keyPresses
    pub move_vec: [f32; 2], // input.moveVector: (left, forward)
    pub xxa: f32,
    pub zza: f32,
    pub no_jump_delay: i32,
    pub sprint_trigger: i32,
    pub health: f32,
    pub hurt_time: i32,
    pub age: i64, // the server's tickCount for the player: Peaceful heals on every 20th
    pub rules18: bool, // 1.8 healing (one point every 80 ticks) instead of Peaceful's
    pub heal_timer: i32,
    // Combat (combat.rs).
    pub slot: u8,             // selected hotbar slot: 0 blocks, 1 sword
    pub resist: i32,          // hurtResistantTime
    pub last_damage: f32,     // lastDamage, for hits inside the resistance
    pub attack_damage: f32,   // the attack attribute: 1, plus the sword
    pub armor: i32,           // armor points
    pub kb_enchant: i32,      // Knockback level of the held item
    pub sent_sprinting: bool, // the sprint state the client last told the server
    pub server_sprinting: bool, // the server's own idea of it: a sprinting hit clears it until the client starts again
    pub unseen: f32, // damage from a hit this tick that its client shows only next tick (fight_tick)
    pub fighting: bool, // in a fight its hurt resistance and healing count after the bots' hits (fight_tick calls count_down)
    pub direct_turn: bool, // a bot driven on the server: turned by the step itself, not through the mouse (Fight18.drive)
    pub pending_knock: Option<[f64; 3]>, // a hit's motion, applied before the next tick as the packet is
    pub held: i32,
    pub server_motion: [f64; 3], // the server's deltaMovement for the player, sent back when it is hurt
    scratch: Vec<Aabb>,
}

impl Player {
    pub fn new(pos: [f64; 3], yaw: f32, pitch: f32, held: i32) -> Player {
        Player {
            pos,
            dm: [0.0; 3],
            yaw,
            pitch,
            on_ground: false,
            horizontal_collision: false,
            minor_horizontal_collision: false,
            fall_distance: 0.0,
            server_fall: 0.0,
            sprinting: false,
            crouching: false,
            crouch_pose: false,
            keys: Keys::default(),
            move_vec: [0.0; 2],
            xxa: 0.0,
            zza: 0.0,
            no_jump_delay: 0,
            sprint_trigger: 0,
            health: MAX_HEALTH,
            hurt_time: 0,
            age: 0,
            rules18: false,
            heal_timer: 0,
            slot: 0,
            resist: 0,
            last_damage: 0.0,
            attack_damage: 1.0,
            armor: 0,
            kb_enchant: 0,
            sent_sprinting: false,
            server_sprinting: false,
            unseen: 0.0,
            fighting: false,
            direct_turn: false,
            pending_knock: None,
            held,
            server_motion: [0.0; 3],
            scratch: Vec::with_capacity(64),
        }
    }

    /// Carries over what a respawn keeps: the inventory's gear and slot, and the rules.
    pub fn keep_gear(&mut self, old: &Player) {
        self.rules18 = old.rules18;
        self.attack_damage = old.attack_damage;
        self.armor = old.armor;
        self.kb_enchant = old.kb_enchant;
        self.slot = old.slot;
    }

    pub fn height(&self) -> f64 {
        if self.crouch_pose {
            CROUCH_HEIGHT
        } else {
            STAND_HEIGHT
        }
    }

    pub fn eye_height(&self) -> f64 {
        if self.crouch_pose {
            CROUCH_EYES
        } else {
            STAND_EYES
        }
    }

    pub fn bbox(&self) -> Aabb {
        box_at(self.pos, self.height())
    }

    fn can_fit(&self, world: &World, crouch: bool) -> bool {
        // canPlayerFitWithinBlocksAndEntitiesWhen
        world.no_collision(&box_at(self.pos, if crouch { CROUCH_HEIGHT } else { STAND_HEIGHT }).deflate(1.0e-7))
    }

    /// Movement speed attribute: 0.1, and 30 % more while sprinting (ADD_MULTIPLIED_TOTAL), read back as a float.
    fn speed(&self) -> f32 {
        let base = 0.1f32 as f64;
        if self.sprinting {
            (base * (1.0 + 0.3f32 as f64)) as f32
        } else {
            base as f32
        }
    }

    /// Entity.turn, from the mod's player.turn(dYaw / 0.15, dPitch / 0.15).
    pub fn turn(&mut self, yaw_degrees: f64, pitch_degrees: f64) {
        if yaw_degrees == 0.0 && pitch_degrees == 0.0 {
            return;
        }
        if self.direct_turn {
            self.pitch = (self.pitch + pitch_degrees as f32).clamp(-90.0, 90.0);
            self.yaw += yaw_degrees as f32;
            return;
        }
        let x_delta = (pitch_degrees / 0.15) as f32 * 0.15f32;
        let y_delta = (yaw_degrees / 0.15) as f32 * 0.15f32;
        self.pitch = (self.pitch + x_delta).clamp(-90.0, 90.0);
        self.yaw += y_delta;
    }

    /// Where its eyes are.
    pub fn eye(&self) -> [f64; 3] {
        [self.pos[0], self.pos[1] + self.eye_height(), self.pos[2]]
    }

    /// Entity.calculateViewVector.
    pub fn view(&self) -> [f64; 3] {
        let x_rot = self.pitch * mth::DEG_TO_RAD_F;
        let y_rot = -self.yaw * mth::DEG_TO_RAD_F;
        let y_cos = mth::cos(y_rot as f64);
        let y_sin = mth::sin(y_rot as f64);
        let x_cos = mth::cos(x_rot as f64);
        let x_sin = mth::sin(x_rot as f64);
        [(y_sin * x_cos) as f64, (-x_sin) as f64, (y_cos * x_cos) as f64]
    }

    /// LocalPlayer.raycastHitResult for blocks: from the eyes along the view, as far as blocks can be reached.
    pub fn pick(&self, world: &World) -> Option<Hit> {
        let eye = self.eye();
        let v = self.view();
        let to = [eye[0] + v[0] * REACH, eye[1] + v[1] * REACH, eye[2] + v[2] * REACH];
        let hit = clip(world, eye, to)?;
        let d = [hit.loc[0] - eye[0], hit.loc[1] - eye[1], hit.loc[2] - eye[2]];
        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < REACH * REACH {
            Some(hit)
        } else {
            None
        }
    }

    /// BlockItem.place through a right click: the spot next to the side aimed at, or the aimed block itself when it
    /// can be replaced, if that spot is free and the player is not in the way.
    fn place(&mut self, world: &mut World, hit: &Hit) -> Option<([i32; 3], [i32; 3])> {
        if self.held <= 0 || self.slot != 0 {
            return None;
        }
        let all = kinds();
        let clicked = world.get(hit.block[0], hit.block[1], hit.block[2]);
        let target = if all[clicked as usize].replaceable && clicked != AIR {
            hit.block
        } else {
            let s = FACE_STEP[hit.face as usize];
            [hit.block[0] + s[0], hit.block[1] + s[1], hit.block[2] + s[2]]
        };
        let there = world.get(target[0], target[1], target[2]);
        if !all[there as usize].replaceable {
            return None;
        }
        let cell = Aabb::new(target[0] as f64, target[1] as f64, target[2] as f64, target[0] as f64 + 1.0, target[1] as f64 + 1.0, target[2] as f64 + 1.0);
        if cell.intersects(&self.bbox()) {
            return None;
        }
        world.set(target[0], target[1], target[2], FULL);
        self.held -= 1;
        Some((target, hit.block))
    }

    /// Health as its own client shows it: a hit's damage only from the tick after it.
    pub fn shown_health(&self) -> f32 {
        (self.health + self.unseen).min(MAX_HEALTH)
    }

    /// One whole tick. Returns what happened along the way.
    pub fn tick(&mut self, world: &mut World, input: &Input) -> Events {
        self.begin(input);
        self.rest(world, input)
    }

    /// The start of a tick, up to where the client handles its keys: a hit's motion packet from the last server tick,
    /// the mod's turn and the hotbar key. With several players, every one does this
    /// and its left click (combat.rs) before anyone moves.
    pub fn begin(&mut self, input: &Input) {
        if let Some(v) = self.pending_knock.take() {
            self.dm = v;
        }
        self.turn(input.turn_yaw, input.turn_pitch);
        if let Some(s) = input.slot {
            self.slot = s;
        }
    }

    /// The rest of the tick after `begin`: the right click, then moving.
    pub fn rest(&mut self, world: &mut World, input: &Input) -> Events {
        let mut ev = Events::default();
        let next_keys = input.keys;
        // 2. Minecraft.tick: pick, then the click.
        if input.click {
            if let Some(hit) = self.pick(world) {
                ev.clicked_block = true;
                ev.placed = self.place(world, &hit);
            }
        }
        // 3. LivingEntity.baseTick
        if self.hurt_time > 0 {
            self.hurt_time -= 1;
        }
        self.ai_step(world, next_keys);
        // 4. Player.updatePlayerPose
        let want_crouch = self.keys.shift;
        if self.can_fit(world, want_crouch) {
            self.crouch_pose = want_crouch;
        } else if self.can_fit(world, true) {
            self.crouch_pose = true;
        }
        // LocalPlayer.sendIsSprintingIfNeeded: the server hears of a change in sprinting, and only of a change.
        if self.sprinting != self.sent_sprinting {
            self.sent_sprinting = self.sprinting;
            self.server_sprinting = self.sprinting;
        }
        // 5. The server: fall damage from the reported landing, sent back with its own motion for the player.
        ev.damage = self.server_tick();
        ev
    }

    fn server_tick(&mut self) -> f32 {
        // ServerGamePacketListenerImpl.handleMovePlayer -> doCheckFallDamage, with the move the client reported.
        // The landing is seen through on_ground, the fall through the server's count, which restarts on any rise.
        let mut damage = 0.0;
        if self.on_ground && self.server_fall > 0.0 {
            let dmg = (self.server_fall + 1.0e-6 - SAFE_FALL).floor();
            if dmg > 0.0 {
                damage = dmg as f32;
                self.health = (self.health - damage).max(0.0);
                self.hurt_time = 10; // handleDamageEvent
                self.dm = self.server_motion; // markHurt -> ClientboundSetEntityMotionPacket -> lerpMotion
            }
            self.server_fall = 0.0;
        } else if self.on_ground {
            self.server_fall = 0.0;
        }
        if !self.fighting {
            self.count_down();
        }
        damage
    }

    /// The server's counters for this player once a tick: hurt resistance down, and healing (1.8's full food, or
    /// Peaceful). In a fight the mod does this after the bots have hit (Fight18.serverTick), so fight_tick calls it then.
    pub fn count_down(&mut self) {
        self.age += 1;
        if self.resist > 0 {
            self.resist -= 1; // EntityPlayerMP.onUpdate
        }
        if self.rules18 {
            // FoodStats with full food: one health point every 80 ticks while hurt.
            if self.health < MAX_HEALTH && self.health > 0.0 {
                self.heal_timer += 1;
                if self.heal_timer >= 80 {
                    self.health = (self.health + 1.0).min(MAX_HEALTH);
                    self.heal_timer = 0;
                }
            } else {
                self.heal_timer = 0;
            }
        } else if self.age % 20 == 0 && self.health < MAX_HEALTH && self.health > 0.0 {
            // ServerPlayer.tickRegeneration: on Peaceful, one health point every 20 ticks (never for the dead).
            self.health = (self.health + 1.0).min(MAX_HEALTH);
        }
    }

    /// LocalPlayer.aiStep, then LivingEntity.aiStep and travel.
    fn ai_step(&mut self, world: &mut World, next_keys: Keys) {
        if self.sprint_trigger > 0 {
            self.sprint_trigger -= 1;
        }
        let was_shift = self.keys.shift;
        let had_forward_impulse = self.move_vec[1] > 1.0e-5;
        // Crouching follows the shift key as it was before this tick's input.
        self.crouching = self.can_fit(world, true) && (was_shift || !self.can_fit(world, false));
        // input.tick (KeyboardInput)
        self.keys = next_keys;
        let left = impulse(next_keys.left, next_keys.right);
        let fwd = impulse(next_keys.forward, next_keys.back);
        let len = (left * left + fwd * fwd).sqrt();
        self.move_vec = if len < 1.0e-4 { [0.0, 0.0] } else { [left / len, fwd / len] };
        let has_forward_impulse = self.move_vec[1] > 1.0e-5;
        // Pushed out of a block it stands in, such as one just placed against it.
        let w = WIDTH * 0.35;
        for (dx, dz) in [(-w, w), (-w, -w), (w, -w), (w, w)] {
            self.move_towards_closest_space(world, self.pos[0] + dx, self.pos[2] + dz);
        }
        if was_shift || next_keys.back {
            self.sprint_trigger = 0;
        }
        let can_start_sprinting = !self.sprinting && has_forward_impulse && !self.crouching;
        if can_start_sprinting {
            if !had_forward_impulse {
                if self.sprint_trigger > 0 {
                    self.sprinting = true;
                } else {
                    self.sprint_trigger = SPRINT_WINDOW;
                }
            }
            if next_keys.sprint {
                self.sprinting = true;
            }
        }
        if self.sprinting && (!has_forward_impulse || self.horizontal_collision && !self.minor_horizontal_collision) {
            self.sprinting = false;
        }
        // LivingEntity.aiStep
        if self.no_jump_delay > 0 {
            self.no_jump_delay -= 1;
        }
        if self.dm[0] * self.dm[0] + self.dm[2] * self.dm[2] < 9.0e-6 {
            self.dm[0] = 0.0;
            self.dm[2] = 0.0;
        }
        if self.dm[1].abs() < 0.003 {
            self.dm[1] = 0.0;
        }
        self.apply_input();
        if next_keys.jump {
            if self.on_ground && self.no_jump_delay == 0 {
                self.jump_from_ground();
                self.no_jump_delay = 10;
            }
        } else {
            self.no_jump_delay = 0;
        }
        self.travel(world);
    }

    /// LocalPlayer.moveTowardsClosestSpace: inside a block at this corner, step toward the nearest free side.
    fn move_towards_closest_space(&mut self, world: &World, x: f64, z: f64) {
        let pos = [mth::floor(x), mth::floor(self.pos[1]), mth::floor(z)];
        if !self.suffocates_at(world, pos[0], pos[2]) {
            return;
        }
        let (xd, zd) = (x - pos[0] as f64, z - pos[2] as f64);
        let mut best: Option<(usize, i32)> = None; // axis (0 x, 2 z) and step
        let mut closest = f64::MAX;
        for (axis, step) in [(0usize, -1), (0, 1), (2, -1), (2, 1)] {
            let along = if axis == 0 { xd } else { zd };
            let to_edge = if step > 0 { 1.0 - along } else { along };
            let (nx, nz) = if axis == 0 { (pos[0] + step, pos[2]) } else { (pos[0], pos[2] + step) };
            if to_edge < closest && !self.suffocates_at(world, nx, nz) {
                closest = to_edge;
                best = Some((axis, step));
            }
        }
        if let Some((axis, step)) = best {
            self.dm[axis] = 0.1 * step as f64;
        }
    }

    fn suffocates_at(&self, world: &World, x: i32, z: i32) -> bool {
        let b = self.bbox();
        world.suffocates(&Aabb::new(x as f64, b.min[1], z as f64, x as f64 + 1.0, b.max[1], z as f64 + 1.0).deflate(1.0e-7))
    }

    /// LocalPlayer.applyInput with modifyInput.
    fn apply_input(&mut self) {
        let mut v = self.move_vec;
        if v[0] * v[0] + v[1] * v[1] != 0.0 {
            v = [v[0] * 0.98, v[1] * 0.98];
            if self.crouching {
                v = [v[0] * SNEAK_SPEED, v[1] * SNEAK_SPEED];
            }
            // modifyInputSpeedForSquareMovement
            let length = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if length > 0.0 {
                let d = [v[0] * (1.0 / length), v[1] * (1.0 / length)];
                let (ax, ay) = (d[0].abs(), d[1].abs());
                let tan = if ay > ax { ax / ay } else { ay / ax };
                let to_square = (1.0 + tan * tan).sqrt();
                let new_length = (length * to_square).min(1.0);
                v = [d[0] * new_length, d[1] * new_length];
            }
        }
        self.xxa = v[0];
        self.zza = v[1];
    }

    fn jump_from_ground(&mut self) {
        self.dm[1] = (JUMP_POWER as f64).max(self.dm[1]);
        if self.sprinting {
            let angle = self.yaw * mth::DEG_TO_RAD_F;
            self.dm[0] += -(mth::sin(angle as f64) as f64) * 0.2;
            self.dm[2] += mth::cos(angle as f64) as f64 * 0.2;
        }
    }

    /// LivingEntity.travelInAir: push from the keys, move, then gravity and drag.
    fn travel(&mut self, world: &mut World) {
        let block_friction: f32 = if self.on_ground { 0.6 } else { 1.0 };
        let speed = if self.on_ground {
            if block_friction > 0.6 {
                self.speed() * (0.21600002f32 / (block_friction * block_friction * block_friction))
            } else {
                self.speed()
            }
        } else if self.sprinting {
            0.025999999f32
        } else {
            0.02f32
        };
        // moveRelative / getInputVector
        let input = [self.xxa as f64, 0.0, self.zza as f64];
        let length = input[0] * input[0] + input[2] * input[2];
        if length >= 1.0e-7 {
            let mut m = input;
            if length > 1.0 {
                let l = length.sqrt();
                m = [m[0] / l, 0.0, m[2] / l];
            }
            let s = speed as f64;
            m = [m[0] * s, 0.0, m[2] * s];
            let r = self.yaw * mth::DEG_TO_RAD_F;
            let sin = mth::sin(r as f64) as f64;
            let cos = mth::cos(r as f64) as f64;
            self.dm[0] += m[0] * cos - m[2] * sin;
            self.dm[2] += m[2] * cos + m[0] * sin;
        }
        let delta = self.dm;
        self.move_self(world, delta);
        let movement_y = self.dm[1] - GRAVITY;
        let friction = (block_friction * 0.91f32) as f64;
        self.dm = [self.dm[0] * friction, movement_y * 0.98f32 as f64, self.dm[2] * friction];
    }

    /// Entity.move for MoverType.SELF.
    fn move_self(&mut self, world: &World, delta_in: [f64; 3]) {
        let delta = self.back_off_from_edge(world, delta_in);
        let movement = self.collide(world, delta);
        let moved_sq = movement[0] * movement[0] + movement[1] * movement[1] + movement[2] * movement[2];
        let delta_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        if moved_sq > 1.0e-7 || delta_sq - moved_sq < 1.0e-7 {
            self.pos = [self.pos[0] + movement[0], self.pos[1] + movement[1], self.pos[2] + movement[2]];
        }
        let x_collision = !mth::equal(delta[0], movement[0]);
        let z_collision = !mth::equal(delta[2], movement[2]);
        self.horizontal_collision = x_collision || z_collision;
        let vertical_collision = delta[1] != movement[1];
        let below = vertical_collision && delta[1] < 0.0;
        self.on_ground = below;
        self.minor_horizontal_collision = self.horizontal_collision && self.collision_is_minor(movement);
        // Entity.checkFallDamage on the client: the count only, damage comes from the server.
        if movement[1] < 0.0 {
            self.fall_distance -= (movement[1] as f32) as f64;
        }
        if movement[1] > 0.0 {
            self.server_fall = 0.0;
        } else if movement[1] < 0.0 {
            self.server_fall -= (movement[1] as f32) as f64;
        }
        if self.on_ground {
            self.fall_distance = 0.0;
        }
        // restituteMovementAfterCollisions with no bounce
        if (delta[1].abs() > 0.0 && vertical_collision) || self.horizontal_collision {
            if x_collision {
                self.dm[0] = -self.dm[0] * 0.0;
            }
            if z_collision {
                self.dm[2] = -self.dm[2] * 0.0;
            }
            if vertical_collision {
                self.dm[1] = 0.0;
            }
        }
    }

    /// LocalPlayer.isHorizontalCollisionMinor: bumped into something at a slant, not head on.
    fn collision_is_minor(&self, movement: [f64; 3]) -> bool {
        let r = self.yaw * mth::DEG_TO_RAD_F;
        let s = mth::sin(r as f64) as f64;
        let c = mth::cos(r as f64) as f64;
        let gx = self.xxa as f64 * c - self.zza as f64 * s;
        let gz = self.zza as f64 * c + self.xxa as f64 * s;
        let a = gx * gx + gz * gz;
        let m = movement[0] * movement[0] + movement[2] * movement[2];
        if a < 1.0e-5f32 as f64 || m < 1.0e-5f32 as f64 {
            return false;
        }
        let dot = gx * movement[0] + gz * movement[2];
        (dot / (a * m).sqrt()).acos() < 0.13962634f32 as f64
    }

    /// Player.maybeBackOffFromEdge: with shift down, never step off an edge it could fall from.
    fn back_off_from_edge(&self, world: &World, delta: [f64; 3]) -> [f64; 3] {
        let max_down = STEP_HEIGHT as f64;
        if delta[1] > 0.0 || !self.keys.shift || !self.above_ground(world, STEP_HEIGHT) {
            return delta;
        }
        let (mut dx, mut dz) = (delta[0], delta[2]);
        let step_x = dx.signum() * 0.05;
        let step_z = dz.signum() * 0.05;
        let sx = if dx == 0.0 { 0.0 } else { step_x };
        let sz = if dz == 0.0 { 0.0 } else { step_z };
        while dx != 0.0 && self.can_fall_at_least(world, dx, 0.0, max_down) {
            if dx.abs() <= 0.05 {
                dx = 0.0;
                break;
            }
            dx -= sx;
        }
        while dz != 0.0 && self.can_fall_at_least(world, 0.0, dz, max_down) {
            if dz.abs() <= 0.05 {
                dz = 0.0;
                break;
            }
            dz -= sz;
        }
        while dx != 0.0 && dz != 0.0 && self.can_fall_at_least(world, dx, dz, max_down) {
            if dx.abs() <= 0.05 {
                dx = 0.0;
            } else {
                dx -= sx;
            }
            if dz.abs() <= 0.05 {
                dz = 0.0;
            } else {
                dz -= sz;
            }
        }
        [dx, delta[1], dz]
    }

    fn above_ground(&self, world: &World, max_down: f32) -> bool {
        let md = max_down as f64;
        self.on_ground || self.fall_distance < md && !self.can_fall_at_least(world, 0.0, 0.0, md - self.fall_distance)
    }

    fn can_fall_at_least(&self, world: &World, dx: f64, dz: f64, min_height: f64) -> bool {
        let b = self.bbox();
        world.no_collision(&Aabb::new(b.min[0] + 1.0e-7 + dx, b.min[1] - min_height - 1.0e-7, b.min[2] + 1.0e-7 + dz, b.max[0] - 1.0e-7 + dx, b.min[1], b.max[2] - 1.0e-7 + dz))
    }

    /// Entity.collide: blocks first, then stepping up onto something low.
    fn collide(&mut self, world: &World, movement: [f64; 3]) -> [f64; 3] {
        let aabb = self.bbox();
        let mut colliders = std::mem::take(&mut self.scratch);
        let step = if movement[0] == 0.0 && movement[1] == 0.0 && movement[2] == 0.0 {
            movement
        } else {
            world.boxes_near(&aabb.expand_towards(movement), &mut colliders);
            collide_with_shapes(movement, &aabb, &colliders)
        };
        let x_c = movement[0] != step[0];
        let y_c = movement[1] != step[1];
        let z_c = movement[2] != step[2];
        let on_ground_after = y_c && movement[1] < 0.0;
        let mut result = step;
        if (on_ground_after || self.on_ground) && (x_c || z_c) {
            let grounded = if on_ground_after { aabb.moved([0.0, step[1], 0.0]) } else { aabb };
            let mut up = grounded.expand_towards([movement[0], STEP_HEIGHT as f64, movement[2]]);
            if !on_ground_after {
                up = up.expand_towards([0.0, -1.0e-5f32 as f64, 0.0]);
            }
            world.boxes_near(&up, &mut colliders);
            let skip = step[1] as f32;
            let mut heights: Vec<f32> = Vec::with_capacity(4);
            for b in &colliders {
                for coord in [b.min[1], b.max[1]] {
                    let rel = (coord - grounded.min[1]) as f32;
                    if rel < 0.0 || rel == skip || rel > STEP_HEIGHT {
                        continue;
                    }
                    if !heights.contains(&rel) {
                        heights.push(rel);
                    }
                }
            }
            heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let flat = step[0] * step[0] + step[2] * step[2];
            for h in heights {
                let try_step = collide_with_shapes([movement[0], h as f64, movement[2]], &grounded, &colliders);
                if try_step[0] * try_step[0] + try_step[2] * try_step[2] > flat {
                    let below = aabb.min[1] - grounded.min[1];
                    result = [try_step[0], try_step[1] - below, try_step[2]];
                    break;
                }
            }
        }
        self.scratch = colliders;
        result
    }
}

#[inline]
fn impulse(positive: bool, negative: bool) -> f32 {
    if positive == negative {
        0.0
    } else if positive {
        1.0
    } else {
        -1.0
    }
}

pub fn box_at(pos: [f64; 3], height: f64) -> Aabb {
    let h = WIDTH / 2.0;
    Aabb::new(pos[0] - h, pos[1], pos[2] - h, pos[0] + h, pos[1] + height, pos[2] + h)
}

/// Entity.collideWithShapes: y first, then the larger of x and z.
fn collide_with_shapes(movement: [f64; 3], aabb: &Aabb, shapes: &[Aabb]) -> [f64; 3] {
    if shapes.is_empty() {
        return movement;
    }
    let order: [usize; 3] = if movement[0].abs() < movement[2].abs() { [1, 2, 0] } else { [1, 0, 2] };
    let mut done = [0.0f64; 3];
    for axis in order {
        let d = movement[axis];
        if d != 0.0 {
            let moved = aabb.moved(done);
            done[axis] = collide_axis(axis, &moved, shapes, d);
        }
    }
    done
}

/// Shapes.collide for boxes: how far the box can go along one axis before it touches one of them.
fn collide_axis(axis: usize, moving: &Aabb, shapes: &[Aabb], mut d: f64) -> f64 {
    let (b, c) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    for s in shapes {
        if d.abs() < 1.0e-7 {
            return 0.0;
        }
        // Overlap on the other two axes, by the same small margin the game uses.
        if !(moving.min[b] + 1.0e-7 < s.max[b] && moving.max[b] - 1.0e-7 >= s.min[b]) {
            continue;
        }
        if !(moving.min[c] + 1.0e-7 < s.max[c] && moving.max[c] - 1.0e-7 >= s.min[c]) {
            continue;
        }
        if d > 0.0 {
            if moving.max[axis] - 1.0e-7 < s.min[axis] {
                let nd = s.min[axis] - moving.max[axis];
                if nd >= -1.0e-7 {
                    d = d.min(nd);
                }
            }
        } else if moving.min[axis] + 1.0e-7 >= s.max[axis] {
            let nd = s.max[axis] - moving.min[axis];
            if nd <= 1.0e-7 {
                d = d.max(nd);
            }
        }
    }
    d
}

/// BlockGetter.clip with block outline shapes: the first block along the line from `from` to `to`.
pub fn clip(world: &World, from: [f64; 3], to: [f64; 3]) -> Option<Hit> {
    if from == to {
        return None;
    }
    let lerp = |t: f64, a: f64, b: f64| a + t * (b - a);
    let to_x = lerp(-1.0e-7, to[0], from[0]);
    let to_y = lerp(-1.0e-7, to[1], from[1]);
    let to_z = lerp(-1.0e-7, to[2], from[2]);
    let from_x = lerp(-1.0e-7, from[0], to[0]);
    let from_y = lerp(-1.0e-7, from[1], to[1]);
    let from_z = lerp(-1.0e-7, from[2], to[2]);
    let mut bx = mth::floor(from_x);
    let mut by = mth::floor(from_y);
    let mut bz = mth::floor(from_z);
    if let Some(h) = clip_block(world, from, to, [bx, by, bz]) {
        return Some(h);
    }
    let (dx, dy, dz) = (to_x - from_x, to_y - from_y, to_z - from_z);
    let sign = |v: f64| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 };
    let (sx, sy, sz) = (sign(dx), sign(dy), sign(dz));
    let t_dx = if sx == 0 { f64::MAX } else { sx as f64 / dx };
    let t_dy = if sy == 0 { f64::MAX } else { sy as f64 / dy };
    let t_dz = if sz == 0 { f64::MAX } else { sz as f64 / dz };
    let mut tx = t_dx * if sx > 0 { 1.0 - mth::frac(from_x) } else { mth::frac(from_x) };
    let mut ty = t_dy * if sy > 0 { 1.0 - mth::frac(from_y) } else { mth::frac(from_y) };
    let mut tz = t_dz * if sz > 0 { 1.0 - mth::frac(from_z) } else { mth::frac(from_z) };
    while tx <= 1.0 || ty <= 1.0 || tz <= 1.0 {
        if tx < ty {
            if tx < tz {
                bx += sx;
                tx += t_dx;
            } else {
                bz += sz;
                tz += t_dz;
            }
        } else if ty < tz {
            by += sy;
            ty += t_dy;
        } else {
            bz += sz;
            tz += t_dz;
        }
        if let Some(h) = clip_block(world, from, to, [bx, by, bz]) {
            return Some(h);
        }
    }
    None
}

/// VoxelShape.clip for the boxes of one block.
fn clip_block(world: &World, from: [f64; 3], to: [f64; 3], pos: [i32; 3]) -> Option<Hit> {
    let id = world.get(pos[0], pos[1], pos[2]);
    if id == AIR {
        return None;
    }
    let shape = &kinds()[id as usize].boxes;
    if shape.is_empty() {
        return None;
    }
    let diff = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    if diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] < 1.0e-7 {
        return None;
    }
    let p = [pos[0] as f64, pos[1] as f64, pos[2] as f64];
    let test = [from[0] + diff[0] * 0.001, from[1] + diff[1] * 0.001, from[2] + diff[2] * 0.001];
    for b in shape {
        let w = b.moved(p);
        if (0..3).all(|i| test[i] >= w.min[i] && test[i] < w.max[i]) {
            // Inside the block already: the hit is right there, on the side facing back along the line.
            return Some(Hit { block: pos, face: opposite(nearest_direction(diff)), loc: test });
        }
    }
    let mut best_t = 1.0;
    let mut best: Option<(u8, [f64; 3])> = None;
    for b in shape {
        let w = b.moved(p);
        if let Some((t, face)) = clip_box(&w, from, diff, best_t) {
            best_t = t;
            best = Some((face, [from[0] + diff[0] * t, from[1] + diff[1] * t, from[2] + diff[2] * t]));
        }
    }
    best.map(|(face, loc)| Hit { block: pos, face, loc })
}

/// AABB.clip: where the line enters the box, and through which side.
pub(crate) fn clip_box(b: &Aabb, from: [f64; 3], d: [f64; 3], max_t: f64) -> Option<(f64, u8)> {
    let mut t_best = max_t;
    let mut face = None;
    // (axis, plane, side), checked in the game's order: x, y, z.
    let mut check = |axis: usize, plane: f64, side: u8| {
        let t = (plane - from[axis]) / d[axis];
        if t > 0.0 && t < t_best {
            let (o1, o2) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let a = from[o1] + t * d[o1];
            let c = from[o2] + t * d[o2];
            if b.min[o1] - 1.0e-7 < a && a < b.max[o1] + 1.0e-7 && b.min[o2] - 1.0e-7 < c && c < b.max[o2] + 1.0e-7 {
                t_best = t;
                face = Some(side);
            }
        }
    };
    if d[0] > 1.0e-7 {
        check(0, b.min[0], 4);
    } else if d[0] < -1.0e-7 {
        check(0, b.max[0], 5);
    }
    if d[1] > 1.0e-7 {
        check(1, b.min[1], 0);
    } else if d[1] < -1.0e-7 {
        check(1, b.max[1], 1);
    }
    if d[2] > 1.0e-7 {
        check(2, b.min[2], 2);
    } else if d[2] < -1.0e-7 {
        check(2, b.max[2], 3);
    }
    face.map(|f| (t_best, f))
}

fn nearest_direction(d: [f64; 3]) -> u8 {
    // Direction.getApproximateNearest: the axis direction closest to the vector.
    let dirs: [[f64; 3]; 6] = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut best = 0;
    let mut best_dot = f64::MIN;
    for (i, v) in dirs.iter().enumerate() {
        let dot = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
        if dot > best_dot {
            best_dot = dot;
            best = i as u8;
        }
    }
    best
}

fn opposite(face: u8) -> u8 {
    face ^ 1
}
