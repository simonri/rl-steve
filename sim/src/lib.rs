//! Minecraft 26.3 player physics and the training courses, in Rust, for fast batched training.

pub mod clutch;
pub mod combat;
pub mod duel;
pub mod env;
pub mod items;
pub mod mth;
pub mod player;
pub mod shaping;
pub mod world;

#[cfg(feature = "python")]
mod python;
