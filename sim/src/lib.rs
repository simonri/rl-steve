//! Minecraft 26.3 player physics and the training courses, in Rust, for fast batched training.

pub mod clutch;
pub mod env;
pub mod mth;
pub mod player;
pub mod world;

#[cfg(feature = "python")]
mod python;
