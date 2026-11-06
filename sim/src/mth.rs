//! Minecraft's own math, so results match the game to the last bit where it matters.

use std::sync::OnceLock;

static SIN: OnceLock<Vec<f32>> = OnceLock::new();

fn table() -> &'static [f32] {
    SIN.get_or_init(|| (0..65536).map(|i| (i as f64 / 10430.378350470453).sin() as f32).collect())
}

/// Mth.sin: a lookup table of 65536 steps, as the game uses for movement and view.
#[inline]
pub fn sin(v: f64) -> f32 {
    table()[((v * 10430.378350470453) as i64 & 65535) as usize]
}

/// Mth.cos.
#[inline]
pub fn cos(v: f64) -> f32 {
    table()[((v * 10430.378350470453 + 16384.0) as i64 & 65535) as usize]
}

#[inline]
pub fn floor(v: f64) -> i32 {
    v.floor() as i32
}

/// Mth.frac: the part after the last whole number, also below zero.
#[inline]
pub fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Mth.equal on doubles.
#[inline]
pub fn equal(a: f64, b: f64) -> bool {
    (b - a).abs() < 1.0e-5
}

/// (float)(Math.PI / 180.0), worked out in double and then rounded, as Java does.
pub const DEG_TO_RAD_F: f32 = (std::f64::consts::PI / 180.0) as f32;
