//! A fight's shaping: the part of its reward that is neither winning, losing nor damage. Why it is built this way,
//! with sources: docs/training-design.md (#nollsumma, #potential, #nedtrappning). In short:
//!
//! - Aim is a potential, Φ = −AIM_POTENTIAL · (angle off the opponent's chest), paid as γ·Φ(s′) − Φ(s) with PPO's own
//!   γ and Φ = 0 when the round ends. Over a round that adds up to Φ(end) − Φ(start), so it cannot be farmed and does
//!   not change which play is best, only how soon it is found (Ng, Harada & Russell 1999).
//! - Every term is zero-sum: the model's own minus its opponents' average, so two sides cannot both gain by playing
//!   along (OpenAI Five; GT Sophy). The opponents' part is what they did, counted by the same rules.
//! - The event bonuses (a hit from further off, a critical hit, the cost of a missed swing) help find those plays and
//!   are scaled by a weight that falls as the duel curriculum is done (train_auto.bonus_weight_for).
//!   Damage, winning and losing are what count; a critical hit already does half as much damage again.
//!
//! env.rs fight_step and AutoEnv._fight_tick both use these functions, so the two agree to the bit.

use crate::player::Player;

/// The default discount (train_auto.GAMMA). A potential's shaping keeps the best play the same only with the discount
/// PPO learns with, so a run that learns with another one sets it on every round (Env.gamma, Batch.set_gamma).
pub const GAMMA: f64 = 0.995;
/// Per radian between the view and the line to the opponent's chest: turning from beside it (90°) to it pays 0.16.
pub const AIM_POTENTIAL: f64 = 0.1;
/// A hit that lands from further off: this per block across (feet to feet) beyond LONG_HIT_FROM, about 0.5 at the
/// edge of reach; the one who hits from further away takes fewer hits back.
pub const LONG_HIT_POINTS: f64 = 0.4;
pub const LONG_HIT_FROM: f64 = 2.0;
/// A critical hit (landed while falling).
pub const CRIT_POINTS: f64 = 0.3;
/// A left click that is not on an opponent in reach.
pub const SWING_COST: f64 = 0.01;
/// A left click with no opponent in view: none in a course, or every one still in the fight more than VIEW_HALF off
/// where it looks. A click at nobody is never of use, and in a fight it gives away where it is looking. Not weighted
/// and not zero-sum: unlike the event bonuses it is no crutch to be dropped later, and no opponent can make the model
/// pay it. Asked for by the project owner (2026-11-07): "minus för försök till attack om det finns 0 spelare i dens
/// synfält". The view is worked out from what the inputs already hold (its own yaw and pitch, the opponents'
/// positions), so nothing about them changes.
pub const BLIND_SWING_COST: f64 = 0.01;
/// Half the view cone, in radians from where it looks: 60°, about the game's horizontal half-angle at FOV 90 on a
/// 16:10 window (camera.py H_HALF). An opponent at the edge of the screen still counts as in view.
pub const VIEW_HALF: f64 = 60.0 * std::f64::consts::PI / 180.0;

/// Whether any opponent still in the fight is in view: its chest within VIEW_HALF of where `me` looks.
pub fn anyone_in_view(me: &Player, foes: &[(&Player, bool)]) -> bool {
    foes.iter().any(|(f, alive)| *alive && aim_error(me, f.pos) <= VIEW_HALF)
}

/// A hit with anything but a sword in hand (blocks, a tool, a bare hand) pays this share of a sword's: its damage and
/// its long-hit and critical-hit points. Fights are won with the sword; a hit with blocks in hand is a mistake that
/// happens to land.
pub const NON_SWORD: f64 = 0.1;

/// The angle, in radians, between `p`'s view and the line from its eyes to the chest of a player standing at `at`.
pub fn aim_error(p: &Player, at: [f64; 3]) -> f64 {
    let eye = p.eye();
    let d = [at[0] - eye[0], at[1] + 1.2 - eye[1], at[2] - eye[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n < 1e-6 {
        return 0.0;
    }
    let v = p.view();
    ((v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / n).clamp(-1.0, 1.0).acos()
}

/// The fight's potential: the model's aim at the nearest-aimed opponent still alive, minus its opponents' average aim
/// at the model (an opponent no longer alive counts 0). 0 when no opponent is alive.
pub fn potential(me: &Player, foes: &[(&Player, bool)]) -> f64 {
    let mine = foes.iter().filter(|(_, alive)| *alive).map(|(f, _)| aim_error(me, f.pos)).fold(f64::INFINITY, f64::min);
    if !mine.is_finite() {
        return 0.0;
    }
    let n = foes.len() as f64;
    let mut theirs = 0.0;
    for (f, alive) in foes {
        if *alive {
            theirs += -AIM_POTENTIAL * aim_error(f, me.pos) / n;
        }
    }
    -AIM_POTENTIAL * mine - theirs
}

/// One player's event points this tick, before the weight: [long hit, critical hit, missed swing]. `landed` is its
/// hit that counted (distance across, critical, with a sword), `swing` its left click (Some(whether it was on an
/// opponent)).
pub fn events(landed: Option<(f64, bool, bool)>, swing: Option<bool>) -> [f64; 3] {
    let mut e = [0.0; 3];
    if let Some((across, critical, sword)) = landed {
        let k = if sword { 1.0 } else { NON_SWORD };
        e[0] = k * LONG_HIT_POINTS * (across - LONG_HIT_FROM).max(0.0);
        if critical {
            e[1] = k * CRIT_POINTS;
        }
    }
    if swing == Some(false) {
        e[2] = -SWING_COST;
    }
    e
}

/// The model's events minus its opponents' average, each scaled by `weight`: [long hit, critical hit, missed swing].
pub fn zero_sum(mine: [f64; 3], foes: &[[f64; 3]], n: usize, weight: f64) -> [f64; 3] {
    let n = n.max(1) as f64;
    let mut theirs = [0.0; 3];
    for f in foes {
        for k in 0..3 {
            theirs[k] += f[k] / n;
        }
    }
    std::array::from_fn(|k| weight * (mine[k] - theirs[k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A potential's shaping adds up over a round to Φ(end) − Φ(start), whatever happens between: aiming and looking
    /// away again and again earns nothing.
    #[test]
    fn a_potential_cannot_be_farmed() {
        let foe = Player::new([0.5, 100.0, 4.5], 180.0, 0.0, 0);
        let mut me = Player::new([0.5, 100.0, 0.5], 0.0, 0.0, 0);
        let start = potential(&me, &[(&foe, true)]);
        let mut prev = start;
        let mut total = 0.0;
        for t in 0..200 {
            me.yaw = if t % 2 == 0 { 0.0 } else { 90.0 }; // on it, away, on it, ...
            let now = if t == 199 { 0.0 } else { potential(&me, &[(&foe, true)]) }; // Φ = 0 at the end
            total += GAMMA * now - prev;
            prev = now;
        }
        // With γ < 1 a little drifts in; what matters: it is nothing like the 100 aimed ticks a per-tick bonus paid.
        assert!(total.abs() < 0.3, "{total}");
        assert!(-start < 0.01, "on it from the start: {start}");
    }

    /// In view: an opponent in front or off to the side within 60°; not one behind, nor one out of the fight.
    #[test]
    fn who_is_in_view() {
        let me = Player::new([0.5, 100.0, 0.5], 0.0, 0.0, 0); // yaw 0 looks toward +z
        let ahead = Player::new([0.5, 100.0, 5.5], 180.0, 0.0, 0);
        let side = Player::new([4.5, 100.0, 4.5], 0.0, 0.0, 0); // 45° off
        let behind = Player::new([0.5, 100.0, -4.5], 0.0, 0.0, 0);
        assert!(anyone_in_view(&me, &[(&ahead, true)]));
        assert!(anyone_in_view(&me, &[(&side, true)]));
        assert!(!anyone_in_view(&me, &[(&behind, true)]));
        assert!(!anyone_in_view(&me, &[(&ahead, false)]), "out of the fight");
        assert!(!anyone_in_view(&me, &[]), "a course: nobody");
    }

    #[test]
    fn the_same_play_on_both_sides_nets_nothing() {
        let e = events(Some((3.0, true, true)), Some(true));
        assert!((events(Some((3.0, true, false)), Some(true))[1] - NON_SWORD * CRIT_POINTS).abs() < 1e-12, "a hit with blocks pays a sliver");
        assert_eq!(zero_sum(e, &[e], 1, 1.0), [0.0; 3]);
        let missed = events(None, Some(false));
        assert_eq!(zero_sum(missed, &[[0.0; 3]], 1, 0.5)[2], -SWING_COST * 0.5);
    }
}
