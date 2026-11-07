//! What can be in a player's hotbar: every item of Hypixel Bed Wars' item shop, plus the wooden sword it starts with.
//!
//! Source: the Hypixel forum guide "Every Item In Bedwars" (ThePrestige, 2020, edited April 2024); the official wiki
//! was unreachable (docs/training-design.md#foremal). The list is fixed and only ever grows at the end: an item's
//! index is part of what the model sees (a one-hot per hotbar slot), so moving one would change what it has learned.
//! Items the simulator cannot do yet (a ladder, a pickaxe, TNT...) are in the list but never given out: a ladder that
//! acted like a full block would make the model's inputs untrue. items.py has the same list.

pub type Item = u8;

pub const EMPTY: Item = 0;
// Blocks
pub const WOOL: Item = 1;
pub const HARDENED_CLAY: Item = 2;
pub const BLAST_PROOF_GLASS: Item = 3;
pub const END_STONE: Item = 4;
pub const LADDER: Item = 5;
pub const OAK_PLANKS: Item = 6;
pub const OBSIDIAN: Item = 7;
// Melee
pub const WOODEN_SWORD: Item = 8;
pub const STONE_SWORD: Item = 9;
pub const IRON_SWORD: Item = 10;
pub const DIAMOND_SWORD: Item = 11;
pub const KNOCKBACK_STICK: Item = 12;
// Tools
pub const WOODEN_PICKAXE: Item = 13;
pub const IRON_PICKAXE: Item = 14;
pub const GOLDEN_PICKAXE: Item = 15;
pub const DIAMOND_PICKAXE: Item = 16;
pub const WOODEN_AXE: Item = 17;
pub const STONE_AXE: Item = 18;
pub const IRON_AXE: Item = 19;
pub const DIAMOND_AXE: Item = 20;
pub const SHEARS: Item = 21;
// Ranged
pub const BOW: Item = 22;
pub const ARROW: Item = 23;
// Potions
pub const SPEED_POTION: Item = 24;
pub const JUMP_POTION: Item = 25;
pub const INVISIBILITY_POTION: Item = 26;
// Utility
pub const GOLDEN_APPLE: Item = 27;
pub const BED_BUG: Item = 28;
pub const DREAM_DEFENDER: Item = 29;
pub const FIREBALL: Item = 30;
pub const TNT: Item = 31;
pub const ENDER_PEARL: Item = 32;
pub const WATER_BUCKET: Item = 33;
pub const BRIDGE_EGG: Item = 34;
pub const MAGIC_MILK: Item = 35;
pub const SPONGE: Item = 36;
pub const POP_UP_TOWER: Item = 37;
/// Anything else the game shows in a hotbar slot.
pub const OTHER: Item = 38;
pub const KINDS: usize = 39;

pub const SLOTS: usize = 9;
/// Per hotbar slot: its kind (one-hot over KINDS) and how many (count / 64); then which slot is selected.
pub const HOTBAR_F: usize = SLOTS * (KINDS + 1) + SLOTS;

/// A block the simulator places as a full cube (wool, clay, glass, end stone, planks, obsidian). Not the ladder.
pub fn placeable(item: Item) -> bool {
    matches!(item, WOOL | HARDENED_CLAY | BLAST_PROOF_GLASS | END_STONE | OAK_PLANKS | OBSIDIAN)
}

pub fn is_sword(item: Item) -> bool {
    matches!(item, WOODEN_SWORD | STONE_SWORD | IRON_SWORD | DIAMOND_SWORD)
}

/// One hotbar slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stack {
    pub item: Item,
    pub count: i32,
}

/// The hotbar's part of the model's input (HOTBAR_F values), as a player sees it at the bottom of the screen.
pub fn features(hotbar: &[Stack; SLOTS], selected: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), HOTBAR_F);
    out.fill(0.0);
    for (i, s) in hotbar.iter().enumerate() {
        let o = &mut out[i * (KINDS + 1)..(i + 1) * (KINDS + 1)];
        if s.count > 0 && s.item != EMPTY {
            o[(s.item as usize).min(KINDS - 1)] = 1.0;
            o[KINDS] = s.count as f32 / 64.0;
        } else {
            o[EMPTY as usize] = 1.0;
        }
    }
    out[SLOTS * (KINDS + 1) + (selected as usize).min(SLOTS - 1)] = 1.0;
}
