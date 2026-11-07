package dev.mcbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.authlib.GameProfile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import net.fabricmc.fabric.api.entity.FakePlayer;
import net.minecraft.network.protocol.game.ClientboundPlayerInfoRemovePacket;
import net.minecraft.network.protocol.game.ClientboundPlayerInfoUpdatePacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.gamerules.GameRules;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Combat as on Hypixel (1.8.9), on our 26.3 server, and scripted opponents as players that live only on the server.
 *
 * The rules are python/combat18.py's, the same numbers the simulator (sim/src/combat.rs) uses: no attack cooldown, damage
 * by sword tier, x1.5 when falling, armor points as (25 - armor) / 25, 20 ticks of hurt resistance where a hit in the
 * second half only counts by how much bigger it is, knockback that halves nothing (the server's motion for a player is
 * taken as 0) and adds 0.4 away and 0.4 up, plus 0.5 from the attacker's yaw and 0.1 up for a sprinting hit, which
 * also ends the attacker's sprint on the server. Healing is 1.8's: one point every 80 ticks while hurt.
 *
 * Opponents are FakePlayers whose movement the server simulates itself. Python sends their inputs every step, worked
 * out by the same scripted bots the simulator uses (sim/src/duel.rs through mcsim.bot_action).
 */
public final class Fight18 {
    private Fight18() {}

    /** On while rounds use 1.8 rules: Player.attack is replaced (PlayerAttackMixin) and healing is ours. */
    public static volatile boolean enabled = false;

    static final double[] TURN_STEPS = {-60.0, -15.0, -5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 15.0, 60.0};
    static final int MAX_RESIST = 20;
    static final double ENTITY_REACH = 3.0;
    static final double BLOCK_REACH = 4.5;
    static final int SPRINT_WINDOW = 7; // options.sprintWindow: ticks to tap forward again for a sprint

    /** What the 1.8 rules keep per player. */
    public static final class State {
        public int resist;
        public float last;
        public int heal;
        public float damage = 1.0f; // attack attribute: 1 plus the sword
        public int armor;
        public int kb;
        public int sinceHit = 99;
        public int sinceHurt = 99;
        public boolean serverSprint; // an opponent's sprint as the server knows it (a real player's is its isSprinting)
    }

    static final Map<UUID, State> STATES = new HashMap<>();

    /** The model was beaten since this was last read (attack): server thread sets it, the client's reply reads it. */
    private static volatile boolean downed;

    /** Whether the model was beaten since the last call; clears it. */
    public static boolean takeDowned() {
        boolean was = downed;
        downed = false;
        return was;
    }

    public static State of(Entity e) {
        return STATES.computeIfAbsent(e.getUUID(), k -> new State());
    }

    // ---- The rules ----

    /** A hit from attacker on target, as the 1.8.9 server works it out. */
    public static void attack(ServerPlayer attacker, Entity target) {
        if (!(target instanceof LivingEntity victim) || !victim.isAlive() || victim.getHealth() <= 0) return;
        State a = of(attacker), v = of(victim);
        float amount = attacker.getInventory().getSelectedSlot() == 1 ? a.damage : 1.0f; // the sword's, or a bare hit's (combat.rs Swing)
        boolean falling = attacker.fallDistance > 0 && !attacker.onGround() && !attacker.onClimbable() && !attacker.isInWater();
        if (falling) amount *= 1.5f;
        boolean sprinting = attacker instanceof Bot ? a.serverSprint : attacker.isSprinting();
        int level = a.kb + (sprinting ? 1 : 0);
        float dealt;
        boolean knock;
        if ((float) v.resist > MAX_RESIST / 2.0f) {
            if (amount <= v.last) return;
            dealt = amount - v.last;
            v.last = amount;
            knock = false;
        } else {
            v.last = amount;
            v.resist = MAX_RESIST;
            dealt = amount;
            knock = true;
        }
        dealt = dealt * (25 - v.armor) / 25.0f;
        if (victim.getHealth() - dealt <= 0.0f && !(victim instanceof Bot) && victim instanceof ServerPlayer) {
            // The model is beaten. It is not killed in the game: a death goes through the game's own respawn, which
            // the next round's reset races and the client was left half dead for 30 s. It keeps a sliver of health,
            // and the round is told it lost (McBridge "died").
            victim.setHealth(1.0f);
            downed = true;
        } else {
            victim.setHealth(Math.max(0.0f, victim.getHealth() - dealt));
        }
        ServerLevel level0 = (ServerLevel) victim.level();
        level0.broadcastDamageEvent(victim, level0.damageSources().playerAttack(attacker)); // the red flash and the sound
        Vec3 motion = null;
        if (knock) {
            double dx = attacker.getX() - victim.getX(), dz = attacker.getZ() - victim.getZ();
            if (dx * dx + dz * dz < 1.0e-4) {
                dx = 0.0;
                dz = 0.01;
            }
            double f = (float) Math.sqrt(dx * dx + dz * dz);
            double y = Math.min(0.4, 0.4000000059604645);
            motion = new Vec3(-dx / f * 0.4, y, -dz / f * 0.4);
        }
        if (level > 0) {
            float r = attacker.getYRot() * (float) Math.PI / 180.0f;
            Vec3 base = motion == null ? Vec3.ZERO : motion;
            motion = new Vec3(base.x + (double) (-((float) Math.sin(r)) * level * 0.5f), base.y + 0.1, base.z + (double) ((float) Math.cos(r) * level * 0.5f));
            if (attacker instanceof Bot) a.serverSprint = false; // its client keeps sprinting; the server's flag is off
            else attacker.setSprinting(false);
        }
        if (motion != null) {
            victim.setDeltaMovement(motion);
            if (!(victim instanceof FakePlayer)) victim.syncVelocity = true; // the motion packet, handled before the client's next tick
        }
        a.sinceHit = 0;
        v.sinceHurt = 0;
    }

    /** Once per server tick, before anything moves: opponents act, then hurt resistance and healing count down. */
    public static void serverTick(MinecraftServer server) {
        if (!enabled) return;
        if (McBridge.stepping()) drive(server); // nobody stepping the game (released, or Python gone): the bots stand still
        for (ServerPlayer p : server.getPlayerList().getPlayers()) tickState(p);
        for (Opponent o : OPPONENTS) tickState(o.player);
    }

    private static void tickState(ServerPlayer p) {
        State s = STATES.get(p.getUUID());
        if (s == null) return;
        if (s.resist > 0) s.resist--;
        s.sinceHit++;
        s.sinceHurt++;
        if (p.getHealth() < p.getMaxHealth() && p.getHealth() > 0) {
            if (++s.heal >= 80) {
                p.heal(1.0f);
                s.heal = 0;
            }
        } else {
            s.heal = 0;
        }
    }

    /** Sets a player's gear for a round: sword by attack value, Bedwars armor by points, Knockback, and the rules. */
    public static void gear(ServerPlayer p, float damage, int armor, int kb, boolean rules18) {
        enabled = rules18;
        if (!rules18) {
            // Back to the game's own rules: its regeneration, and the attack cooldown the 1.8 rules turned off.
            MinecraftServer server = p.level().getServer();
            server.getGameRules().set(GameRules.NATURAL_HEALTH_REGENERATION, true, server);
            var speed = p.getAttribute(Attributes.ATTACK_SPEED);
            if (speed != null) speed.setBaseValue(4.0);
        }
        if (rules18) {
            MinecraftServer server = p.level().getServer();
            server.getGameRules().set(GameRules.NATURAL_HEALTH_REGENERATION, false, server); // ours instead (tickState)
            // As on Hypixel, players do not push each other: a team with collisionRule never (the client keeps to it too).
            var board = server.getScoreboard();
            var team = board.getPlayerTeam("mcbridge");
            if (team == null) team = board.addPlayerTeam("mcbridge");
            team.setCollisionRule(net.minecraft.world.scores.Team.CollisionRule.NEVER);
            board.addPlayerToTeam(p.getScoreboardName(), team);
        }
        State s = of(p);
        s.damage = damage;
        s.armor = armor;
        s.kb = kb;
        s.resist = 0;
        s.last = 0;
        s.heal = 0;
        s.sinceHit = 99;
        s.sinceHurt = 99;
        s.serverSprint = false;
        var attackSpeed = p.getAttribute(Attributes.ATTACK_SPEED);
        if (attackSpeed != null && rules18) attackSpeed.setBaseValue(1024.0); // no cooldown: every click is a full hit
        p.getInventory().setItem(1, sword(damage));
        dress(p, armor);
    }

    static ItemStack sword(float damage) {
        if (damage >= 8) return new ItemStack(Items.DIAMOND_SWORD);
        if (damage >= 7) return new ItemStack(Items.IRON_SWORD);
        if (damage >= 6) return new ItemStack(Items.STONE_SWORD);
        if (damage >= 5) return new ItemStack(Items.WOODEN_SWORD);
        return ItemStack.EMPTY;
    }

    /** The team's leather helmet and chestplate, with leggings and boots by the armor points (combat18.bedwars_armor). */
    static void dress(ServerPlayer p, int armor) {
        p.setItemSlot(EquipmentSlot.HEAD, armor > 0 ? new ItemStack(Items.LEATHER_HELMET) : ItemStack.EMPTY);
        p.setItemSlot(EquipmentSlot.CHEST, armor > 0 ? new ItemStack(Items.LEATHER_CHESTPLATE) : ItemStack.EMPTY);
        if (armor >= 13) {
            p.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.DIAMOND_LEGGINGS));
            p.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.DIAMOND_BOOTS));
        } else if (armor >= 11) {
            p.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.IRON_LEGGINGS));
            p.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.IRON_BOOTS));
        } else if (armor > 0) {
            p.setItemSlot(EquipmentSlot.LEGS, new ItemStack(Items.LEATHER_LEGGINGS));
            p.setItemSlot(EquipmentSlot.FEET, new ItemStack(Items.LEATHER_BOOTS));
        } else {
            p.setItemSlot(EquipmentSlot.LEGS, ItemStack.EMPTY);
            p.setItemSlot(EquipmentSlot.FEET, ItemStack.EMPTY);
        }
    }

    // ---- Opponents ----

    /** A player only the server has, moved by the server as it would move a mob. */
    static final class Bot extends FakePlayer {
        Bot(ServerLevel level, GameProfile profile) {
            super(level, profile);
        }

        @Override
        public boolean isClientAuthoritative() {
            return false; // the server moves it (LivingEntity.travel), there being no client
        }

        @Override
        public void tick() {
            // ticked by drive(), after its inputs are set
        }

        boolean crouching; // as its client would have it, set by drive()

        /**
         * Its keys as a client turns them into movement (LocalPlayer.applyInput with modifyInput, player.rs apply_input),
         * not as the server does for others (only the 0.98): a diagonal on the keys is as fast as straight ahead.
         */
        @Override
        protected void applyInput() {
            float x = this.xxa, z = this.zza; // drive() sets the keys' direction, of length 1
            if (x * x + z * z != 0.0f) {
                x *= 0.98f;
                z *= 0.98f;
                if (crouching) {
                    x *= 0.3f;
                    z *= 0.3f;
                }
                float length = (float) Math.sqrt(x * x + z * z);
                if (length > 0.0f) {
                    float dx = x * (1.0f / length), dz = z * (1.0f / length);
                    float ax = Math.abs(dx), az = Math.abs(dz);
                    float tan = az > ax ? ax / az : az / ax;
                    float toSquare = (float) Math.sqrt(1.0f + tan * tan);
                    float newLength = Math.min(length * toSquare, 1.0f);
                    x = dx * newLength;
                    z = dz * newLength;
                }
            }
            this.xxa = x;
            this.zza = z;
        }
    }

    static final class Opponent {
        final Bot player;
        final int bot;
        boolean[] keys = new boolean[7];
        int[] keyAge = {2, 2, 2, 2, 2, 2, 2};
        int[] action = {0, 0, 0, 0, 0, 0, 5, 5, 0};
        // Its client's sprint (LocalPlayer.aiStep), as player.rs has it.
        boolean sprinting, sentSprinting, wasShift;
        int sprintTrigger;
        float lastForward;

        Opponent(Bot player, int bot) {
            this.player = player;
            this.bot = bot;
        }
    }

    static final List<Opponent> OPPONENTS = new ArrayList<>();

    /** Brings in an opponent where the round puts it, facing the given way, with its gear. */
    public static void spawn(MinecraftServer server, JsonObject cmd) {
        ServerLevel level = server.overworld();
        int i = OPPONENTS.size();
        GameProfile profile = new GameProfile(UUID.nameUUIDFromBytes(("mcbridge-opponent-" + i).getBytes()), "Enemy " + (i + 1)); // the name over its head
        Bot p = new Bot(level, profile);
        JsonArray pos = cmd.getAsJsonArray("pos");
        float yaw = cmd.get("yaw").getAsFloat();
        p.setPos(pos.get(0).getAsDouble(), pos.get(1).getAsDouble(), pos.get(2).getAsDouble());
        p.setYRot(yaw);
        p.setYHeadRot(yaw);
        p.setXRot(0);
        // It stands on the arena as a player at rest does: on the ground, gravity's last tick in its motion (so its first
        // move meets the floor again). Otherwise its first steps are air steps (0.02 instead of 0.1).
        p.setOnGround(true);
        p.setDeltaMovement(0.0, -0.0784000015258789, 0.0);
        p.setHealth(p.getMaxHealth());
        p.setGameMode(GameType.SURVIVAL);
        server.getPlayerList().broadcastAll(ClientboundPlayerInfoUpdatePacket.createPlayerInitializing(List.of(p)));
        level.addNewPlayer(p);
        gear(p, cmd.get("damage").getAsFloat(), cmd.get("armor").getAsInt(), cmd.get("kb").getAsInt(), cmd.get("rules18").getAsBoolean());
        p.getInventory().setSelectedSlot(1);
        OPPONENTS.add(new Opponent(p, cmd.get("bot").getAsInt()));
    }

    /** Tells a watcher who just joined about the bots, which are not in the player list it was sent. */
    public static void introduce(ServerPlayer watcher) {
        if (OPPONENTS.isEmpty()) return;
        List<ServerPlayer> bots = new ArrayList<>();
        for (Opponent o : OPPONENTS) bots.add(o.player);
        watcher.connection.send(ClientboundPlayerInfoUpdatePacket.createPlayerInitializing(bots));
    }

    /** Takes every opponent out of the world, before a new round. */
    public static void clear(MinecraftServer server) {
        if (OPPONENTS.isEmpty()) return;
        List<UUID> ids = new ArrayList<>();
        for (Opponent o : OPPONENTS) {
            ids.add(o.player.getUUID());
            server.overworld().removePlayerImmediately(o.player, Entity.RemovalReason.DISCARDED);
            STATES.remove(o.player.getUUID());
        }
        server.getPlayerList().broadcastAll(new ClientboundPlayerInfoRemovePacket(ids));
        OPPONENTS.clear();
    }

    /** The inputs for the next tick, one per opponent, in the model's action space. */
    public static void setActions(JsonArray actions) {
        for (int i = 0; i < actions.size() && i < OPPONENTS.size(); i++) {
            JsonArray a = actions.get(i).getAsJsonArray();
            int[] v = OPPONENTS.get(i).action;
            for (int k = 0; k < v.length && k < a.size(); k++) v[k] = a.get(k).getAsInt();
        }
    }

    /** Each opponent: keys through the model's limits, turn, a left click as a 26.3 client sees it, then moving. */
    private static void drive(MinecraftServer server) {
        ServerPlayer target = null;
        for (ServerPlayer p : server.getPlayerList().getPlayers()) {
            if (!(p instanceof FakePlayer) && p.gameMode() == GameType.SURVIVAL) target = p;
        }
        for (Opponent o : OPPONENTS) {
            Bot p = o.player;
            if (!p.isAlive() || p.getHealth() <= 0) continue;
            int[] a = o.action;
            boolean[] wanted = {a[0] == 1, a[0] == 2, a[1] == 1, a[1] == 2, a[2] != 0, a[4] != 0, a[5] != 0};
            for (int k = 0; k < 7; k++) {
                o.keyAge[k]++;
                if (wanted[k] != o.keys[k] && o.keyAge[k] >= 2) {
                    o.keys[k] = wanted[k];
                    o.keyAge[k] = 0;
                }
            }
            boolean fwd = o.keys[0], back = o.keys[1], left = o.keys[2], right = o.keys[3], sneak = o.keys[4], jump = o.keys[5], sprint = o.keys[6];
            p.setYRot(p.getYRot() + (float) TURN_STEPS[a[6]]);
            p.setYHeadRot(p.getYRot());
            p.setXRot(Math.max(-90.0f, Math.min(90.0f, p.getXRot() + (float) TURN_STEPS[a[7]])));
            State s = of(p);
            if (a[8] != 0 && target != null) {
                Vec3 eye = p.getEyePosition();
                Vec3 view = p.getViewVector(1.0f);
                BlockHitResult block = p.level().clip(new ClipContext(eye, eye.add(view.scale(BLOCK_REACH)), ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, p));
                double blockD2 = block.getType() == HitResult.Type.MISS ? Double.MAX_VALUE : block.getLocation().distanceToSqr(eye);
                double max = blockD2 == Double.MAX_VALUE ? BLOCK_REACH : Math.sqrt(blockD2);
                AABB box = target.getBoundingBox();
                var clip = box.contains(eye) ? java.util.Optional.of(eye) : box.clip(eye, eye.add(view.scale(max)));
                // A click on nothing would lock attacks, but a click is not a held button: 26.3 clears it the same tick.
                if (clip.isPresent() && clip.get().distanceToSqr(eye) < blockD2 && clip.get().distanceToSqr(eye) < ENTITY_REACH * ENTITY_REACH) {
                    attack(p, target);
                    // Its client's side of the click (26.3 Player.attack, combat.rs recoil): knockback slows it and stops its sprint.
                    if (s.kb + (o.sprinting ? 1 : 0) > 0) {
                        p.setDeltaMovement(p.getDeltaMovement().multiply(0.6, 1.0, 0.6));
                        o.sprinting = false;
                    }
                }
                p.swing(net.minecraft.world.InteractionHand.MAIN_HAND, p.getMainHandItem().getAttackAnimation(), true); // watchers see the arm swing
            }
            float strafe = left == right ? 0 : (left ? 1 : -1);
            float forward = fwd == back ? 0 : (fwd ? 1 : -1);
            float len = (float) Math.sqrt(strafe * strafe + forward * forward);
            if (len > 1.0e-4f) {
                strafe /= len;
                forward /= len;
            }
            p.xxa = strafe;
            p.zza = forward;
            p.setShiftKeyDown(sneak);
            if (o.sprintTrigger > 0) o.sprintTrigger--;
            boolean hadForward = o.lastForward > 1.0e-5f, hasForward = forward > 1.0e-5f;
            boolean crouching = o.wasShift; // crouching follows the shift key as it was before this tick
            p.crouching = crouching;
            if (o.wasShift || back) o.sprintTrigger = 0;
            if (!o.sprinting && hasForward && !crouching) {
                if (!hadForward) {
                    if (o.sprintTrigger > 0) o.sprinting = true;
                    else o.sprintTrigger = SPRINT_WINDOW;
                }
                if (sprint) o.sprinting = true;
            }
            if (o.sprinting && (!hasForward || p.horizontalCollision && !p.minorHorizontalCollision)) o.sprinting = false;
            o.wasShift = sneak;
            o.lastForward = forward;
            p.setSprinting(o.sprinting);
            p.setJumping(jump);
            p.doTick();
            if (o.sprinting != o.sentSprinting) { // LocalPlayer.sendIsSprintingIfNeeded: only a change reaches the server
                o.sentSprinting = o.sprinting;
                s.serverSprint = o.sprinting;
            }
        }
    }

    /**
     * At the start of each client tick: the model's client sees every opponent where the server has it, without the
     * three ticks of smoothing a client gives other players, so what it aims at and is pushed by is where the simulator
     * has them (combat.rs, env.rs fight_tick).
     */
    public static void snapClientCopies(net.minecraft.client.Minecraft mc) {
        if (mc.level == null) return;
        for (Opponent o : OPPONENTS) {
            Entity e = mc.level.getEntity(o.player.getId());
            if (e == null) continue;
            var smoothing = e.getInterpolation();
            if (smoothing != null) smoothing.cancel();
            e.snapTo(o.player.getX(), o.player.getY(), o.player.getZ(), o.player.getYRot(), o.player.getXRot());
            e.setOldPosAndRot();
        }
    }

    /** What the model sees of its opponents (duel.rs View), for the observation. */
    public static JsonArray players() {
        JsonArray out = new JsonArray();
        for (Opponent o : OPPONENTS) {
            Bot p = o.player;
            State s = of(p);
            JsonObject e = new JsonObject();
            e.add("pos", McBridge.vec(p.getX(), p.getY(), p.getZ()));
            Vec3 v = p.getDeltaMovement();
            e.add("vel", McBridge.vec(v.x, v.y, v.z));
            e.addProperty("yaw", p.getYRot());
            e.addProperty("pitch", p.getXRot());
            e.addProperty("health", p.getHealth());
            e.addProperty("armor", s.armor);
            e.addProperty("damage", p.getInventory().getSelectedSlot() == 1 ? s.damage : 1.0f);
            e.addProperty("hurt_time", p.hurtTime);
            e.addProperty("on_ground", p.onGround());
            e.addProperty("sprinting", p.isSprinting());
            e.addProperty("crouching", p.isCrouching());
            e.addProperty("since_hit", s.sinceHit);
            e.addProperty("bot", o.bot);
            out.add(e);
        }
        return out;
    }
}
