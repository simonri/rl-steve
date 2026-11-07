package dev.mcbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import dev.mcbridge.mixin.ServerTickRateManagerAccessor;
import java.io.IOException;
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.component.DataComponents;
import net.minecraft.network.protocol.game.ClientboundSetHeldSlotPacket;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.HolderGetter;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.core.registries.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.MutableComponent;
import net.minecraft.network.chat.numbers.BlankFormat;
import net.minecraft.world.Difficulty;
import net.minecraft.world.scores.DisplaySlot;
import net.minecraft.world.scores.Objective;
import net.minecraft.world.scores.ScoreAccess;
import net.minecraft.world.scores.ScoreHolder;
import net.minecraft.world.scores.Scoreboard;
import net.minecraft.world.scores.criteria.ObjectiveCriteria;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.gamerules.GameRules;
import net.minecraft.world.level.storage.LevelData;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.WorldDimensions;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.flat.FlatLevelGeneratorSettings;
import net.minecraft.world.level.levelgen.placement.PlacedFeature;
import net.minecraft.world.level.levelgen.presets.WorldPresets;
import net.minecraft.world.level.levelgen.structure.StructureSet;
import net.minecraft.world.level.levelgen.FlatLevelSource;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Entry point. Active only when MCBRIDGE_PORT is set, so the same jar is harmless in a normal game.
 *
 * Client thread per tick: wait for a Python command, apply it through the real key bindings, run the
 * real client tick, run exactly one server tick, apply the server's packets, reply with the state.
 */
public final class McBridge implements ClientModInitializer {
    static final Logger LOG = LoggerFactory.getLogger("mcbridge");
    private static final String WORLD = "mcbridge_void";
    private static final int CHAT_HISTORY_LINES = 100; // blank lines that push old chat out of view

    private static boolean driven;
    private static volatile boolean renderOff;
    private static final Lockstep lockstep = new Lockstep();

    private final Link link = new Link();
    private int watchPort;
    private MinecraftServer server;
    private boolean worldRequested;
    private boolean sprinting;
    private static volatile boolean stepping; // Python is driving the game tick by tick (Fight18 bots act only then)

    static boolean stepping() { return stepping; }
    private boolean watchPortOpen;

    // Step in progress.
    private int ticksLeft;
    private JsonObject pendingReply; // the command being answered
    private boolean resetting;
    private boolean died; // a death since the last reply (the respawn follows at once)
    private boolean beatenLastTick; // a fight beat it during the last reply's tick (Fight18.takeDowned)

    private int settleTicks;
    private long settleStart;
    private String settleReason = "";
    private JsonObject resetSpec;
    private boolean serverAppliedReset;

    private long tickCount;
    private int blockRadius = 3;
    private int mapRadius; // columns in each direction reported as a height map. 0 leaves it out.
    private static final int MAP_ABOVE = 8, MAP_BELOW = 16; // how far above and below the feet a column is searched
    private boolean sendEntities;
    private long lastCameraMove;
    private volatile JsonArray beam;
    private long lastBeam;
    private static final long BEAM_NANOS = 200_000_000L;
    private static final int BEAM_HEIGHT = 30;
    private boolean testMenu;
    private boolean humanMode; // a person plays the courses on this server and is recorded
    private volatile boolean recording;
    private final java.util.Queue<JsonObject> humanTicks = new java.util.concurrent.ConcurrentLinkedQueue<>();
    private static final int HUMAN_TICKS_KEPT = 4000; // if Python stops asking, the oldest go first
    private static final int HUMAN_RADIUS = 8; // same block cube as the model sees
    private static final int BLOCK_ID_RADIUS = 5;
    private static final int HUMAN_MAP_RADIUS = 16;
    private final java.util.Map<net.minecraft.world.level.block.Block, Character> blockChars = new java.util.concurrent.ConcurrentHashMap<>();
    private volatile String menuRequest;
    private static final int MENU_REST_SLOT = 8;
    private volatile JsonArray lastCamera;
    private List<Component> sidebarLines = List.of(); // server thread only
    private JsonArray clicks = new JsonArray(); // block clicks since the last reply: target, then support

    public static boolean isDriven() { return driven; }

    public static boolean renderOff() { return renderOff; }

    public static boolean isLockstep() { return driven && lockstep.enabled(); }

    @Override
    public void onInitializeClient() {
        String autobridge = System.getenv("MCBRIDGE_AUTOBRIDGE");
        if (autobridge != null) {
            dev.mcbridge.auto.AutoBridge.register(autobridge.equals("on"));
            if (autobridge.equals("on")) AutoBridgeTestPlayer.register();
        }
        String port = System.getenv("MCBRIDGE_PORT");
        if (port == null) return;
        driven = true;
        String watch = System.getenv("MCBRIDGE_WATCH_PORT");
        watchPort = watch == null ? 0 : Integer.parseInt(watch);
        link.start(Integer.parseInt(port));

        ServerLifecycleEvents.SERVER_STARTED.register(this::onServerStarted);
        ServerLifecycleEvents.SERVER_STOPPING.register(s -> lockstep.disable());
        ServerTickEvents.START_SERVER_TICK.register(lockstep::serverStartTick);
        ServerTickEvents.START_SERVER_TICK.register(Fight18::serverTick); // opponents act, 1.8 resistance and healing
        testMenu = System.getenv("MCBRIDGE_TEST_MENU") != null;
        humanMode = System.getenv("MCBRIDGE_HUMAN") != null;
        ServerTickEvents.END_SERVER_TICK.register(s -> {
            if (testMenu) checkMenu(s);
            if (humanMode && recording) recordHuman(s);
            showBeam(s);
            lockstep.serverEndTick();
        });
        ServerPlayConnectionEvents.JOIN.register((handler, sender, s) -> onPlayerJoin(handler.getPlayer(), s));
        // No permission needed: everyone who can reach these local servers is a spectator or the bot.
        CommandRegistrationCallback.EVENT.register((dispatcher, context, selection) ->
                dispatcher.register(Commands.literal("clearchat").executes(c -> {
                    for (ServerPlayer player : c.getSource().getServer().getPlayerList().getPlayers()) {
                        for (int i = 0; i < CHAT_HISTORY_LINES; i++) player.sendSystemMessage(Component.empty());
                    }
                    return 1;
                })));
        UseBlockCallback.EVENT.register((player, world, hand, hit) -> {
            if (world.isClientSide()) {
                BlockPos support = hit.getBlockPos();
                BlockPos target = support.relative(hit.getDirection());
                JsonArray click = new JsonArray();
                click.add(target.getX());
                click.add(target.getY());
                click.add(target.getZ());
                click.add(support.getX());
                click.add(support.getY());
                click.add(support.getZ());
                clicks.add(click);
            }
            return InteractionResult.PASS;
        });
        ClientTickEvents.START_CLIENT_TICK.register(this::onClientTickStart);
        ClientTickEvents.END_CLIENT_TICK.register(this::onClientTickEnd);
    }

    // ---- World ----

    private void createWorld(Minecraft mc) {
        worldRequested = true;
        mc.options.pauseOnLostFocus = false;
        Path dir = mc.gameDirectory.toPath().resolve("saves").resolve(WORLD);
        if (Files.exists(dir)) {
            try (Stream<Path> files = Files.walk(dir)) {
                files.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            } catch (IOException e) {
                LOG.warn("could not delete old world", e);
            }
        }
        LevelSettings settings = new LevelSettings(WORLD, GameType.SURVIVAL,
                new LevelSettings.DifficultySettings(Difficulty.PEACEFUL, false, false), true, WorldDataConfiguration.DEFAULT);
        WorldOptions options = new WorldOptions(1L, false, false);
        mc.createWorldOpenFlows().createFreshLevel(WORLD, settings, options, McBridge::voidWorld, new TitleScreen());
    }

    private static WorldDimensions voidWorld(HolderLookup.Provider access) {
        HolderGetter<Biome> biomes = access.lookupOrThrow(Registries.BIOME);
        HolderGetter<StructureSet> structures = access.lookupOrThrow(Registries.STRUCTURE_SET);
        HolderGetter<PlacedFeature> features = access.lookupOrThrow(Registries.PLACED_FEATURE);
        Holder<Biome> theVoid = biomes.getOrThrow(Biomes.THE_VOID);
        FlatLevelGeneratorSettings flat = FlatLevelGeneratorSettings.getDefault(biomes, structures, features)
                .withBiomeAndLayers(List.of(), Optional.empty(), theVoid);
        return access.lookupOrThrow(Registries.WORLD_PRESET).getOrThrow(WorldPresets.FLAT).value()
                .createWorldDimensions()
                .replaceOverworldGenerator(access, new FlatLevelSource(flat));
    }

    private void onServerStarted(MinecraftServer s) {
        server = s;
        for (ServerLevel level : s.getAllLevels()) level.noSave = true;
        // A void world has no ground: give players somewhere to stand before the first reset.
        ServerLevel overworld = s.overworld();
        BlockPos spawn = new BlockPos(0, 100, 0);
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int x = -1; x <= 1; x++) {
            for (int z = -2; z <= 0; z++) overworld.setBlock(p.set(x, 100, z), Blocks.STONE.defaultBlockState(), 2 | 16);
        }
        overworld.setRespawnData(LevelData.RespawnData.of(Level.OVERWORLD, spawn.above(), 0f, 0f));
        GameRules rules = overworld.getGameRules();
        rules.set(GameRules.RESPAWN_RADIUS, 0, s);
        rules.set(GameRules.ADVANCE_TIME, false, s);
        rules.set(GameRules.ADVANCE_WEATHER, false, s);
        rules.set(GameRules.SPAWN_MOBS, false, s);
        // A death keeps the hotbar, as the simulator does (Player.keep_gear): the respawn gives nothing back, and an
        // empty hotbar after a real fall would be a round the simulator never plays.
        rules.set(GameRules.KEEP_INVENTORY, true, s);
        if (watchPort > 0 && !watchPortOpen) {
            try {
                // Loopback only, and offline auth so a second client on this machine can join to watch.
                s.setUsesAuthentication(false);
                s.getConnection().startTcpServerListener(InetAddress.getLoopbackAddress(), watchPort);
                watchPortOpen = true;
                LOG.info("watchers can join localhost:{}", watchPort);
            } catch (IOException e) {
                LOG.error("could not open watch port {}", watchPort, e);
            }
        }
    }

    private void onPlayerJoin(ServerPlayer player, MinecraftServer s) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getGameProfile().id().equals(player.getUUID())) {
            if (testMenu || humanMode) player.setGameMode(GameType.SPECTATOR); // the bot only hosts here
            return;
        }
        Fight18.introduce(player); // a fight going on: its bots are not in the player list this watcher was sent
        if (humanMode) {
            player.setGameMode(GameType.ADVENTURE); // nothing can be built or broken: no bridging yet
            player.getInventory().clearContent();
            LOG.info("player joined: {}", player.getGameProfile().name());
            return;
        }
        if (testMenu) {
            giveMenu(player);
        } else {
            player.setGameMode(GameType.SPECTATOR);
        }
        ServerPlayer host = s.getPlayerList().getPlayer(mc.getGameProfile().id());
        JsonArray camera = lastCamera;
        if (camera != null) {
            placeWatcher(s.overworld(), player, camera);
        } else if (host != null) {
            player.teleportTo(host.level(), host.getX() + 6, host.getY() + 4, host.getZ() + 3, Set.of(), 90f, 25f, false);
        }
        LOG.info("watcher joined: {}", player.getGameProfile().name());
    }

    private void startLockstep() {
        var tm = server.tickRateManager();
        tm.setFrozen(false);
        tm.requestGameToSprint(1);
        ((ServerTickRateManagerAccessor) tm).mcbridge$setRemainingSprintTicks(Long.MAX_VALUE);
        ((ServerTickRateManagerAccessor) tm).mcbridge$setScheduledCurrentSprintTicks(Long.MAX_VALUE);
        lockstep.enable();
        sprinting = true;
        stepping = true;
    }

    /** Nobody is driving: give the game back its normal clock, so joined watchers stay connected. */
    private void stopLockstep(Minecraft mc) {
        applyAction(mc, new JsonObject());
        renderOff = false;
        ticksLeft = 0;
        resetting = false;
        lockstep.disable();
        stepping = false;
        server.execute(() -> server.tickRateManager().stopSprinting());
        sprinting = false;
    }

    private void wipeBlocks(int[] b) {
        ServerLevel level = server.overworld();
        BlockState air = Blocks.AIR.defaultBlockState();
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int x = b[0]; x <= b[3]; x++) {
            for (int y = b[1]; y <= b[4]; y++) {
                for (int z = b[2]; z <= b[5]; z++) {
                    if (!level.getBlockState(p.set(x, y, z)).isAir()) level.setBlock(p, air, 2 | 16);
                }
            }
        }
    }

    /**
     * A new round's hotbar (AutoEnv._with_hotbar): "hotbar_items" as [registry name, count] per slot and "slot" selected;
     * an older spec without them gets its blocks in the first slot, as before. On the server, and the client is told the slot.
     */
    private static void giveHotbar(ServerPlayer player, JsonObject spec) {
        Inventory inv = player.getInventory();
        inv.clearContent();
        int slot = spec.has("slot") ? spec.get("slot").getAsInt() : 0;
        if (spec.has("hotbar_items")) {
            JsonArray h = spec.getAsJsonArray("hotbar_items");
            for (int i = 0; i < Math.min(9, h.size()); i++) {
                JsonArray e = h.get(i).getAsJsonArray();
                Item item = itemNamed(e.get(0).getAsString());
                int count = e.get(1).getAsInt();
                if (item != null && count > 0) inv.setItem(i, new ItemStack(item, count));
            }
        } else {
            int blocks = spec.has("blocks") ? spec.get("blocks").getAsInt() : 0;
            if (blocks > 0) inv.setItem(0, new ItemStack(Items.STONE, blocks));
        }
        inv.setSelectedSlot(slot);
        player.connection.send(new ClientboundSetHeldSlotPacket(slot));
        player.inventoryMenu.sendAllDataToRemote();
    }

    /** What a course's platforms are built of: "floor", a Bed Wars shop block (AutoEnv._with_hotbar); stone if none. */
    private BlockState floorBlock(JsonObject spec) {
        if (!spec.has("floor")) return Blocks.STONE.defaultBlockState();
        try {
            return BlockStateParser.parseForBlock(server.registryAccess().lookupOrThrow(Registries.BLOCK), spec.get("floor").getAsString(), false).blockState();
        } catch (Exception e) {
            LOG.warn("unknown floor block: {}", spec.get("floor"));
            return Blocks.STONE.defaultBlockState();
        }
    }

    /** Blocks in hand: the count of a block stack, 0 for anything else (a sword, a tool, nothing). */
    private static int blocksInHand(ItemStack s) {
        return s.getItem() instanceof net.minecraft.world.item.BlockItem ? s.getCount() : 0;
    }

    private static java.util.Map<String, Item> itemsByName;

    /** An item by its registry name ("minecraft:white_wool"), or null. */
    private static Item itemNamed(String name) {
        if (itemsByName == null) {
            itemsByName = new java.util.HashMap<>();
            for (Item it : BuiltInRegistries.ITEM) itemsByName.put(BuiltInRegistries.ITEM.getKey(it).toString(), it);
        }
        return itemsByName.get(name);
    }

    private void pasteBlocks(JsonArray palette, JsonArray rows) {
        var lookup = server.registryAccess().lookupOrThrow(Registries.BLOCK);
        BlockState[] states = new BlockState[palette.size()];
        for (int i = 0; i < states.length; i++) {
            String name = palette.get(i).getAsString();
            try {
                states[i] = BlockStateParser.parseForBlock(lookup, name, false).blockState();
            } catch (Exception e) {
                LOG.warn("unknown block state in map: {}", name);
            }
        }
        ServerLevel level = server.overworld();
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (JsonElement e : rows) {
            JsonArray r = e.getAsJsonArray();
            BlockState state = states[r.get(3).getAsInt()];
            if (state != null) level.setBlock(p.set(r.get(0).getAsInt(), r.get(1).getAsInt(), r.get(2).getAsInt()), state, 2 | 16);
        }
    }

    /** On a test server, watchers get two items: selecting one asks for that kind of task. */
    private static void giveMenu(ServerPlayer player) {
        player.setGameMode(GameType.CREATIVE);
        Inventory inv = player.getInventory();
        inv.clearContent();
        ItemStack straight = new ItemStack(Items.EMERALD);
        straight.set(DataComponents.CUSTOM_NAME, Component.literal("Straight bridge"));
        ItemStack diagonal = new ItemStack(Items.DIAMOND);
        diagonal.set(DataComponents.CUSTOM_NAME, Component.literal("Diagonal bridge"));
        inv.setItem(0, straight);
        inv.setItem(1, diagonal);
        selectEmptySlot(player);
    }

    private static void selectEmptySlot(ServerPlayer player) {
        player.getInventory().setSelectedSlot(MENU_REST_SLOT);
        player.connection.send(new ClientboundSetHeldSlotPacket(MENU_REST_SLOT));
    }

    private void checkMenu(MinecraftServer s) {
        var host = Minecraft.getInstance().getGameProfile().id();
        for (ServerPlayer player : s.getPlayerList().getPlayers()) {
            if (player.getUUID().equals(host)) continue;
            int slot = player.getInventory().getSelectedSlot();
            if (slot == 0 || slot == 1) {
                menuRequest = slot == 0 ? "straight" : "diagonal";
                selectEmptySlot(player);
            }
        }
    }

    /** The person playing on this server: everyone but the bot that hosts it. */
    private static ServerPlayer humanOn(MinecraftServer s) {
        var host = Minecraft.getInstance().getGameProfile().id();
        for (ServerPlayer player : s.getPlayerList().getPlayers()) {
            if (!player.getUUID().equals(host)) return player;
        }
        return null;
    }

    /** One tick of what the person did and saw, as the server knows it. */
    private void recordHuman(MinecraftServer s) {
        ServerPlayer player = humanOn(s);
        if (player == null) return;
        JsonObject t = new JsonObject();
        t.addProperty("tick", s.getTickCount());
        t.add("pos", vec(player.getX(), player.getY(), player.getZ()));
        t.addProperty("yaw", player.getYRot());
        t.addProperty("pitch", player.getXRot());
        t.addProperty("on_ground", player.onGround());
        t.addProperty("health", player.getHealth());
        t.addProperty("held_count", blocksInHand(player.getMainHandItem()));
        addHit(t, player.pick(player.blockInteractionRange(), 1.0f, false));
        var input = player.getLastClientInput();
        t.addProperty("forward", input.forward());
        t.addProperty("back", input.backward());
        t.addProperty("left", input.left());
        t.addProperty("right", input.right());
        t.addProperty("jump", input.jump());
        t.addProperty("sneak", input.shift());
        t.addProperty("sprint", input.sprint());
        t.addProperty("sprinting", player.isSprinting());
        t.addProperty("collision", player.horizontalCollision);
        t.addProperty("fall_distance", player.fallDistance);
        JsonArray effects = new JsonArray();
        player.getActiveEffects().forEach(e -> {
            JsonArray one = new JsonArray();
            one.add(e.getEffect().getRegisteredName());
            one.add(e.getAmplifier());
            effects.add(one);
        });
        t.add("effects", effects);
        t.addProperty("ids", blockIdsAround(player.level(), player.blockPosition(), BLOCK_ID_RADIUS));
        BlockPos feet = player.blockPosition();
        t.add("feet", vec(feet.getX(), feet.getY(), feet.getZ()));
        t.addProperty("solid", solidAround(player.level(), feet, HUMAN_RADIUS));
        t.addProperty("surface", surfaceAround(player.level(), feet, HUMAN_MAP_RADIUS));
        t.addProperty("hurt_time", player.hurtTime);
        humanTicks.add(t);
        while (humanTicks.size() > HUMAN_TICKS_KEPT) humanTicks.poll();
    }

    private void buildForHuman(JsonObject cmd) {
        ServerLevel level = server.overworld();
        if (cmd.has("wipe")) wipeBlocks(ints(cmd.getAsJsonArray("wipe")));
        wipeBlocks(ints(cmd.getAsJsonArray("clear")));
        BlockState stone = floorBlock(cmd);
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (JsonElement e : cmd.getAsJsonArray("platforms")) {
            int[] b = ints(e.getAsJsonArray());
            for (int x = b[0]; x <= b[3]; x++) {
                for (int y = b[1]; y <= b[4]; y++) {
                    for (int z = b[2]; z <= b[5]; z++) level.setBlock(p.set(x, y, z), stone, 2 | 16);
                }
            }
        }
        int[] goal = ints(cmd.getAsJsonArray("goal"));
        level.setBlock(p.set(goal[0], goal[1], goal[2]), Blocks.REDSTONE_BLOCK.defaultBlockState(), 2 | 16);
        ServerPlayer player = humanOn(server);
        if (player != null) {
            // Blocks to place only on courses that need them: then it has to be survival, otherwise nothing can be changed.
            int blocks = cmd.has("blocks") ? cmd.get("blocks").getAsInt() : 0;
            player.setGameMode(blocks > 0 ? GameType.SURVIVAL : GameType.ADVENTURE);
            giveHotbar(player, cmd);
        }
        moveHuman(cmd);
    }

    private void moveHuman(JsonObject cmd) {
        ServerPlayer player = humanOn(server);
        if (player == null) return;
        JsonArray at = cmd.getAsJsonArray("pos");
        player.setDeltaMovement(Vec3.ZERO);
        player.fallDistance = 0;
        player.setHealth(player.getMaxHealth());
        player.teleportTo(server.overworld(), at.get(0).getAsDouble(), at.get(1).getAsDouble(), at.get(2).getAsDouble(),
                Set.of(), cmd.get("yaw").getAsFloat(), cmd.get("pitch").getAsFloat(), false);
    }

    /** Commands for a server with a person on it. These run on the game's normal clock, never in lockstep. */
    private boolean handleHuman(JsonObject cmd, String op) {
        switch (op) {
            case "human_ticks" -> {
                JsonObject o = new JsonObject();
                JsonArray ticks = new JsonArray();
                JsonObject t;
                while ((t = humanTicks.poll()) != null) ticks.add(t);
                o.add("ticks", ticks);
                JsonObject names = new JsonObject();
                blockChars.forEach((block, c) -> names.addProperty(String.valueOf(c), BuiltInRegistries.BLOCK.getKey(block).toString()));
                o.add("block_names", names);
                ServerPlayer player = humanOn(server);
                if (player != null) o.addProperty("player", player.getGameProfile().name());
                tag(o, cmd);
                link.reply(o);
            }
            case "human_record" -> {
                recording = cmd.get("on").getAsBoolean();
                ok(cmd);
            }
            case "human_build" -> {
                server.execute(() -> buildForHuman(cmd));
                ok(cmd);
            }
            case "human_wipe" -> {
                int[] box = ints(cmd.getAsJsonArray("box"));
                server.execute(() -> wipeBlocks(box));
                ok(cmd);
            }
            case "human_paste" -> {
                JsonArray palette = cmd.getAsJsonArray("palette");
                JsonArray rows = cmd.getAsJsonArray("blocks");
                server.execute(() -> pasteBlocks(palette, rows));
                ok(cmd);
            }
            case "human_move" -> {
                server.execute(() -> moveHuman(cmd));
                ok(cmd);
            }
            case "human_block" -> {
                int[] at = ints(cmd.getAsJsonArray("at"));
                JsonObject o = new JsonObject();
                o.addProperty("name", BuiltInRegistries.BLOCK.getKey(server.overworld().getBlockState(new BlockPos(at[0], at[1], at[2])).getBlock()).toString());
                tag(o, cmd);
                link.reply(o);
            }
            case "human_board" -> {
                Component title = styled(cmd.getAsJsonArray("title"));
                List<Component> lines = new ArrayList<>();
                for (JsonElement e : cmd.getAsJsonArray("lines")) lines.add(styled(e.getAsJsonArray()));
                server.execute(() -> showSidebar(title, lines));
                ok(cmd);
            }
            case "human_say" -> {
                Component text = styled(cmd.getAsJsonArray("text"));
                server.execute(() -> {
                    ServerPlayer player = humanOn(server);
                    if (player != null) player.sendSystemMessage(text);
                });
                ok(cmd);
            }
            default -> {
                return false;
            }
        }
        return true;
    }

    /** A column of light over the goal. Particles go to watchers only, so the bot's game never gets them. */
    private void showBeam(MinecraftServer s) {
        JsonArray at = beam;
        long now = System.nanoTime();
        if (at == null || now - lastBeam < BEAM_NANOS) return;
        lastBeam = now;
        var host = Minecraft.getInstance().getGameProfile().id();
        double x = at.get(0).getAsDouble(), y = at.get(1).getAsDouble(), z = at.get(2).getAsDouble();
        for (ServerPlayer watcher : s.getPlayerList().getPlayers()) {
            if (watcher.getUUID().equals(host)) continue;
            for (int i = 0; i < BEAM_HEIGHT; i += 2) {
                s.overworld().sendParticles(watcher, ParticleTypes.END_ROD, true, true, x, y + i, z, 1, 0.05, 0.3, 0.05, 0.0);
            }
        }
    }

    /** Builds text from parts like ["Test: ", "gray"], with "bold" as an optional third value. */
    private static Component styled(JsonArray parts) {
        MutableComponent out = Component.empty();
        for (JsonElement e : parts) {
            JsonArray part = e.getAsJsonArray();
            MutableComponent piece = Component.literal(part.get(0).getAsString());
            piece.withStyle(ChatFormatting.valueOf(part.get(1).getAsString().toUpperCase(java.util.Locale.ROOT)));
            if (part.size() > 2) piece.withStyle(ChatFormatting.BOLD);
            out.append(piece);
        }
        return out;
    }

    /** Everyone on the server sees this scoreboard. Only changed lines are sent. */
    private void showSidebar(Component title, List<Component> lines) {
        Scoreboard board = server.getScoreboard();
        Objective objective = board.getObjective("mcbridge");
        if (objective == null) {
            objective = board.addObjective("mcbridge", ObjectiveCriteria.DUMMY, title,
                    ObjectiveCriteria.RenderType.INTEGER, true, BlankFormat.INSTANCE);
            board.setDisplayObjective(DisplaySlot.SIDEBAR, objective);
        } else if (!title.equals(objective.getDisplayName())) {
            objective.setDisplayName(title);
        }
        for (int i = lines.size(); i < sidebarLines.size(); i++) {
            board.resetSinglePlayerScore(ScoreHolder.forNameOnly("line" + i), objective);
        }
        for (int i = 0; i < lines.size(); i++) {
            // Unchanged only if the text and its place are: a score is its row, and it moves when the row count does.
            if (lines.size() == sidebarLines.size() && lines.get(i).equals(sidebarLines.get(i))) continue;
            ScoreAccess score = board.getOrCreatePlayerScore(ScoreHolder.forNameOnly("line" + i), objective);
            score.set(lines.size() - i);
            score.display(lines.get(i));
        }
        sidebarLines = lines;
    }

    // ---- Tick loop ----

    private void onClientTickStart(Minecraft mc) {
        if (mc.level == null) {
            if (!worldRequested && mc.gui.screen() instanceof TitleScreen) createWorld(mc);
            return;
        }
        if (mc.player == null || server == null) return;
        Fight18.snapClientCopies(mc);
        if (mc.player.isDeadOrDying()) {
            if (!resetting) died = true;
            mc.player.respawn();
            // Between commands nothing else steps the server, and the respawn only happens on a server tick: without
            // this step it never comes, no command is read again, and the game hangs for good.
            if (lockstep.enabled() && ticksLeft <= 0 && !resetting) {
                lockstep.stepServer();
                ((dev.mcbridge.mixin.EventLoopInvoker) (Object) mc).mcbridge$runAllTasks();
            }
            return;
        }
        if (mc.gui.screen() != null) {
            mc.gui.setScreen(null);
            return;
        }
        if (ticksLeft > 0 || resetting) return; // still working through the current command
        readCommands(mc);
    }

    private void readCommands(Minecraft mc) {
        try {
            while (true) {
                JsonObject cmd = link.poll(50);
                if (cmd == null) {
                    ((dev.mcbridge.mixin.MinecraftInvoker) mc).mcbridge$sdl().pumpEvents(); // keeps the window responsive while Python is busy
                    if (mc.options != null) mc.options.pauseOnLostFocus = false;
                    if (!lockstep.enabled()) return; // nobody driving: let the game run
                    continue;
                }
                boolean waits;
                try {
                    waits = handle(mc, cmd);
                } catch (RuntimeException e) { // a malformed command is answered, not a crashed game
                    LOG.error("command failed: {}", cmd, e);
                    ticksLeft = 0;
                    pendingReply = null;
                    JsonObject o = new JsonObject();
                    o.addProperty("error", cmd.has("op") ? cmd.get("op").getAsString() + ": " + e : String.valueOf(e));
                    tag(o, cmd);
                    link.reply(o);
                    continue;
                }
                if (waits) return;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** @return true when the command needs game ticks to complete. */
    private boolean handle(Minecraft mc, JsonObject cmd) {
        String op = cmd.get("op").getAsString();
        if (op.equals("disconnected")) {
            stopLockstep(mc);
            return false;
        }
        if (op.equals("watchers")) {
            // Answered without touching the game's clock, so asking never starts stepping.
            JsonObject o = new JsonObject();
            o.addProperty("watchers", Math.max(0, server.getPlayerCount() - 1));
            JsonArray where = new JsonArray();
            for (ServerPlayer p : server.getPlayerList().getPlayers()) {
                if (p.getUUID().equals(mc.getGameProfile().id())) continue;
                JsonArray one = vec(p.getX(), p.getY(), p.getZ());
                one.add(p.getYRot());
                one.add(p.getXRot());
                where.add(one);
            }
            o.add("positions", where);
            tag(o, cmd);
            link.reply(o);
            return false;
        }
        if (op.equals("menu")) {
            // What a watcher asked for on a test server, if anything. Asking does not touch the game's clock.
            JsonObject o = new JsonObject();
            String wanted = menuRequest;
            menuRequest = null;
            if (wanted != null) o.addProperty("request", wanted);
            tag(o, cmd);
            link.reply(o);
            return false;
        }
        if (op.equals("release")) {
            // Stay connected but let the game run on its own clock, with no drawing, until needed.
            if (sprinting) stopLockstep(mc);
            renderOff = true;
            ok(cmd);
            return false;
        }
        if (humanMode && handleHuman(cmd, op)) return false;
        if (!sprinting) startLockstep();
        switch (op) {
            case "step" -> {
                JsonObject act = cmd.getAsJsonObject("action");
                if (act != null && act.has("opponents")) {
                    JsonArray actions = act.getAsJsonArray("opponents");
                    lockstep.onServer(() -> Fight18.setActions(actions));
                }
                applyAction(mc, act);
                ticksLeft = cmd.has("ticks") ? Math.max(1, cmd.get("ticks").getAsInt()) : 1;
                pendingReply = cmd;
                return true;
            }
            case "reset" -> {
                applyAction(mc, new JsonObject());
                resetSpec = cmd;
                died = false;
                Fight18.takeDowned(); // a beating from the last round is not this round's
                beatenLastTick = false;
                if (cmd.has("beam")) beam = cmd.getAsJsonArray("beam");
                resetting = true;
                settleTicks = 0;
                settleStart = System.nanoTime();
                serverAppliedReset = false;
                pendingReply = cmd;
                if (cmd.has("block_radius")) blockRadius = cmd.get("block_radius").getAsInt();
                if (cmd.has("map_radius")) mapRadius = cmd.get("map_radius").getAsInt();
                if (cmd.has("entities")) sendEntities = cmd.get("entities").getAsBoolean();
                JsonObject spec = cmd;
                lockstep.onServer(() -> applyResetOnServer(spec));
                return true;
            }
            case "render" -> {
                renderOff = !cmd.get("on").getAsBoolean();
                ok(cmd);
                return false;
            }
            case "paste" -> {
                // Blocks from a saved map, as a palette of block states and [x, y, z, palette index] rows.
                JsonArray palette = cmd.getAsJsonArray("palette");
                JsonArray rows = cmd.getAsJsonArray("blocks");
                lockstep.onServer(() -> pasteBlocks(palette, rows));
                ok(cmd);
                return false;
            }
            case "wipe" -> {
                int[] box = ints(cmd.getAsJsonArray("box"));
                lockstep.onServer(() -> wipeBlocks(box));
                ok(cmd);
                return false;
            }
            case "watch_at" -> {
                JsonArray at = cmd.getAsJsonArray("camera");
                lockstep.onServer(() -> {
                    var hostId = Minecraft.getInstance().getGameProfile().id();
                    for (ServerPlayer watcher : server.getPlayerList().getPlayers()) {
                        if (!watcher.getUUID().equals(hostId)) placeWatcher(server.overworld(), watcher, at);
                    }
                });
                lastCamera = at;
                ok(cmd);
                return false;
            }
            case "gear" -> {
                // The player's gear and the rules for this round (Fight18): the sword's damage, Bedwars armor, Knockback,
                // and the slot selected (the hotbar itself came with the reset).
                float damage = cmd.get("damage").getAsFloat();
                int armor = cmd.get("armor").getAsInt(), kb = cmd.get("kb").getAsInt(), slot = cmd.get("slot").getAsInt();
                boolean rules18 = cmd.get("rules18").getAsBoolean();
                var id = mc.getGameProfile().id();
                lockstep.onServer(() -> {
                    ServerPlayer p = server.getPlayerList().getPlayer(id);
                    if (p == null) return;
                    Fight18.gear(p, damage, armor, kb, rules18);
                    // The server holds the item too (watchers see it, and the hit's damage follows it), and tells the
                    // client, which otherwise keeps its own idea.
                    p.getInventory().setSelectedSlot(slot);
                    p.connection.send(new net.minecraft.network.protocol.game.ClientboundSetHeldSlotPacket(slot));
                });
                mc.player.getInventory().setSelectedSlot(slot);
                ok(cmd);
                return false;
            }
            case "opponent" -> {
                // A scripted opponent (Fight18.Bot), only on the server; Python sends its inputs with every step.
                lockstep.onServer(() -> Fight18.spawn(server, cmd));
                ok(cmd);
                return false;
            }
            case "push" -> {
                // A hit's knockback, as ClientboundSetEntityMotionPacket and the hurt event give it: new motion before
                // the next tick, and the red flash. Python decides it (clutch.rs), so it can always be saved.
                JsonArray v = cmd.getAsJsonArray("vel");
                mc.player.setDeltaMovement(v.get(0).getAsDouble(), v.get(1).getAsDouble(), v.get(2).getAsDouble());
                mc.player.animateHurt(0.0F);
                ok(cmd);
                return false;
            }
            case "sidebar" -> {
                Component title = styled(cmd.getAsJsonArray("title"));
                List<Component> lines = new ArrayList<>();
                for (JsonElement e : cmd.getAsJsonArray("lines")) lines.add(styled(e.getAsJsonArray()));
                lockstep.onServer(() -> showSidebar(title, lines));
                ok(cmd);
                return false;
            }
            case "state" -> {
                JsonObject o = observation(mc);
                tag(o, cmd);
                link.reply(o);
                return false;
            }
            case "blocks" -> {
                JsonObject o = blockNames(mc, cmd.getAsJsonArray("min"), cmd.getAsJsonArray("max"));
                tag(o, cmd);
                link.reply(o);
                return false;
            }
            case "quit" -> {
                ok(cmd);
                mc.stop();
                return false;
            }
            default -> {
                JsonObject o = new JsonObject();
                o.addProperty("error", "unknown op " + op);
                tag(o, cmd);
                link.reply(o);
                return false;
            }
        }
    }

    private void onClientTickEnd(Minecraft mc) {
        if (!lockstep.enabled() || (ticksLeft <= 0 && !resetting)) return;
        lockstep.stepServer();
        ((dev.mcbridge.mixin.EventLoopInvoker) (Object) mc).mcbridge$runAllTasks(); // apply packets the server just sent
        tickCount++;

        if (resetting) {
            settleTicks++;
            if (settleTicks % 200 == 0) LOG.info("reset still settling after {} ticks: {}", settleTicks, settleReason); // a slow one shows why
            if (settleTicks % 20 == 0 && serverAppliedReset) {
                // A click from just before the reset can reach the server after it, and bring back a
                // block or change the item count. Apply the reset again until both sides agree.
                JsonObject again = resetSpec;
                lockstep.onServer(() -> applyResetOnServer(again));
            }
            if (serverAppliedReset && resetSettled(mc)) {
                resetting = false;
                sendObservation(mc);
            } else if (System.nanoTime() - settleStart > 60_000_000_000L) {
                resetting = false;
                JsonObject o = new JsonObject();
                o.addProperty("error", "reset did not settle: " + settleReason);
                tag(o, pendingReply);
                link.reply(o);
            }
            return;
        }
        if (--ticksLeft == 0) sendObservation(mc);
    }

    private void sendObservation(Minecraft mc) {
        JsonObject o = observation(mc);
        tag(o, pendingReply);
        link.reply(o);
    }

    private void ok(JsonObject cmd) {
        JsonObject o = new JsonObject();
        o.addProperty("ok", true);
        tag(o, cmd);
        link.reply(o);
    }

    private static void tag(JsonObject reply, JsonObject cmd) {
        if (cmd != null && cmd.has("id")) reply.add("id", cmd.get("id"));
    }

    // ---- Input: same key bindings and turn method that a human's keyboard and mouse use ----

    private static boolean flag(JsonObject a, String k) {
        return a.has(k) && a.get(k).getAsBoolean();
    }

    private static void applyAction(Minecraft mc, JsonObject a) {
        var o = mc.options;
        o.keyUp.setDown(flag(a, "forward"));
        o.keyDown.setDown(flag(a, "back"));
        o.keyLeft.setDown(flag(a, "left"));
        o.keyRight.setDown(flag(a, "right"));
        o.keyJump.setDown(flag(a, "jump"));
        o.keyShift.setDown(flag(a, "sneak"));
        o.keySprint.setDown(flag(a, "sprint"));
        if (flag(a, "use")) net.minecraft.client.KeyMapping.click(o.keyUse.getDefaultKey());
        if (flag(a, "attack")) net.minecraft.client.KeyMapping.click(o.keyAttack.getDefaultKey());
        if (a.has("hotbar")) mc.player.getInventory().setSelectedSlot(a.get("hotbar").getAsInt());
        double dYaw = a.has("yaw") ? a.get("yaw").getAsDouble() : 0;
        double dPitch = a.has("pitch") ? a.get("pitch").getAsDouble() : 0;
        if (dYaw != 0 || dPitch != 0) mc.player.turn(dYaw / 0.15, dPitch / 0.15);
    }

    // ---- Episode reset ----

    private void applyResetOnServer(JsonObject spec) {
        Minecraft mc = Minecraft.getInstance();
        ServerLevel level = server.overworld();
        ServerPlayer player = server.getPlayerList().getPlayer(mc.getGameProfile().id());
        if (player == null) return;

        if (!spec.has("keep_arena")) Fight18.clear(server); // a new round: last round's opponents go
        if (spec.has("keep_arena")) {
            // A respawn: the player goes back to the start, and everything built stays where it is.
            JsonArray at = spec.getAsJsonArray("pos");
            // A block placed where the player spawns would push them away from the start, so clear the spot.
            double hx = at.get(0).getAsDouble(), hy = at.get(1).getAsDouble(), hz = at.get(2).getAsDouble();
            BlockPos.MutableBlockPos spot = new BlockPos.MutableBlockPos();
            for (int x = (int) Math.floor(hx - 0.31); x <= (int) Math.floor(hx + 0.31); x++) {
                for (int z = (int) Math.floor(hz - 0.31); z <= (int) Math.floor(hz + 0.31); z++) {
                    for (int y = (int) Math.floor(hy); y <= (int) Math.floor(hy + 1.81); y++) {
                        spot.set(x, y, z);
                        if (!level.getBlockState(spot).isAir()) level.setBlock(spot, Blocks.AIR.defaultBlockState(), 2 | 16);
                    }
                }
            }
            player.setDeltaMovement(Vec3.ZERO);
            player.fallDistance = 0;
            player.setHealth(player.getMaxHealth());
            player.teleportTo(level, at.get(0).getAsDouble(), at.get(1).getAsDouble(), at.get(2).getAsDouble(),
                    Set.of(), spec.get("yaw").getAsFloat(), spec.get("pitch").getAsFloat(), false);
            serverAppliedReset = true;
            return;
        }
        wipeBlocks(ints(spec.getAsJsonArray("clear")));
        BlockState stone = floorBlock(spec);
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (JsonElement e : spec.getAsJsonArray("platforms")) {
            int[] b = ints(e.getAsJsonArray());
            for (int x = b[0]; x <= b[3]; x++) {
                for (int y = b[1]; y <= b[4]; y++) {
                    for (int z = b[2]; z <= b[5]; z++) level.setBlock(p.set(x, y, z), stone, 2 | 16);
                }
            }
        }
        giveHotbar(player, spec);
        player.setDeltaMovement(Vec3.ZERO);
        player.fallDistance = 0;
        player.setHealth(player.getMaxHealth());
        moveWatchers(level, player, spec);
        JsonArray pos = spec.getAsJsonArray("pos");
        player.teleportTo(level, pos.get(0).getAsDouble(), pos.get(1).getAsDouble(), pos.get(2).getAsDouble(),
                Set.of(), spec.get("yaw").getAsFloat(), spec.get("pitch").getAsFloat(), false);
        serverAppliedReset = true;
    }

    /** Places watchers once per episode. A spec can ask for a minimum time between moves. */
    private void moveWatchers(ServerLevel level, ServerPlayer host, JsonObject spec) {
        if (!spec.has("camera") || spec.has("camera_done")) return;
        spec.addProperty("camera_done", true);
        JsonArray at = spec.getAsJsonArray("camera");
        lastCamera = at;
        long now = System.nanoTime();
        double every = spec.has("camera_every") ? spec.get("camera_every").getAsDouble() : 0.0;
        if (now - lastCameraMove < every * 1e9) return;
        lastCameraMove = now;
        for (ServerPlayer watcher : server.getPlayerList().getPlayers()) {
            if (watcher != host) placeWatcher(level, watcher, at);
        }
    }

    private static void placeWatcher(ServerLevel level, ServerPlayer watcher, JsonArray at) {
        watcher.teleportTo(level, at.get(0).getAsDouble(), at.get(1).getAsDouble(), at.get(2).getAsDouble(), Set.of(),
                at.get(3).getAsFloat(), at.get(4).getAsFloat(), false);
    }

    private boolean resetSettled(Minecraft mc) {
        LocalPlayer pl = mc.player;
        JsonArray pos = resetSpec.getAsJsonArray("pos");
        settleReason = "position " + pl.position() + " on_ground=" + pl.onGround();
        if (Math.abs(pl.getX() - pos.get(0).getAsDouble()) > 1e-3) return false;
        if (Math.abs(pl.getZ() - pos.get(2).getAsDouble()) > 1e-3) return false;
        if (Math.abs(pl.getY() - pos.get(1).getAsDouble()) > 1e-3 || !pl.onGround()) return false;
        if (resetSpec.has("keep_arena")) {
            settleReason = "velocity " + pl.getDeltaMovement();
            return pl.getDeltaMovement().horizontalDistanceSqr() <= 1e-6;
        }
        if (resetSpec.has("hotbar_items")) {
            // The client shows the hotbar the round gives (giveHotbar): every slot's count, and the slot selected.
            JsonArray h = resetSpec.getAsJsonArray("hotbar_items");
            for (int i = 0; i < Math.min(9, h.size()); i++) {
                JsonArray e = h.get(i).getAsJsonArray();
                int want = e.get(1).getAsInt();
                ItemStack have = pl.getInventory().getItem(i);
                settleReason = "slot " + i + " count " + have.getCount() + ", want " + want;
                if (have.getCount() != want) return false;
                // The same item too: last round's 16 wool would pass a count check for this round's 16 end stone.
                Item item = itemNamed(e.get(0).getAsString());
                settleReason = "slot " + i + " item " + have.getItem() + ", want " + e.get(0).getAsString();
                if (want > 0 && item != null && !have.is(item)) return false;
            }
            int slot = resetSpec.has("slot") ? resetSpec.get("slot").getAsInt() : 0;
            settleReason = "selected " + pl.getInventory().getSelectedSlot() + ", want " + slot;
            if (pl.getInventory().getSelectedSlot() != slot) return false;
        } else {
            settleReason = "count " + pl.getInventory().getItem(0).getCount();
            if (pl.getInventory().getItem(0).getCount() != resetSpec.get("blocks").getAsInt()) return false;
        }
        settleReason = "health " + pl.getHealth();
        if (pl.getHealth() < pl.getMaxHealth()) return false;
        settleReason = "velocity " + pl.getDeltaMovement();
        if (pl.getDeltaMovement().horizontalDistanceSqr() > 1e-6) return false;
        if (resetSpec.has("skip_block_check")) return true; // a pasted map is checked by the caller instead
        // The client has to show the same arena as the server.
        int[] c = ints(resetSpec.getAsJsonArray("clear"));
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        List<int[]> platforms = new ArrayList<>();
        for (JsonElement e : resetSpec.getAsJsonArray("platforms")) platforms.add(ints(e.getAsJsonArray()));
        for (int x = c[0]; x <= c[3]; x++) {
            for (int y = c[1]; y <= c[4]; y++) {
                for (int z = c[2]; z <= c[5]; z++) {
                    boolean want = false;
                    for (int[] b : platforms) {
                        if (x >= b[0] && x <= b[3] && y >= b[1] && y <= b[4] && z >= b[2] && z <= b[5]) want = true;
                    }
                    if (mc.level.getBlockState(p.set(x, y, z)).isAir() == want) {
                        settleReason = "block " + x + "," + y + "," + z + " want solid=" + want;
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static int[] ints(JsonArray a) {
        int[] out = new int[a.size()];
        for (int i = 0; i < out.length; i++) out[i] = a.get(i).getAsInt();
        return out;
    }

    // ---- State: read from the real client objects ----

    static JsonArray vec(double x, double y, double z) {
        JsonArray a = new JsonArray();
        a.add(x);
        a.add(y);
        a.add(z);
        return a;
    }

    private JsonObject observation(Minecraft mc) {
        LocalPlayer pl = mc.player;
        JsonObject o = new JsonObject();
        o.addProperty("tick", tickCount);
        o.add("pos", vec(pl.getX(), pl.getY(), pl.getZ()));
        Vec3 v = pl.getDeltaMovement();
        o.add("vel", vec(v.x, v.y, v.z));
        o.addProperty("yaw", pl.getYRot());
        o.addProperty("pitch", pl.getXRot());
        o.addProperty("on_ground", pl.onGround());
        o.addProperty("sneaking", pl.isShiftKeyDown());
        o.addProperty("sprinting", pl.isSprinting());
        o.addProperty("collision", pl.horizontalCollision);
        o.addProperty("fall_distance", pl.fallDistance);
        o.addProperty("eye_height", pl.getEyeHeight());
        o.addProperty("held_count", blocksInHand(pl.getMainHandItem())); // blocks in hand, 0 with a sword (Player.held)
        o.addProperty("slot", pl.getInventory().getSelectedSlot());
        JsonArray hotbar = new JsonArray(); // each slot as the player sees it: registry name and count (items.py kind_of)
        for (int i = 0; i < 9; i++) {
            ItemStack st = pl.getInventory().getItem(i);
            JsonArray slot = new JsonArray();
            slot.add(st.isEmpty() ? "minecraft:air" : BuiltInRegistries.ITEM.getKey(st.getItem()).toString());
            slot.add(st.isEmpty() ? 0 : st.getCount());
            hotbar.add(slot);
        }
        o.add("hotbar", hotbar);
        o.addProperty("crouch_pose", pl.isCrouching());
        ServerPlayer sp = server == null ? null : server.getPlayerList().getPlayer(mc.getGameProfile().id());
        if (sp != null) {
            Fight18.State s = Fight18.of(sp);
            o.addProperty("resist", s.resist);
            o.addProperty("attack_damage", s.damage);
            o.addProperty("armor", s.armor);
            o.addProperty("server_sprinting", sp.isSprinting());
            o.addProperty("since_hit", s.sinceHit);
            o.addProperty("since_hurt", s.sinceHurt);
        }
        o.add("players", Fight18.players());
        boolean st = Fight18.takeStruck();
        if (st) o.addProperty("struck", true); // its left click was on an opponent (a click that is not costs)
        double[] hit = Fight18.takeLanded();
        if (hit != null) { // its own hit that counted: feet distance across, critical (AutoEnv's long and critical hit points)
            JsonArray h = new JsonArray();
            h.add(hit[0]);
            h.add(hit[1] != 0);
            h.add(hit[2] != 0); // with a sword in hand
            o.add("landed", h);
        }
        o.addProperty("health", pl.getHealth());
        // It died since the last reply and is already back at the spawn (a fall), or a fight beat it without killing
        // it (Fight18): either way the round has to know.
        // A beating shows a reply later, as the client shows a hit's damage a tick later (the simulator's lost tick).
        boolean beaten = beatenLastTick;
        beatenLastTick = Fight18.takeDowned();
        if (beaten | died) {
            o.addProperty("died", true);
            died = false;
        }
        o.add("clicks", clicks);
        clicks = new JsonArray();

        ((dev.mcbridge.mixin.MinecraftInvoker) mc).mcbridge$pick(1.0f); // what the crosshair points at right now
        addHit(o, mc.hitResult);

        BlockPos feet = pl.blockPosition();
        o.addProperty("solid", solidAround(mc.level, feet, blockRadius));
        if (mapRadius > 0) o.addProperty("surface", surfaceAround(mc.level, feet, mapRadius));
        o.addProperty("hurt_time", pl.hurtTime);
        o.add("feet", vec(feet.getX(), feet.getY(), feet.getZ()));

        if (sendEntities) {
            JsonArray list = new JsonArray();
            for (Entity e : mc.level.entitiesForRendering()) {
                JsonObject j = new JsonObject();
                j.addProperty("id", e.getId());
                j.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                j.add("pos", vec(e.getX(), e.getY(), e.getZ()));
                Vec3 ev = e.getDeltaMovement();
                j.add("vel", vec(ev.x, ev.y, ev.z));
                j.addProperty("yaw", e.getYRot());
                j.addProperty("pitch", e.getXRot());
                list.add(j);
            }
            o.add("entities", list);
        }
        return o;
    }

    private static void addHit(JsonObject o, HitResult hit) {
        if (hit instanceof BlockHitResult bh && hit.getType() == HitResult.Type.BLOCK) {
            JsonObject h = new JsonObject();
            h.addProperty("type", "block");
            BlockPos b = bh.getBlockPos();
            h.add("block", vec(b.getX(), b.getY(), b.getZ()));
            h.addProperty("face", bh.getDirection().getName());
            Vec3 l = bh.getLocation();
            h.add("loc", vec(l.x, l.y, l.z));
            o.add("hit", h);
        } else if (hit instanceof EntityHitResult eh && hit.getType() == HitResult.Type.ENTITY) {
            JsonObject h = new JsonObject();
            h.addProperty("type", "entity");
            h.addProperty("entity", eh.getEntity().getId());
            o.add("hit", h);
        }
    }

    /** The top block of every column near the feet, x then z: 'A' is MAP_BELOW under the feet, and '.' is nothing there. */
    private static String surfaceAround(Level level, BlockPos feet, int r) {
        StringBuilder sb = new StringBuilder((2 * r + 1) * (2 * r + 1));
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int x = -r; x <= r; x++) {
            for (int z = -r; z <= r; z++) {
                char found = '.';
                for (int y = MAP_ABOVE; y >= -MAP_BELOW; y--) {
                    if (!level.getBlockState(p.set(feet.getX() + x, feet.getY() + y, feet.getZ() + z)).isAir()) {
                        found = (char) ('A' + y + MAP_BELOW);
                        break;
                    }
                }
                sb.append(found);
            }
        }
        return sb.toString();
    }

    /** Which block is where, one character per block, for what recordings may be needed for later. Server thread only. */
    private String blockIdsAround(Level level, BlockPos feet, int r) {
        StringBuilder sb = new StringBuilder((2 * r + 1) * (2 * r + 1) * (2 * r + 1));
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int x = -r; x <= r; x++) {
            for (int y = -r; y <= r; y++) {
                for (int z = -r; z <= r; z++) {
                    var block = level.getBlockState(p.set(feet.getX() + x, feet.getY() + y, feet.getZ() + z)).getBlock();
                    sb.append(blockChars.computeIfAbsent(block, b -> (char) (0x100 + blockChars.size())));
                }
            }
        }
        return sb.toString();
    }

    /** Solid (1) or not (0) in a cube around the feet, x then y then z. */
    private static String solidAround(Level level, BlockPos feet, int r) {
        StringBuilder sb = new StringBuilder((2 * r + 1) * (2 * r + 1) * (2 * r + 1));
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        for (int x = -r; x <= r; x++) {
            for (int y = -r; y <= r; y++) {
                for (int z = -r; z <= r; z++) {
                    sb.append(level.getBlockState(p.set(feet.getX() + x, feet.getY() + y, feet.getZ() + z)).isAir() ? '0' : '1');
                }
            }
        }
        return sb.toString();
    }

    /** Any region of the client's world, with block names. */
    private static JsonObject blockNames(Minecraft mc, JsonArray min, JsonArray max) {
        JsonArray names = new JsonArray();
        BlockPos.MutableBlockPos p = new BlockPos.MutableBlockPos();
        int[] a = ints(min);
        int[] b = ints(max);
        for (int x = a[0]; x <= b[0]; x++) {
            for (int y = a[1]; y <= b[1]; y++) {
                for (int z = a[2]; z <= b[2]; z++) {
                    names.add(BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(p.set(x, y, z)).getBlock()).toString());
                }
            }
        }
        JsonObject o = new JsonObject();
        o.add("names", names);
        return o;
    }
}
