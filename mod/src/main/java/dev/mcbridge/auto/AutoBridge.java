package dev.mcbridge.auto;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ServerData;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Two helpers for recording bridges by hand. The player walks and aims; these only press what a hand would press.
 * G turns both on and off. Scaffold: while looking down and walking backward, holds shift at the edge of the ground, and
 * shift held by hand always wins. Clicker: only while the
 * right mouse button is held and the crosshair is on a block, clicks every tick, with shift down now or within the last
 * 2 ticks, which is the rule the model has to follow too. Both need a block in the hand and a server on this machine.
 */
public final class AutoBridge {
    private static final double SNEAK_AT = 0.2; // blocks before the edge, plus how far it moves in two ticks
    private static final double RELEASE_AT = 0.45;
    private static final int CLICK_AFTER_SHIFT = 2; // ticks
    private static final float BRIDGING_PITCH = 60f; // degrees below level: looking down this far means bridging
    private static boolean on;
    private static boolean pressingShift;
    private static int sinceShift = 100;

    private AutoBridge() {}

    public static void register(boolean startOn) {
        on = startOn;
        KeyMapping toggle = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.mcbridge.helpers", InputConstants.Type.KEYBOARD, InputConstants.KEY_G, KeyMapping.Category.MISC));
        ClientTickEvents.START_CLIENT_TICK.register(mc -> {
            while (toggle.consumeClick()) {
                on = !on;
                if (!on) letGoOfShift(mc);
                if (mc.player != null) mc.player.sendOverlayMessage(Component.literal("Bridge helpers " + (on ? "ON" : "OFF")));
            }
            if (on) tick(mc);
        });
    }

    private static boolean onThisMachine(Minecraft mc) {
        if (mc.isLocalServer()) return true;
        ServerData server = mc.getCurrentServer();
        return server != null && (server.ip.startsWith("localhost") || server.ip.startsWith("127."));
    }

    /** Hands shift back to the player: if they are holding it themselves, it stays down. */
    private static void letGoOfShift(Minecraft mc) {
        if (pressingShift) mc.options.keyShift.setDown(holdingShiftByHand(mc));
        pressingShift = false;
    }

    private static boolean holdingShiftByHand(Minecraft mc) {
        InputConstants.Key key = KeyMappingHelper.getBoundKeyOf(mc.options.keyShift);
        return key.getType() == InputConstants.Type.KEYBOARD && InputConstants.isKeyDown(key.getValue());
    }

    private static void tick(Minecraft mc) {
        LocalPlayer player = mc.player;
        boolean usable = player != null && mc.level != null && mc.gui.screen() == null && onThisMachine(mc) && player.getMainHandItem().getItem() instanceof BlockItem;
        if (!usable) {
            letGoOfShift(mc);
            return;
        }
        scaffold(mc, player);
        sinceShift = mc.options.keyShift.isDown() ? 0 : sinceShift + 1;
        if (sinceShift <= CLICK_AFTER_SHIFT && mc.options.keyUse.isDown() && mc.hitResult != null && mc.hitResult.getType() == HitResult.Type.BLOCK) {
            KeyMapping.click(KeyMappingHelper.getBoundKeyOf(mc.options.keyUse));
        }
    }

    private static void scaffold(Minecraft mc, LocalPlayer player) {
        // Bridging means looking down and walking backward. Anything else is running or parkour.
        if (player.getXRot() < BRIDGING_PITCH || !mc.options.keyDown.isDown() || mc.options.keyUp.isDown()) {
            letGoOfShift(mc);
            return;
        }
        Vec3 moving = player.getDeltaMovement().multiply(1, 0, 1);
        double speed = moving.length();
        if (speed < 0.005) return; // standing still changes nothing
        double ground = groundAhead(mc, player, moving.scale(1 / speed));
        if (player.onGround() && ground < SNEAK_AT + speed * 2) {
            pressingShift = true;
            mc.options.keyShift.setDown(true);
        } else if (pressingShift && ground >= RELEASE_AT) {
            letGoOfShift(mc);
        }
    }

    /** How far the ground continues the way the player moves, up to one block. */
    private static double groundAhead(Minecraft mc, LocalPlayer player, Vec3 way) {
        int y = (int) Math.floor(player.getY() - 0.01);
        for (double s = 0; s <= 1.0; s += 0.05) {
            BlockPos under = BlockPos.containing(player.getX() + way.x * s, y, player.getZ() + way.z * s);
            if (mc.level.getBlockState(under).isAir()) return s;
        }
        return 1.0;
    }
}
