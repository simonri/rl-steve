package dev.mcbridge;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.world.item.BlockItem;

/** For testing the helpers without a person: walks backward with the use button down and a fixed look. Never in the jar. */
final class AutoBridgeTestPlayer {
    private AutoBridgeTestPlayer() {}

    static void register() {
        ClientTickEvents.START_CLIENT_TICK.register(mc -> {
            if (mc.player == null || mc.gui.screen() != null || !(mc.player.getMainHandItem().getItem() instanceof BlockItem)) return;
            float yaw = Math.round(mc.player.getYRot() / 45f) * 45f;
            mc.player.setYRot(yaw);
            mc.player.setXRot(80f);
            mc.options.keyDown.setDown(true);
            mc.options.keyUse.setDown(true);
        });
    }
}
