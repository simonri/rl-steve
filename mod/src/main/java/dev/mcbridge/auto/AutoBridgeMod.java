package dev.mcbridge.auto;

import net.fabricmc.api.ClientModInitializer;

/** Entry point of the small mod that holds only the bridging helper, for a normal client. */
public final class AutoBridgeMod implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        AutoBridge.register(false);
    }
}
