package dev.mcbridge.mixin;

import net.minecraft.client.FramerateLimiter;
import dev.mcbridge.McBridge;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Stops the frame limiter, including the inactive-window limit, from slowing down the game loop. */
@Mixin(FramerateLimiter.class)
public class RenderSystemMixin {
    @Inject(method = "limitDisplayFPS", at = @At("HEAD"), cancellable = true)
    private static void mcbridge$noLimit(int fps, CallbackInfo ci) {
        if (McBridge.isLockstep()) ci.cancel();
    }
}
