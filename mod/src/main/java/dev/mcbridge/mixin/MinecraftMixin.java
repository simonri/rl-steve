package dev.mcbridge.mixin;

import dev.mcbridge.McBridge;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyVariable;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public class MinecraftMixin {
    /** While driven, run exactly one client tick per game loop instead of following the wall clock. */
    @ModifyVariable(method = "runTick(Z)V", at = @At("STORE"), ordinal = 0)
    private int mcbridge$oneTickPerLoop(int ticks) {
        return McBridge.isLockstep() ? 1 : ticks;
    }

    /** Training does not need pixels, so skip drawing the frame. Game logic is unaffected. */
    @Inject(method = "renderFrame", at = @At("HEAD"), cancellable = true)
    private void mcbridge$skipFrame(boolean advanceGameTime, CallbackInfo ci) {
        if (!McBridge.isDriven() || !McBridge.renderOff()) return;
        ci.cancel();
        if (!McBridge.isLockstep()) {
            // Idle: the skipped frame limiter would otherwise let this loop spin at full speed.
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
