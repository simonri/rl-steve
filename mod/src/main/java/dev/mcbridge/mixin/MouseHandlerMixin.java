package dev.mcbridge.mixin;

import dev.mcbridge.McBridge;
import net.minecraft.client.MouseHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** A driven client must never capture the user's real mouse cursor. */
@Mixin(MouseHandler.class)
public class MouseHandlerMixin {
    @Inject(method = "grabMouse", at = @At("HEAD"), cancellable = true)
    private void mcbridge$noGrab(CallbackInfo ci) {
        if (McBridge.isDriven()) ci.cancel();
    }
}
