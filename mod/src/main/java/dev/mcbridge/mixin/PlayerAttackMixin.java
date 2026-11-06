package dev.mcbridge.mixin;

import dev.mcbridge.Fight18;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** On the server, a player's hit follows 1.8 (Fight18) while those rules are on. The client's own side stays 26.3. */
@Mixin(Player.class)
public abstract class PlayerAttackMixin {
    @Inject(method = "attack", at = @At("HEAD"), cancellable = true)
    private void mcbridge$attack18(Entity target, CallbackInfo ci) {
        if (Fight18.enabled && (Object) this instanceof ServerPlayer attacker) {
            Fight18.attack(attacker, target);
            ci.cancel();
        }
    }
}
