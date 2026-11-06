package dev.mcbridge.mixin;

import net.minecraft.server.ServerTickRateManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Lets us sprint the server forever, so ticks only wait for Python. Approach from MCio (MIT). */
@Mixin(ServerTickRateManager.class)
public interface ServerTickRateManagerAccessor {
    @Accessor("remainingSprintTicks")
    void mcbridge$setRemainingSprintTicks(long ticks);

    @Accessor("scheduledCurrentSprintTicks")
    void mcbridge$setScheduledCurrentSprintTicks(long ticks);
}
