package dev.mcbridge.mixin;

import com.mojang.blaze3d.platform.SDLEventHandler;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(Minecraft.class)
public interface MinecraftInvoker {
    @Invoker("pick")
    void mcbridge$pick(float partialTicks);

    @Accessor("sdlEventHandler")
    SDLEventHandler mcbridge$sdl();
}
