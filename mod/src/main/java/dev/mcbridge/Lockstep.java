package dev.mcbridge;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import net.minecraft.server.MinecraftServer;

/**
 * Hands one tick at a time from the client thread to the integrated server thread.
 * The server only runs a tick after the client asks for it, and the client waits for it to finish.
 */
final class Lockstep {
    private final Semaphore go = new Semaphore(0);
    private final Semaphore done = new Semaphore(0);
    private final Queue<Runnable> serverTasks = new ConcurrentLinkedQueue<>();
    private volatile boolean enabled;
    private boolean holdingTick; // server thread only

    boolean enabled() { return enabled; }

    void enable() {
        // disable() can leave a permit behind, which would otherwise run one tick nobody asked for.
        go.drainPermits();
        done.drainPermits();
        enabled = true;
    }

    void disable() {
        enabled = false;
        go.release(); // let a waiting server thread continue
    }

    /** Runs on the server thread at the start of the next stepped tick. */
    void onServer(Runnable task) { serverTasks.add(task); }

    void serverStartTick(MinecraftServer server) {
        holdingTick = false;
        if (!enabled) return;
        try {
            while (!go.tryAcquire(200, TimeUnit.MILLISECONDS)) {
                if (!enabled || !server.isRunning()) return;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
        }
        if (!enabled) return;
        holdingTick = true;
        Runnable task;
        while ((task = serverTasks.poll()) != null) task.run();
    }

    void serverEndTick() {
        if (holdingTick) {
            holdingTick = false;
            done.release();
        }
    }

    /** Client thread: runs one server tick to completion. */
    void stepServer() {
        go.release();
        try {
            if (!done.tryAcquire(30, TimeUnit.SECONDS)) {
                throw new IllegalStateException("server did not finish its tick within 30 s");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }
}
