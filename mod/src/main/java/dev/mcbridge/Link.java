package dev.mcbridge;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Newline-delimited JSON on 127.0.0.1. One Python client at a time. */
final class Link {
    private final BlockingQueue<JsonObject> inbox = new LinkedBlockingQueue<>();
    private volatile BufferedWriter out;

    void start(int port) {
        Thread t = new Thread(() -> serve(port), "mcbridge-link");
        t.setDaemon(true);
        t.start();
    }

    private void serve(int port) {
        try (ServerSocket server = new ServerSocket(port, 1, InetAddress.getLoopbackAddress())) {
            while (true) {
                try (Socket s = server.accept()) {
                    s.setTcpNoDelay(true);
                    BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8));
                    out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream(), StandardCharsets.UTF_8));
                    McBridge.LOG.info("Python connected");
                    String line;
                    while ((line = in.readLine()) != null) inbox.add(JsonParser.parseString(line).getAsJsonObject());
                } catch (IOException | RuntimeException e) {
                    McBridge.LOG.warn("link closed: {}", e.toString());
                }
                out = null;
                JsonObject bye = new JsonObject();
                bye.addProperty("op", "disconnected");
                inbox.add(bye);
            }
        } catch (IOException e) {
            McBridge.LOG.error("could not listen on port {}", port, e);
        }
    }

    JsonObject poll(long millis) throws InterruptedException {
        return inbox.poll(millis, TimeUnit.MILLISECONDS);
    }

    void reply(JsonObject msg) {
        BufferedWriter w = out;
        if (w == null) return;
        try {
            w.write(msg.toString());
            w.write('\n');
            w.flush();
        } catch (IOException e) {
            McBridge.LOG.warn("reply failed: {}", e.toString());
        }
    }
}
