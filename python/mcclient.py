"""Talks to one driven Minecraft client: every step blocks until the game has run those ticks."""
import json
import socket
import time


class McClient:
    def __init__(self, port, timeout=180.0):
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"no Minecraft client on port {port}")
                time.sleep(1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.file = self.sock.makefile("r", encoding="utf-8")
        self.next_id = 0

    def call(self, **msg):
        self.next_id += 1
        msg["id"] = self.next_id
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        line = self.file.readline()
        if not line:
            raise ConnectionError("Minecraft closed the connection")
        reply = json.loads(line)
        if reply.get("id") != self.next_id:
            raise RuntimeError(f"reply out of order: {reply.get('id')} != {self.next_id}")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def step(self, ticks=1, **action):
        return self.call(op="step", action=action, ticks=ticks)

    def reset(self, **spec):
        return self.call(op="reset", **spec)

    def render(self, on):
        return self.call(op="render", on=on)

    def close(self):
        """Really ends the connection: the file from makefile() holds the socket open otherwise, and the mod, which
        serves one connection at a time, would wait on it forever while a new one queues behind it."""
        try:
            self.file.close()
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
