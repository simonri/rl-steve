"""Starts real Minecraft dev clients, silently on their own Hyprland workspace."""
import argparse
import json
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

DISPLAY = ":99"
SLICE = "mcai.slice"
SLICE_LIMITS = ["CPUWeight=20", "CPUQuota=1200%", "MemoryMax=28G"]  # other services win any contest for the CPU

OPTIONS = """renderDistance:4
simulationDistance:5
maxFps:260
enableVsync:false
pauseOnLostFocus:false
soundCategory_master:0.0
guiScale:1
"""


def launch(name, port=None, watch_port=None, join=None, workspace=19, width=480, height=270, heap="2G", username=None, test_menu=False, render_distance=None, headless=False, human=False, autobridge=None):
    """Writes a launch script for one client and starts it through Hyprland, or on a server without a screen."""
    cfg = json.loads((ROOT / "mod/build/client-command.json").read_text())
    game_dir = RUNS / name
    game_dir.mkdir(parents=True, exist_ok=True)
    opts = game_dir / "options.txt"
    text = opts.read_text() if opts.exists() else OPTIONS
    if render_distance:
        text = re.sub(r"renderDistance:\d+", f"renderDistance:{render_distance}", text)
        text = re.sub(r"simulationDistance:\d+", f"simulationDistance:{min(render_distance, 10)}", text)
    opts.write_text(text)

    args = list(cfg["args"])
    args += ["--gameDir", str(game_dir), "--width", str(width), "--height", str(height)]
    args += ["--username", username or name]
    if join:
        args += ["--quickPlayMultiplayer", join]
    # The argument file only repeats the classpath given below, and exists only after a run through Gradle.
    jvm = [a for a in cfg["jvmArgs"] if not a.startswith(("-Xmx", "@"))] + [f"-Xmx{heap}"]
    cmd = [cfg["java"], *jvm, "-cp", os.pathsep.join(cfg["classpath"]), cfg["mainClass"], *args]

    env = {}
    if port:
        env["MCBRIDGE_PORT"] = port
    if watch_port:
        env["MCBRIDGE_WATCH_PORT"] = watch_port
    if test_menu:
        env["MCBRIDGE_TEST_MENU"] = "1"
    if human:
        env["MCBRIDGE_HUMAN"] = "1"
    if autobridge:
        env["MCBRIDGE_AUTOBRIDGE"] = autobridge
    exports = "".join(f"export {k}={v}\n" for k, v in env.items())
    script = game_dir / "start.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"cd {shlex.quote(cfg['workingDir'])}\n"
        "unset WAYLAND_DISPLAY\n"
        + (f"export DISPLAY={DISPLAY} SDL_VIDEODRIVER=x11 LIBGL_ALWAYS_SOFTWARE=1\n" if headless else "")
        + f"{exports}"
        f"exec {shlex.join(cmd)} > {shlex.quote(str(game_dir / 'client.log'))} 2>&1\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    if headless:
        start_headless(name, script)
        return game_dir
    lua = f'hl.exec_cmd("{script}", {{ workspace = "{workspace} silent", no_initial_focus = true }})'
    subprocess.run(["hyprctl", "eval", lua], check=True, capture_output=True)
    return game_dir


def start_headless(name, script):
    """A virtual screen and the client, in a slice that always gives way to everything else on the server."""
    subprocess.run(["systemctl", "set-property", "--runtime", SLICE, *SLICE_LIMITS], check=True)
    if subprocess.run(["systemctl", "is-active", "--quiet", "mcai-screen"]).returncode != 0:
        subprocess.run(["systemd-run", f"--slice={SLICE}", "--unit=mcai-screen", "Xvfb", DISPLAY, "-screen", "0", "1024x768x24", "-nolisten", "tcp"], check=True, capture_output=True)
    subprocess.run(["systemd-run", f"--slice={SLICE}", f"--unit=mcai-{name}", "--collect", str(script)], check=True, capture_output=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--port")
    p.add_argument("--watch-port")
    p.add_argument("--join")
    p.add_argument("--workspace", type=int, default=19)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--test-menu", action="store_true", help="watchers get items that ask for tasks")
    p.add_argument("--render-distance", type=int)
    p.add_argument("--human", action="store_true", help="a person plays courses here and is recorded")
    p.add_argument("--username")
    p.add_argument("--autobridge", choices=("key", "on"), help="bridge helpers on a local server. key: G and H toggle them. on: a test player that uses them")
    p.add_argument("--headless", action="store_true", help="on a server: virtual screen, limited resources")
    a = p.parse_args()
    print(launch(a.name, a.port, a.watch_port, a.join, a.workspace, a.width, a.height, test_menu=a.test_menu, render_distance=a.render_distance, headless=a.headless, human=a.human, username=a.username, autobridge=a.autobridge))
