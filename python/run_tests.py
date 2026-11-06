"""Runs every offline test (tests/*.py) side by side, and the simulator's Rust tests, and says which failed and why.
No game needed. The checks in checks/ need a running game and are run one by one by hand.

    python run_tests.py              # everything
    python run_tests.py duel camera  # only tests whose name contains one of these
    python run_tests.py --no-rust
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE / "tests"
SIM = HERE.parent / "sim"


def run(cmd, cwd):
    start = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode == 0, time.time() - start, (p.stdout + p.stderr).strip()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rust = "--no-rust" not in sys.argv and not args
    files = sorted(f for f in TESTS.glob("test_*.py") if not args or any(a in f.stem for a in args))
    jobs = {f.stem: ([sys.executable, str(f)], TESTS) for f in files}
    if rust:
        jobs["rust"] = (["cargo", "test", "--release", "--no-default-features", "-q"], SIM)
    workers = max(1, min(len(jobs), (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(workers) as pool:
        results = dict(zip(jobs, pool.map(lambda j: run(*j), jobs.values())))
    failed = [n for n, (ok, _, _) in results.items() if not ok]
    for name, (ok, secs, out) in sorted(results.items(), key=lambda r: r[1][0]):
        results_line = [line for line in out.splitlines() if line.startswith("test result")]  # cargo: the unit tests' line
        last = results_line[0] if results_line else (out.splitlines()[-1] if out else "")
        print(f"{'ok  ' if ok else 'FAIL'} {name:22s} {secs:5.0f} s   {last[:90]}")
    for name in failed:
        print(f"\n==== {name} ====\n" + "\n".join(results[name][2].splitlines()[-25:]))
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
