"""The RL cockpit: serves rl/cockpit.html, which tails a training run's logs/rl/<run>.jsonl.

    uv run python rl/cockpit.py          # http://localhost:8766 (ports 8000-8007 are the old project's)

/runs lists the runs with their status, /log?run=x&from=N returns the log from byte N on (the page
polls every 3 s). POST /start launches lesson2_ppo.py and pauses whatever runs, /pause, /resume and
/stop act on one run: one run trains at a time, paused runs keep their memory but use no CPU.
/start with seeds "0,1,2" makes one run per seed (<run>_s0 ...): the first starts, the rest wait in
the queue (/queue, POST /unqueue), which starts the next whenever nothing is training. The queue
lives in memory: restarting the cockpit forgets it.
"""

import argparse
import http.server
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs" / "rl"
NAME = re.compile(r"^[\w.-]{1,60}$")
QUEUE = []  # flags of runs waiting their turn
LAST = {"run": "", "t": 0.0}  # the last launch: its log appears only after torch has loaded
LAUNCHED = set()  # names launched by this cockpit, taken even before their log exists
LOCK = threading.Lock()  # "pause the others, then launch / resume" is one step


def run_info(path):
    """name, status (running / paused / done / stopped) and pid, from the log and /proc."""
    with path.open("rb") as f:
        first = json.loads(f.readline() or b"{}")
        f.seek(max(0, path.stat().st_size - 4096))
        ended = b'"kind": "end"' in f.read()
    pid, status = first.get("pid"), "done" if ended else "stopped"
    if pid and not ended:
        try:  # alive, and still this run (pids get reused)
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            if b"lesson2_ppo.py" in b" ".join(cmd) and path.stem.encode() in cmd and state != "Z":
                status = "paused" if state == "T" else "running"
        except OSError:
            pass
    return {"name": path.stem, "mtime": path.stat().st_mtime, "size": path.stat().st_size, "status": status, "pid": pid,
            "pt": path.with_suffix(".pt").is_file(), "config": first.get("config", {})}


def runs():
    return [run_info(p) for p in sorted(LOGS.glob("*.jsonl"), key=lambda p: -p.stat().st_mtime)]


def pause_all(but=None):
    for r in runs():
        if r["status"] == "running" and r["name"] != but:
            os.kill(r["pid"], signal.SIGTSTP)


def taken(name):
    return (not NAME.match(name) or (LOGS / f"{name}.jsonl").exists() or name in LAUNCHED
            or any(q["run"] == name for q in QUEUE))


def submit(body):
    """/start: one run, or one per seed; queue=true queues them all instead of starting the first now."""
    seeds = [x.strip() for x in str(body.pop("seeds", "")).split(",") if x.strip()]
    later = body.pop("queue", False)
    jobs = [{**body, "run": f"{body.get('run')}_s{x}", "seed": x} for x in seeds] or [body]
    bad = [j["run"] for j in jobs if taken(str(j.get("run", "")))]
    if bad:
        return f"bad or taken run name: {', '.join(map(str, bad))}"
    if not later:
        err = start(jobs.pop(0))
        if err:
            return err
    QUEUE.extend(jobs)
    return ""


def run_queue():
    """every 5 s: nothing training and a run waiting -> start it."""
    while True:
        threading.Event().wait(5)
        with LOCK:
            starting = not (LOGS / f"{LAST['run']}.jsonl").exists() and time.time() - LAST["t"] < 120
            if QUEUE and not starting and not any(r["status"] == "running" for r in runs()):
                start(QUEUE.pop(0))


def start(flags):
    """flags: {"run": name, "algo": "ppo", "pool": true, ...} -> lesson2_ppo.py --run name --algo ppo --pool ..."""
    name = str(flags.get("run", ""))
    if taken(name):
        return f"bad or taken run name: {name!r}"
    argv = [sys.executable, "-u", str(HERE / "lesson2_ppo.py")]
    for k, v in flags.items():
        k = k.replace("_", "-")  # config keys -> flags
        if not re.match(r"^[a-z][a-z-]*$", k) or v in (None, "", False):
            continue
        argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    pause_all()
    LAST.update(run=name, t=time.time())
    LAUNCHED.add(name)
    out = (LOGS / f"{name}.out").open("w")  # stdout and errors, if the run dies at startup
    subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=HERE.parent,
                     env={**os.environ, "OMP_NUM_THREADS": "1"})
    return ""


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, body, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        if url.path == "/":
            self.reply((HERE / "cockpit.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/runs":
            self.reply(json.dumps(runs()).encode())
        elif url.path == "/queue":
            self.reply(json.dumps([{"run": q["run"], "note": q.get("note", "")} for q in QUEUE]).encode())
        elif url.path in ("/log", "/out") and NAME.match(q.get("run", "")):
            path = LOGS / (q["run"] + (".jsonl" if url.path == "/log" else ".out"))
            if not path.is_file():
                return self.send_error(404)
            with path.open("rb") as f:
                f.seek(int(q.get("from", 0)))
                body = f.read()
            self.reply(body[: body.rfind(b"\n") + 1] if url.path == "/log" else body[-4000:], "text/plain; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if self.headers.get("Content-Type") != "application/json":  # other web pages cannot send this without asking
            return self.send_error(415)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK:
            err = self.act(url.path, body)
        self.reply(json.dumps({"error": err}).encode())

    def act(self, path, body):
        if path == "/start":
            return submit(body)
        if path == "/unqueue":
            QUEUE[:] = [q for q in QUEUE if body.get("run") not in (None, q["run"])]  # no run = clear all
            return ""
        r = next((r for r in runs() if r["name"] == body.get("run")), None)
        if r is None or r["status"] not in ("running", "paused"):
            return "that run is not alive"
        if path == "/pause":
            os.kill(r["pid"], signal.SIGTSTP)  # the trainer logs "pause" and stops itself
        elif path == "/resume":
            pause_all(but=r["name"])
            os.kill(r["pid"], signal.SIGCONT)
        elif path == "/stop":
            os.kill(r["pid"], signal.SIGCONT)  # a stopped process only dies once it runs again
            os.kill(r["pid"], signal.SIGTERM)
        else:
            return "unknown action"
        return ""

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # finished runs are reaped, no zombies
    threading.Thread(target=run_queue, daemon=True).start()
    print(f"cockpit on http://localhost:{args.port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
