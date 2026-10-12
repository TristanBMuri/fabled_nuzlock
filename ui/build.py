"""Dashboard data (UI_TODO.md): folds the probe's event files (`<run>.events.jsonl`) and
the trace / deaths / matrix JSON (`rr_trace.py`, `rr_deaths.py`, `rr_matchups.py --json`)
from logs/ and results/ into one object for ui/index.html.

  uv run python ui/build.py               # writes ui/data.json once
  uv run python ui/build.py --serve       # http://localhost:8765, data.json rebuilt on every fetch

`/replay?file=logs/trace_x.json[&turn=N]` (served mode) plays a trace's protocol log in
Showdown's replay viewer (play.pokemonshowdown.com/js/replay-embed.js: needs internet).

Ports 8000-8007 belong to the old project.
"""

import argparse
import functools
import http.server
import json
import subprocess
import urllib.parse
from pathlib import Path

UI = Path(__file__).resolve().parent
ROOT = UI.parent
DIRS = ("logs", "results")
REPLAY = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto"><div class="battle"></div><div class="battle-log"></div>
<div class="replay-controls"></div><div class="replay-controls-2"></div></div>
<script type="text/plain" class="battle-log-data">{log}</script>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
<script>const init = window.onload; window.onload = () => {{ init(); if ({turn}) Replays.battle.seekTurn({turn}, true); }};</script>
"""


def read_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            pass
    return out


def fold_gauntlet(events: list[dict], start: dict, base: dict) -> dict:
    """A BDSP planner run (plan_gauntlet.py --out): per episode its progress and deaths, per
    trainer the deaths and lost battles, deaths by species. The battles themselves stay on disk."""
    order = start.get("order", [])
    eps = {s: {"seed": s, "at": 0, "deaths": [], "result": None, "party": []} for s in start.get("seeds", [])}
    trainers, species = {}, {}
    for e in events:
        if e["kind"] == "battle":
            ep = eps.setdefault(e["seed"], {"seed": e["seed"], "at": 0, "deaths": [], "result": None, "party": []})
            ep["at"], ep["party"] = e["i"] + e["won"], e["party"]
            if e.get("trace"):
                ep.setdefault("traces", []).append([e["i"], e["trace"], e["won"]])
            tr = trainers.setdefault(e["i"], {"i": e["i"], "deaths": 0, "lost": 0, "fought": 0, "killed": {}})
            tr["fought"] += 1
            tr["lost"] += not e["won"]
            for sp in e["died"]:
                ep["deaths"].append([e["i"], sp])
                tr["deaths"] += 1
                tr["killed"][sp] = tr["killed"].get(sp, 0) + 1
                species[sp] = species.get(sp, 0) + 1
        elif e["kind"] == "episode":
            eps.setdefault(e["seed"], {"seed": e["seed"], "deaths": [], "party": []}).update(
                result=e, at=e["cleared"], party=e["party"])
    for tr in trainers.values():
        tr["name"], tr["location"], tr["breakpoint"] = (order[tr["i"]] if tr["i"] < len(order) else ["?", "?", "none"])
    done = [ep["result"] for ep in eps.values() if ep["result"]]
    mean = lambda k: sum(r[k] for r in done) / len(done) if done else None
    end = next((e for e in events if e["kind"] == "end"), None)
    return {**base, "n_episodes": len(eps), "episodes_done": len(done), "slice": len(order),
            "milestones": [[i, n, b] for i, (n, loc, b) in enumerate(order) if b != "none"],
            "summary": {"cleared": mean("cleared"), "no_death": mean("no_death"), "deaths": mean("deaths"),
                        "clear": mean("clear"), "final": bool(end)},
            "episodes": sorted(eps.values(), key=lambda ep: ep["seed"]),
            "trainers": sorted(trainers.values(), key=lambda tr: (-tr["deaths"], -tr["lost"]))[:25],
            "species": sorted(species.items(), key=lambda kv: -kv[1])[:20]}


def fold(path: Path, where: str, ps: str) -> dict:
    """One run: its config, state and, per boss, every event in the order it came."""
    events = read_events(path)
    start = next((e for e in events if e["kind"] == "start"), {"config": {}, "bosses": [], "mode": "?", "t": 0})
    if start["mode"] == "gauntlet":
        end = next((e for e in events if e["kind"] == "end"), None)
        out = start["config"].get("out") or ""
        running = end is None and where == "logs" and bool(out) and Path(out).name in ps
        base = {"id": f"{where}/{path.name.removesuffix('.events.jsonl')}", "where": where, "mode": "gauntlet",
                "config": start["config"], "state": "done" if end else "running" if running else "stopped",
                "started": start["t"], "updated": max((e["t"] for e in events), default=0),
                "minutes": end and end["minutes"], "bosses": []}
        return fold_gauntlet(events, start, base)
    bosses = {b: {"name": b, "teams": [], "boxes": []} for b in start["bosses"]}
    for e in events:
        b = bosses.setdefault(e.get("boss"), {"name": e.get("boss"), "teams": [], "boxes": []}) if "boss" in e else None
        if e["kind"] in ("team", "box"):
            b[{"team": "teams", "box": "boxes"}[e["kind"]]].append(e)
        elif b is not None:
            b[e["kind"]] = e
    for b in bosses.values():
        b["boxes"].sort(key=lambda e: e.get("i", 0))  # the workers finish boxes out of order
    end = next((e for e in events if e["kind"] == "end"), None)
    out = start["config"].get("out") or ""
    running = end is None and where == "logs" and bool(out) and Path(out).name in ps
    n_boxes = start["config"].get("boxes") or 0
    return {"id": f"{where}/{path.name.removesuffix('.events.jsonl')}", "where": where, "mode": start["mode"],
            "config": start["config"], "state": "done" if end else "running" if running else "stopped",
            "started": start["t"], "updated": max((e["t"] for e in events), default=0), "minutes": end and end["minutes"],
            "n_bosses": len(start["bosses"]), "bosses_done": sum("boss_done" in b for b in bosses.values()),
            "n_boxes": n_boxes * len(start["bosses"]), "boxes_done": sum(len(b["boxes"]) for b in bosses.values()),
            "bosses": [b for name, b in bosses.items() if name is not None]}


def build() -> dict:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    data = {"runs": [], "traces": [], "deaths": [], "matrices": []}
    for where in DIRS:
        for f in sorted((ROOT / where).glob("*.events.jsonl")):
            try:
                data["runs"].append(fold(f, where, ps))
            except Exception as e:  # one bad file must not blank the page
                print(f"{f}: {type(e).__name__}: {e}", flush=True)
        for kind, glob in (("traces", "trace_*.json"), ("deaths", "deaths_*.json"), ("matrices", "matrix_*.json")):
            for f in sorted((ROOT / where).glob(glob)):
                try:
                    d = json.loads(f.read_text())
                    d["log"] = "log" in d  # the replay route reads the log itself; the page only needs to know
                    data[kind].append({"file": f"{where}/{f.name}", "mtime": f.stat().st_mtime, **d})
                except json.JSONDecodeError:
                    pass
    data["runs"].sort(key=lambda r: -r["updated"])
    return data


def replay(query: str) -> str | None:
    """Showdown's downloaded-replay page for one trace file (only files the globs list)."""
    q = urllib.parse.parse_qs(query)
    name = q.get("file", [""])[0]
    if name not in {f"{w}/{f.name}" for w in DIRS for f in (ROOT / w).glob("trace_*.json")}:
        return None
    t = json.loads((ROOT / name).read_text())
    if "log" not in t:
        return None
    turn = int(q.get("turn", ["0"])[0] or 0)
    log = "\n".join(t["log"]).replace("</", "<\\/")  # the viewer turns \/ back into /
    return REPLAY.format(title=f"{t['boss']} battle {t['battle']}", log=log, turn=turn)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/replay":
            page = replay(query)
            return self.send_body(page.encode(), "text/html") if page else self.send_error(404, "no such trace (or it has no log)")
        if path != "/data.json":
            return super().do_GET()
        self.send_body(json.dumps(build()).encode(), "application/json")

    def send_body(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="serve ui/ and rebuild data.json on every fetch")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not args.serve:
        (UI / "data.json").write_text(json.dumps(build()))
        return
    print(f"http://localhost:{args.port}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(UI))).serve_forever()


if __name__ == "__main__":
    main()
