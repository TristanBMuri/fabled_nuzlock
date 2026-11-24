"""Dashboard data (UI_TODO.md): folds the probe's event files (`<run>.events.jsonl`) and
the trace / deaths / matrix JSON (`rr_trace.py`, `rr_deaths.py`, `rr_matchups.py --json`)
from logs/ and results/ into one object for ui/index.html. The searched battles
(`<run>.states.jsonl`, `--states`) and battle lines (`rl/line.py --out logs/line_*.json`) are
indexed here and read one at a time by the page (served mode only):
`/battle?file=logs/x.states.jsonl&k=N` (one battle, every decision), `/line?file=logs/line_x.json`.

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
        elif e["kind"] in ("draft", "screen"):  # matched to their six below
            b.setdefault(e["kind"], []).append(e)
        elif b is not None:
            b[e["kind"]] = e
    for b in bosses.values():
        b["boxes"].sort(key=lambda e: e.get("i", 0))  # the workers finish boxes out of order
        # a box's draft / screen events carry no box number: the six they produced identifies them
        drafts = {tuple(sorted(p["pick"] for p in d["picks"])): d["picks"] for d in b.pop("draft", [])}
        screens = {tuple(sorted(s["scored"][0][0])): s["scored"][:24] for s in b.pop("screen", []) if s["scored"]}
        for x in b["boxes"] + b["teams"]:
            six = tuple(sorted(x["team"]))
            x["draft"], x["screen"] = drafts.get(six), screens.get(six)
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


CACHE: dict = {}  # path -> ((mtime, size), value): the states files are tens of MB, data.json is rebuilt every fetch


def cached(path: Path, fn):
    st = path.stat()
    key = (st.st_mtime, st.st_size)
    if CACHE.get(path, (None,))[0] != key:
        CACHE[path] = (key, fn(path))
    return CACHE[path][1]


def enemy_act(chosen: list, turn: int) -> str | None:
    """The enemy's first chosen action of a turn, as a move / species id ("move bulletseed mega" -> bulletseed)."""
    a = next((a for t, side, a in chosen if t == turn and side == "p2"), None)
    return a and "".join(c for c in a.split()[1].lower() if c.isalnum())


def pred_hit(pred: list | None, act: str | None) -> bool | None:
    """Did the AI model name the enemy's actual choice? None when there is nothing to compare."""
    pred = [p.lower().removeprefix("switch ") for p in pred or [] if p != "No Move"]
    pred = [p.rstrip("0123456789") if p.startswith("hiddenpower") else p for p in pred]  # hiddenpowerrock60
    if not pred or not act:
        return None
    return any(act == "".join(c for c in p if c.isalnum()) or act.startswith(p) for p in pred)


def picked(d: dict) -> list:
    return max(d["options"], key=lambda o: (o[1], o[2], o[0])) if d["options"] else [d.get("pick"), 0, None]


def states_index(path: Path) -> list[dict]:
    """One row per battle of a states file: its result, AI-model misses, the largest one-turn
    drop of the picked option's value, and the byte offsets of its decision lines. Decisions are
    written before their battle line by the same worker; workers interleave, the tag separates them."""
    pending, out, off = {}, [], 0
    with open(path, "rb") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                e = {}
            key = (e.get("boss"), e.get("battle"), str(e.get("battle_seed")), str(e.get("team")))
            if e.get("kind") == "decision":
                pending.setdefault(key, []).append((off, e["turn"], e.get("ai_pred"), picked(e)[2]))
            elif e.get("kind") == "battle":
                ds, seen = pending.pop(key, []), set()
                hits = [pred_hit(p, enemy_act(e["chosen"], t)) for _, t, p, _ in ds if not (t in seen or seen.add(t))]
                vals = [v for *_, v in ds if v is not None]
                drop = max([(a - b, ds[i + 1][1]) for i, (a, b) in enumerate(zip(vals, vals[1:]))], default=(0, None))
                out.append({"k": len(out), "boss": e["boss"], "battle": e["battle"], "team": e["team"], "won": e["won"],
                            "deaths": sum(e["fainted"]), "turns": e["turns"], "timed_out": e["timed_out"],
                            "n": len(ds), "miss": hits.count(False), "checked": len(hits) - hits.count(None),
                            "drop": round(drop[0], 3), "drop_turn": drop[1], "offsets": [o for o, *_ in ds], "line": off})
            off += len(line)
    return out


def states_summary(path: Path, where: str) -> dict:
    rows = cached(path, states_index)
    # the page lists at most 300 battles a file: the bad ones first (lost, most deaths, largest drop)
    show = sorted(rows, key=lambda r: (r["won"], -r["deaths"], -r["drop"]))[:300]
    return {"file": f"{where}/{path.name}", "mtime": path.stat().st_mtime, "n": len(rows),
            "won": sum(r["won"] for r in rows), "miss": sum(r["miss"] for r in rows), "checked": sum(r["checked"] for r in rows),
            "battles": [{k: v for k, v in r.items() if k not in ("offsets", "line")} for r in show]}


def battle(query: str) -> dict | None:
    """One battle of a states file: the battle line plus every decision (state string, options, pick, AI model's prediction)."""
    q = urllib.parse.parse_qs(query)
    name, k = q.get("file", [""])[0], int(q.get("k", ["-1"])[0] or -1)
    if name not in {f"{w}/{f.name}" for w in DIRS for f in (ROOT / w).glob("*.states.jsonl")}:
        return None
    rows = cached(ROOT / name, states_index)
    if not 0 <= k < len(rows):
        return None
    row, decisions = dict(rows[k]), []
    with open(ROOT / name, "rb") as f:
        for o in row.pop("offsets"):
            f.seek(o)
            decisions.append(json.loads(f.readline()))
        f.seek(row.pop("line"))
        e = json.loads(f.readline())  # the battle line: both sides' chosen actions
    row.update(chosen=e["chosen"], fainted=e["fainted"], battle_seed=e["battle_seed"])
    for d in decisions:
        d["actual"] = enemy_act(row["chosen"], d["turn"])
        d["hit"] = pred_hit(d.get("ai_pred"), d["actual"])
    row["decisions"] = decisions
    return row


def line_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {"boss": d["boss"], "team": d["team"], "summary": d["summary"]}


def build() -> dict:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    data = {"runs": [], "traces": [], "deaths": [], "matrices": [], "states": [], "lines": []}
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
        for f in sorted((ROOT / where).glob("*.states.jsonl")):
            data["states"].append(states_summary(f, where))
        for f in sorted((ROOT / where).glob("line_*.json")):
            try:
                data["lines"].append({"file": f"{where}/{f.name}", "mtime": f.stat().st_mtime, **cached(f, line_summary)})
            except (json.JSONDecodeError, KeyError):
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
        if path == "/battle":
            b = battle(query)
            return self.send_body(json.dumps(b).encode(), "application/json") if b else self.send_error(404, "no such battle")
        if path == "/line":
            name = urllib.parse.parse_qs(query).get("file", [""])[0]
            if name not in {f"{w}/{f.name}" for w in DIRS for f in (ROOT / w).glob("line_*.json")}:
                return self.send_error(404, "no such line file")
            return self.send_body((ROOT / name).read_bytes(), "application/json")
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
