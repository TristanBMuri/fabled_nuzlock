"""Dashboard data (UI_TODO.md): folds the probe's event files (`<run>.events.jsonl`) and
the trace / deaths / matrix JSON (`rr_trace.py`, `rr_deaths.py`, `rr_matchups.py --json`)
from logs/ and results/ into one object for ui/index.html.

  uv run python ui/build.py               # writes ui/data.json once
  uv run python ui/build.py --serve       # http://localhost:8765, data.json rebuilt on every fetch

Ports 8000-8007 belong to the old project.
"""

import argparse
import functools
import http.server
import json
import subprocess
from pathlib import Path

UI = Path(__file__).resolve().parent
ROOT = UI.parent
DIRS = ("logs", "results")


def read_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            pass
    return out


def fold(path: Path, where: str, ps: str) -> dict:
    """One run: its config, state and, per boss, every event in the order it came."""
    events = read_events(path)
    start = next((e for e in events if e["kind"] == "start"), {"config": {}, "bosses": [], "mode": "?", "t": 0})
    bosses = {b: {"name": b, "teams": [], "boxes": []} for b in start["bosses"]}
    for e in events:
        b = bosses.setdefault(e.get("boss"), {"name": e.get("boss"), "teams": [], "boxes": []}) if "boss" in e else None
        if e["kind"] in ("team", "box"):
            b[{"team": "teams", "box": "boxes"}[e["kind"]]].append(e)
        elif b is not None:
            b[e["kind"]] = e
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
                    data[kind].append({"file": f"{where}/{f.name}", "mtime": f.stat().st_mtime, **json.loads(f.read_text())})
                except json.JSONDecodeError:
                    pass
    data["runs"].sort(key=lambda r: -r["updated"])
    return data


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/data.json":
            return super().do_GET()
        body = json.dumps(build()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
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
