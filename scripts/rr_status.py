"""Progress of the Radical Red runs in logs/: one line per log touched in the last day —
running or done, bosses finished, boxes done, the latest result line.

  uv run python scripts/rr_status.py          # once
  watch -n 60 uv run python scripts/rr_status.py   # live
"""

import re
import subprocess
import time
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "logs"


def main() -> None:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    now = time.time()
    for log in sorted(LOGS.glob("rr_*.log"), key=lambda p: p.stat().st_mtime):
        if now - log.stat().st_mtime > 86400:
            continue
        stem = log.stem.replace("rr_", "", 1)
        text = log.read_text(errors="ignore")
        lines = [l for l in text.splitlines() if "poke-env - WARNING" not in l and l.strip()]
        done = bool(re.search(r"^Radical Red best-six probe|^by turn:|wins, \d+ deathless", text, re.M))
        running = (stem in ps or log.name in ps) and not done
        state = "done" if done else ("running" if running else "stopped?")
        boxes = len(re.findall(r" box +\d+/\d+", text))
        bosses = len(re.findall(r"best team under search", text))
        last = next((l for l in reversed(lines) if re.search(r"best team|boxes: mean|box +\d+/|threat |repaired|wins,|cover team|A solo|B \d+ teams|trainer #", l)), lines[-1] if lines else "")
        age = (now - log.stat().st_mtime) / 60
        print(f"{stem:<28} {state:<9} {bosses:>2} bosses {boxes:>3} boxes  {age:5.0f} min ago  {last[:110]}")


if __name__ == "__main__":
    main()
