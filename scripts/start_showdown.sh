#!/usr/bin/env bash
# Start N local Showdown servers on START_PORT .. START_PORT+N-1.
# Usage: ./scripts/start_showdown.sh NUM_SERVERS [START_PORT]
# Default START_PORT=8000. Ports 8000-8007 belong to the old project — don't collide.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 NUM_SERVERS [START_PORT]" >&2
  exit 1
fi

NUM=$1
START_PORT=${2:-8000}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD="$ROOT/pokemon-showdown"
LOGDIR="$ROOT/logs"

if (( START_PORT < 8000 )); then
  echo "START_PORT must be >= 8000 (8000-8007 belong to the old project)" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

for (( i=0; i<NUM; i++ )); do
  port=$((START_PORT + i))
  if curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:$port/"; then
    echo "port $port already up, skipping"
    continue
  fi
  (cd "$SD" && node pokemon-showdown start --no-security --port "$port" \
    > "$LOGDIR/showdown_$port.log" 2>&1 &)
  ok=0
  for _ in $(seq 1 50); do
    if curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:$port/"; then
      ok=1
      break
    fi
    sleep 0.3
  done
  if (( ok == 0 )); then
    echo "failed to start showdown on $port (see $LOGDIR/showdown_$port.log)" >&2
    exit 1
  fi
  echo "showdown up on $port"
done
