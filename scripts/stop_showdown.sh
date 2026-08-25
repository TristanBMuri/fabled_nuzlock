#!/usr/bin/env bash
# Stop Showdown servers on START_PORT .. START_PORT+N-1.
# Usage: ./scripts/stop_showdown.sh NUM_SERVERS [START_PORT]
# Default START_PORT=8000. Only kills this project's start command (not 8000-8007).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 NUM_SERVERS [START_PORT]" >&2
  exit 1
fi

NUM=$1
START_PORT=${2:-8000}

for (( i=0; i<NUM; i++ )); do
  port=$((START_PORT + i))
  pkill -f "pokemon-showdown start --no-security --port $port" 2>/dev/null || true
  echo "stopped $port (if it was running)"
done
