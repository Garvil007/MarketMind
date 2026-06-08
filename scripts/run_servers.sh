#!/usr/bin/env bash
#
# Launch all three MarketMind MCP servers in the background, wait for each
# /mcp port to accept connections, then block until Ctrl-C — which kills them all.
#
#   Market Data :8001  |  News :8002  |  Portfolio :8003
#
# Honors $PYTHON (defaults to the venv python if present, else `python`).
#
set -euo pipefail

# Project root = parent of this script's directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pick an interpreter: explicit $PYTHON, else venv, else system python.
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
else
  PY="python"
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

NAMES=(market_data news portfolio)
MODULES=(marketmind.servers.market_data_server marketmind.servers.news_server marketmind.servers.portfolio_server)
PORTS=(8001 8002 8003)
PIDS=()

cleanup() {
  echo
  echo "Stopping MCP servers..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

wait_for_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      exec 3>&- 3<&-
      return 0
    fi
    sleep 0.5
  done
  echo "ERROR: ${name} did not come up on :${port}" >&2
  return 1
}

echo "Starting MarketMind MCP servers (PYTHON=${PY})..."
for i in "${!NAMES[@]}"; do
  echo "  - ${NAMES[$i]} on :${PORTS[$i]}"
  "$PY" -m "${MODULES[$i]}" &
  PIDS+=("$!")
done

for i in "${!NAMES[@]}"; do
  wait_for_port "${PORTS[$i]}" "${NAMES[$i]}"
  echo "  ready: ${NAMES[$i]} -> http://localhost:${PORTS[$i]}/mcp"
done

echo
echo "All three MCP servers online. Press Ctrl-C to stop."
wait
