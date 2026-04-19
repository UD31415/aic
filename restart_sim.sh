#!/usr/bin/env bash
# restart_sim.sh
# Kills all running AIC simulation processes and starts a fresh simulation.
#
# Usage: ./restart_sim.sh [POLICY]
#   POLICY  Python class path for aic_model (default: aic_example_policies.ros.WaveArm)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${1:-aic_example_policies.ros.WaveArm}"

log()  { echo "[sim] $*"; }
warn() { echo "[sim] WARN: $*"; }

# ─── Clean up temp scripts from previous runs ────────────────────────────────
rm -f /tmp/aic_sim_*.sh 2>/dev/null || true

# ─── Kill existing simulation processes ──────────────────────────────────────
log "Stopping simulation..."

pkill -f "ros2 run aic_model"      2>/dev/null && log "  killed: aic_model node"      || true
pkill -f "pixi.*ros2.*aic"         2>/dev/null && log "  killed: pixi ros2 process"   || true
pkill -f "ros2 launch aic_bringup" 2>/dev/null && log "  killed: aic_bringup launch"  || true
pkill -f "gz sim"                  2>/dev/null && log "  killed: gazebo"               || true
pkill -f "rviz2"                   2>/dev/null && log "  killed: rviz2"                || true
pkill -f "rmw_zenohd"              2>/dev/null && log "  killed: zenoh router"         || true

export DBX_CONTAINER_MANAGER=docker
if sudo docker stop aic_eval 2>/dev/null; then
    log "  stopped: aic_eval container"
else
    log "  aic_eval was not running"
fi

sleep 2

# ─── Terminal helper ─────────────────────────────────────────────────────────
# Writes cmd to a temp script and opens it in a new terminal window.
open_terminal() {
    local title="$1"
    local cmd="$2"
    local tmp
    tmp=$(mktemp /tmp/aic_sim_XXXXXX.sh)
    printf '#!/bin/bash\nexport PATH="/home/administrato/.pixi/bin:$PATH"\n%s\nexec bash\n' "$cmd" > "$tmp"
    chmod +x "$tmp"

    if [ -x /usr/bin/terminator ]; then
        /usr/bin/terminator --title="$title" -e "$tmp" &
    elif command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$title" -- bash "$tmp" &
    elif command -v xterm &>/dev/null; then
        xterm -T "$title" -e "$tmp" &
    else
        warn "No terminal emulator found. Run manually in a new terminal:"
        warn "  $cmd"
        rm -f "$tmp"
        return 1
    fi
}

# ─── Start eval container (Gazebo + aic_engine) ───────────────────────────────
log "Starting eval container..."
open_terminal "AIC Eval" \
"export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true"

# ─── Wait for Zenoh router on port 7447 ──────────────────────────────────────
log "Waiting for Zenoh router (port 7447)..."
WAITED=0
until bash -c 'echo > /dev/tcp/localhost/7447' 2>/dev/null; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 60 ]; then
        warn "Zenoh router not up after 60s — starting model anyway."
        break
    fi
done
[ "$WAITED" -lt 60 ] && log "Zenoh router ready after ${WAITED}s."

# ─── Start aic_model node ─────────────────────────────────────────────────────
log "Starting aic_model (policy: $POLICY)..."
open_terminal "AIC Model" \
"cd \"$SCRIPT_DIR\"
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=$POLICY"

log "Done."
log "  AIC Eval  — Gazebo + aic_engine running inside distrobox"
log "  AIC Model — aic_model node, policy: $POLICY"
log ""
log "To use a different policy: $0 <policy.module.ClassName>"
