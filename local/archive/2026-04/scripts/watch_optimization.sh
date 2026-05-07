#!/bin/bash
# Compatibility wrapper for the packaged optimization monitor CLI.
# Usage: ./scripts/watch_optimization.sh [OPTIONS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
    local dir="$SCRIPT_DIR"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ] && [ -d "$dir/lw_integrator" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

REPO_ROOT="$(find_repo_root)" || {
    echo "Could not locate repository root from $SCRIPT_DIR" >&2
    exit 1
}
LOGCACHE_DIR="$REPO_ROOT/logcache"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Default options
INTERVAL=60
TOP=5
MODE="normal"
LATEST_ONLY=""
SPECIFIC_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -n|--top)
            TOP="$2"
            shift 2
            ;;
        -c|--compact)
            MODE="compact"
            shift
            ;;
        --once)
            MODE="once"
            shift
            ;;
        -l|--latest)
            LATEST_ONLY="--latest"
            shift
            ;;
        -r|--run)
            SPECIFIC_RUN="--run $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Monitor optimization progress in logcache/"
            echo ""
            echo "Options:"
            echo "  -i, --interval SECONDS   Update interval (default: 60)"
            echo "  -n, --top N             Number of top results (default: 5)"
            echo "  -c, --compact           Compact one-line output"
            echo "  --once                  Run once and exit"
            echo "  -l, --latest            Monitor only latest/current run"
            echo "  -r, --run FILE          Monitor specific run (partial filename)"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                      # Monitor with defaults (60s interval, top 5)"
            echo "  $0 -i 30 -n 10          # Update every 30s, show top 10"
            echo "  $0 -c                   # Compact mode"
            echo "  $0 --once               # Run once"
            echo "  $0 --latest             # Monitor only latest run"
            echo "  $0 --run 20260219       # Monitor specific run by date"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD=(
    python3
    -m
    lw_integrator.optimization_monitor
    --logcache
    "$LOGCACHE_DIR"
    --interval
    "$INTERVAL"
    --top
    "$TOP"
)

if [ -n "$LATEST_ONLY" ]; then
    CMD+=("$LATEST_ONLY")
fi

if [ -n "$SPECIFIC_RUN" ]; then
    CMD+=(--run "${SPECIFIC_RUN#--run }")
fi

if [ "$MODE" = "compact" ]; then
    CMD+=(--compact)
elif [ "$MODE" = "once" ]; then
    CMD+=(--once)
fi

# Run the monitor
echo "🔬 Starting Optimization Monitor..."
echo "Press Ctrl+C to stop"
echo ""
exec "${CMD[@]}"
