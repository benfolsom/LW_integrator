#!/bin/bash
# Quick launcher for live plotting of the latest sweep

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Live Energy Gain Plot - Latest Sweep Monitor          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Find the latest sweep log (either GUI *_sweep.log or CLI *_sweep_cli.log)
LATEST_LOG=$(ls -t logcache/*_sweep*.log 2>/dev/null | head -n 1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No sweep logs found in logcache/"
    echo ""
    echo "Please run a sweep first or specify a log file:"
    echo "  ./plot_from_logcache_live.py --live logcache/your_sweep.log"
    exit 1
fi

echo "📊 Latest sweep log: $LATEST_LOG"
echo ""
echo "Starting live monitor..."
echo "  - Plot will auto-update every 3 seconds"
echo "  - Press Ctrl+C to stop"
echo ""
echo "────────────────────────────────────────────────────────────────"
echo ""

# Run the live plotter with venv Python if available, otherwise use system Python
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python ./plot_from_logcache_live.py --live "$LATEST_LOG" "$@"
else
    ./plot_from_logcache_live.py --live "$LATEST_LOG" "$@"
fi
