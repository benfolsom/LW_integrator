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
    echo "  lw-plot-from-logcache-live --live logcache/your_sweep.log"
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

# Run the packaged launcher if available, otherwise fall back to Python module
if command -v lw-plot-latest-live >/dev/null 2>&1; then
    lw-plot-latest-live "$@"
elif [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m lw_integrator.plot_latest_live "$@"
else
    python3 -m lw_integrator.plot_latest_live "$@"
fi
