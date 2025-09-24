#!/bin/bash
# Virtual Environment Setup Script for LW_integrator
# =====================================================
# This script sets up the virtual environment with numba-compatible packages
# for optimal performance of the Lienard-Wiechert electromagnetic integrator.

echo "🐍 Setting up LW_integrator Virtual Environment"
echo "================================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core scientific packages
echo "Installing core packages..."
pip install numpy scipy matplotlib

# Install numba for JIT compilation (critical for performance)
echo "Installing numba for JIT compilation..."
pip install numba

# Verify numba compatibility
echo "Verifying numba installation..."
python -c "
import numba
import numpy
print(f'✅ NumPy version: {numpy.__version__}')
print(f'✅ Numba version: {numba.__version__}')
from numba import jit
print('✅ Numba JIT compilation available!')
"

# Install additional useful packages
echo "Installing additional packages..."
pip install pytest ipython jupyter

echo ""
echo "🎉 Virtual environment setup complete!"
echo "======================================"
echo ""
echo "To activate this environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify numba is working:"
echo "  python -c 'import numba; print(\"Numba available:\", numba.__version__)'"
echo ""
echo "📝 IMPORTANT: Always use this setup to ensure numba compatibility!"
echo "   - NumPy 2.x is supported with Numba 0.62+"
echo "   - JIT compilation provides ~10-100x speedup for EM calculations"
echo "   - If numba warnings appear, check version compatibility"
