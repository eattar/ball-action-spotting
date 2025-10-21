#!/bin/bash
# Setup conda environment for ball-action-spotting

set -e  # Exit on error

echo "=================================================="
echo "Ball Action Spotting - Environment Setup"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"

# Remove existing environment if it exists
if conda env list | grep -q "ball-action-spotting"; then
    echo ""
    read -p "Environment 'ball-action-spotting' already exists. Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ball-action-spotting -y
    else
        echo "Keeping existing environment. Updating instead..."
        conda env update -f environment.yml --prune
        echo "✓ Environment updated"
        exit 0
    fi
fi

# Create environment from YAML file
echo ""
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=================================================="
echo "✓ Environment setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ball-action-spotting"
echo ""
echo "To test the installation:"
echo "  conda activate ball-action-spotting"
echo "  python -c 'from src.player_tracking import PlayerTracker; print(\"✓ Installation OK\")'"
echo ""
