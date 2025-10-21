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

# Ask user which environment to create
echo ""
echo "Select environment type:"
echo "  1) Standard (GPU, no NvDec) - Recommended for most users"
echo "  2) GPU Full (with NvDec) - Requires NVIDIA drivers + complex setup"
echo "  3) CPU Only - No GPU required"
echo ""
read -p "Enter choice [1-3] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        ENV_FILE="environment.yml"
        echo "Using standard GPU environment (without NvDec)"
        ;;
    2)
        ENV_FILE="environment-gpu-full.yml"
        echo "Using full GPU environment (with NvDec)"
        echo "⚠️  Warning: This requires NVIDIA GPU drivers and may fail on some systems"
        ;;
    3)
        ENV_FILE="environment-cpu.yml"
        echo "Using CPU-only environment"
        ;;
    *)
        echo "Invalid choice. Using default (standard GPU)"
        ENV_FILE="environment.yml"
        ;;
esac

echo "Environment file: $ENV_FILE"

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
echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

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
