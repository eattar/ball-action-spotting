#!/bin/bash
# VM Container Restart Auto-Setup Script
# Run this after container restart to restore environment

set -e  # Exit on error

echo "============================================================"
echo "🔧 Ball-Action-Spotting VM Auto-Setup"
echo "============================================================"
echo ""

# Navigate to project
if [ -d "/workspace/ball-action-spotting" ]; then
    cd /workspace/ball-action-spotting
    echo "✅ Project directory: /workspace/ball-action-spotting"
else
    echo "❌ Project directory not found: /workspace/ball-action-spotting"
    echo "   You may need to re-clone the repository:"
    echo "   git clone https://github.com/eattar/ball-action-spotting.git /workspace/ball-action-spotting"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found"
    echo ""
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    echo "✅ Miniconda installed"
fi

# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "⚠️  Could not find conda.sh, trying direct conda command..."
fi

# Check if environment exists
echo ""
if conda env list | grep -q "^ball-action-spotting "; then
    echo "✅ Environment found: ball-action-spotting"
else
    echo "📦 Creating environment (this may take 5-10 minutes)..."
    conda env create -f environment.yml
    echo "✅ Environment created"
fi

# Activate environment
echo ""
echo "🔄 Activating environment..."
conda activate ball-action-spotting

# Verify setup
echo ""
echo "🔍 Verifying installation..."
echo "------------------------------------------------------------"

# Check PyTorch
if python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
    true
else
    echo "❌ PyTorch import failed"
    exit 1
fi

# Check OpenCV
if python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" 2>/dev/null; then
    true
else
    echo "❌ OpenCV import failed"
    exit 1
fi

# Check Ultralytics
if python -c "from ultralytics import YOLO; print('✅ Ultralytics/YOLO')" 2>/dev/null; then
    true
else
    echo "❌ Ultralytics import failed"
    exit 1
fi

# Check Argus
if python -c "import argus; print('✅ Argus')" 2>/dev/null; then
    true
else
    echo "⚠️  Argus import failed (needed for real model)"
fi

# Check custom modules
if python -c "from src.player_tracking import PlayerTracker; print('✅ Player tracking module')" 2>/dev/null; then
    true
else
    echo "❌ Player tracking module import failed"
    exit 1
fi

# Check GPU
echo ""
echo "🎮 GPU Status:"
echo "------------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
else
    echo "⚠️  nvidia-smi not available (CPU mode only)"
fi

# Check data
echo ""
echo "💾 Data Status:"
echo "------------------------------------------------------------"
if [ -d "data/ball_action/experiments/ball_finetune_long_004" ]; then
    MODEL_COUNT=$(find data/ball_action/experiments -name "*.pth" 2>/dev/null | wc -l)
    echo "✅ Models found: $MODEL_COUNT .pth files"
    echo "   Best model: data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth"
else
    echo "⚠️  Model data not found in data/ball_action/experiments/"
    echo "   Placeholder mode will be used"
fi

# Git status
echo ""
echo "📂 Repository Status:"
echo "------------------------------------------------------------"
BRANCH=$(git branch --show-current)
echo "   Branch: $BRANCH"
echo "   Latest commit: $(git log -1 --oneline)"

echo ""
echo "============================================================"
echo "🎉 Setup Complete!"
echo "============================================================"
echo ""
echo "Environment is ready to use:"
echo ""
echo "  • Conda environment: ball-action-spotting (activated)"
echo "  • Working directory: $(pwd)"
echo "  • Branch: $BRANCH"
echo ""
echo "Quick test:"
echo "  python mvp/run_mvp.py --help"
echo ""
echo "Full run (placeholder mode):"
echo "  python mvp/run_mvp.py video.mp4 7"
echo ""
echo "Full run (real model):"
echo "  python mvp/run_mvp.py video.mp4 7 data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth"
echo ""
echo "============================================================"
