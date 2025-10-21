#!/bin/bash
# Quick one-line setup after VM restart
# Usage: source quick_setup.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate ball-action-spotting

# Go to project directory
cd /workspace/ball-action-spotting

# Show status
echo "✅ Environment: ball-action-spotting (activated)"
echo "✅ Directory: $(pwd)"
echo "✅ Ready to run: python mvp/run_mvp.py --help"
