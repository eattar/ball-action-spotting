#!/bin/bash
echo "Fixing PyTorch/torchvision compatibility..."
conda activate ball-action-spotting
pip install -U torch torchvision --force-reinstall
echo "✓ Fix applied"
