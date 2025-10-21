# Post-Installation Steps (VM)

## You've successfully installed! 🎉

The warnings you see are **non-critical** but can be fixed:

```
✓ OK  ← Installation successful!

⚠️ torchvision==0.15 is incompatible with torch==2.5
⚠️ Ultralytics settings reset
```

## Quick Fixes

### Fix 1: PyTorch/TorchVision Compatibility (Optional)

The environment installed PyTorch 2.5 but torchvision 0.15. Update torchvision:

```bash
conda activate ball-action-spotting

# Option A: Update both (recommended)
pip install -U torch torchvision

# Option B: Just update torchvision
pip install torchvision==0.20

# Verify
python -c "import torch, torchvision; print(f'torch: {torch.__version__}, torchvision: {torchvision.__version__}')"
```

### Fix 2: Ultralytics Settings (Ignore)

This warning is harmless - Ultralytics just reset its config to defaults. No action needed.

## Verify Everything Works

```bash
conda activate ball-action-spotting

# Test all components
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('✓ YOLO OK')"
python -c "from src.player_tracking import PlayerTracker; print('✓ Player Tracking OK')"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')"

# Test MVP script
python mvp/run_mvp.py --help
```

## Quick Test with Video

```bash
# Download a test video or use your own
# python mvp/run_mvp.py /path/to/video.mp4

# Quick test on first 500 frames
python mvp/player_action_counter.py /path/to/video.mp4 \
    --max-frames 500 \
    --sample-rate 5 \
    --player 7
```

## Expected Output

When running player tracking, you should see:

```
Loading YOLO model: yolov8n.pt
✓ Player detector initialized on cuda
✓ Player action counter initialized

============================================================
Processing: video.mp4
============================================================

Step 1: Tracking players...
Tracking players in video: 500 frames @ 25.0 fps
  Processed 0/500 frames...
  Processed 100/500 frames...
  ...
✓ Tracking complete: 100 frames processed, 12 unique players

Step 2: Auto-selected most frequent player: #7
  Player visible in 95 frames
  ...

Step 4: Matching actions to player...
✓ Matched 6 actions to player

============================================================
Player #7 Action Summary
============================================================
Total Actions: 6
...
```

## Troubleshooting

### If you still see warnings:

**1. TorchVision warning:**
```bash
pip install -U torch torchvision
```

**2. Image.so undefined symbol:**
```bash
# Reinstall torchvision
pip uninstall torchvision -y
pip install torchvision --no-cache-dir
```

**3. CUDA not available:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

## All Good? Start Using It!

```bash
# Basic usage
python mvp/run_mvp.py video.mp4

# With specific player
python mvp/run_mvp.py video.mp4 7

# Advanced options
python mvp/player_action_counter.py video.mp4 \
    --player 7 \
    --yolo-model yolov8s.pt \
    --sample-rate 5 \
    --output results.json
```

## Status Check

Run this to verify everything:

```bash
conda activate ball-action-spotting

echo "=== Python ==="
python --version

echo "=== PyTorch ==="
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "=== Packages ==="
python -c "from ultralytics import YOLO; print('✓ YOLO')"
python -c "from src.player_tracking import PlayerTracker; print('✓ PlayerTracker')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo ""
echo "✓ All checks passed! Ready to use."
```

## Next Steps

1. ✅ Installation complete
2. ✅ Verify tests pass
3. 🎯 **Try with a real video!**

See `mvp/README.md` for full usage documentation.
