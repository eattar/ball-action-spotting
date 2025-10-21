# Quick Fix: NvDec Installation Error

If you encountered the VideoProcessingFramework (NvDec) installation error, follow this guide.

## The Error

```
ERROR: Failed building wheel for PyNvCodec
CMake Error... Compatibility with CMake < 3.5 has been removed
```

## Quick Solution: Use Standard Environment (Recommended)

NvDec is **optional** for the player tracking MVP. Use the standard environment instead:

```bash
# Remove the failed environment
conda env remove -n ball-action-spotting

# Create standard environment (without NvDec)
conda env create -f environment.yml

# Activate
conda activate ball-action-spotting

# Test - should work now!
python mvp/run_mvp.py --help
```

## What Changed?

The standard `environment.yml` now:
- ✅ Has GPU support (CUDA)
- ✅ Has all player tracking dependencies
- ✅ Uses OpenCV for video (not NvDec)
- ❌ Does NOT include VideoProcessingFramework

## Do You Need NvDec?

**You DON'T need NvDec if:**
- Using player tracking MVP ✓
- Running predictions on existing models ✓
- Processing videos at normal speed ✓

**You MIGHT need NvDec if:**
- Training new ball-action models ⚠️
- Need hardware-accelerated video decoding ⚠️
- Processing thousands of videos ⚠️

## Advanced: Installing NvDec (Optional)

Only attempt if you really need it:

### Prerequisites
```bash
# Check NVIDIA driver
nvidia-smi

# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build

# Or on conda
conda install cmake>=3.20 ninja gcc_linux-64 gxx_linux-64
```

### Install NvDec Separately
```bash
# After creating standard environment
conda activate ball-action-spotting

# Try installing NvDec manually
pip install git+https://github.com/NVIDIA/VideoProcessingFramework@v2.0.0
pip install git+https://github.com/NVIDIA/VideoProcessingFramework@v2.0.0#subdirectory=src/PytorchNvCodec

# If it fails, you can still use the project without it
```

### Alternative: Use environment-gpu-full.yml
```bash
# Remove failed environment
conda env remove -n ball-action-spotting

# Try full GPU environment with build tools
conda env create -f environment-gpu-full.yml

# This includes CMake 3.20+ and build tools
# But may still fail if drivers are incompatible
```

## Verification

Test your installation:

```bash
conda activate ball-action-spotting

# Test core functionality
python -c "import torch; print(f'PyTorch: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
python -c "from src.player_tracking import PlayerTracker; print('Player Tracking: OK')"

# Test player tracking (doesn't need NvDec)
python mvp/run_mvp.py --help
```

## Summary

| Environment | NvDec | Use Case | Recommendation |
|-------------|-------|----------|----------------|
| `environment.yml` | ❌ No | Player tracking, predictions | ✅ **Use this** |
| `environment-cpu.yml` | ❌ No | CPU-only systems | ✅ Use if no GPU |
| `environment-gpu-full.yml` | ✅ Yes | Advanced, training | ⚠️ Optional |

**Bottom line**: The standard environment (`environment.yml`) is all you need for the player tracking MVP!
