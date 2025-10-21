# Environment Setup Guide

## Quick Setup with Conda

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script (interactive)
./setup_conda_env.sh

# It will ask you to choose:
#   1) Standard (GPU, no NvDec) - Recommended
#   2) GPU Full (with NvDec) - Advanced users only
#   3) CPU Only

# Activate the environment
conda activate ball-action-spotting

# Verify installation
python -c "from src.player_tracking import PlayerTracker; print('✓ Installation OK')"
```

### Option 2: Manual Setup

**Standard GPU Setup (Recommended):**
```bash
# Create environment without NvDec (most compatible)
conda env create -f environment.yml

# Activate environment
conda activate ball-action-spotting

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('✓ YOLO OK')"
```

**CPU-Only Setup:**
```bash
conda env create -f environment-cpu.yml
conda activate ball-action-spotting
```

**Advanced: GPU with NvDec (Optional):**
```bash
# Only if you need NVIDIA hardware video decoding
# Requires: NVIDIA GPU, drivers, and build tools
conda env create -f environment-gpu-full.yml
conda activate ball-action-spotting
```

## Environment Details

- **Name**: `ball-action-spotting`
- **Python**: 3.10
- **PyTorch**: 2.0+ with CUDA support
- **Key packages**: 
  - YOLOv8 (ultralytics)
  - OpenCV
  - PyTorch
  - Video processing (NVIDIA VPF) - **Optional**

## Available Environment Files

1. **`environment.yml`** - Standard GPU setup (Recommended)
   - GPU support via CUDA
   - No NvDec (uses OpenCV for video)
   - Most compatible

2. **`environment-cpu.yml`** - CPU-only setup
   - No GPU required
   - Slower but works everywhere

3. **`environment-gpu-full.yml`** - Advanced GPU setup
   - Includes NVIDIA NvDec for hardware video decoding
   - Requires NVIDIA GPU + drivers
   - Complex build, may fail on some systems

## Updating the Environment

```bash
# Update from YAML file
conda activate ball-action-spotting
conda env update -f environment.yml --prune
```

## Alternative: Using pip only

If you don't want to use conda:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## CUDA Version Notes

The `environment.yml` specifies `cudatoolkit=11.8`. Adjust based on your system:

- **CUDA 11.8**: `cudatoolkit=11.8` (default)
- **CUDA 12.x**: `cudatoolkit=12.1`
- **CPU only**: Remove cudatoolkit line

To check your CUDA version:
```bash
nvidia-smi
```

## Troubleshooting

### "conda: command not found"
Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

### "CUDA out of memory"
- Use CPU version: Edit `environment.yml` and remove `cudatoolkit`
- Or reduce batch sizes in code

### "VideoProcessingFramework installation fails"

**Solution**: Use standard environment without NvDec

```bash
# Use the standard environment (recommended)
conda env create -f environment.yml

# NvDec is OPTIONAL - the project works fine without it
# OpenCV will be used for video decoding instead
```

**Why NvDec fails:**
- Requires NVIDIA GPU drivers
- Needs CMake 3.20+
- Complex C++ build process
- May not work on all systems

**Do you need NvDec?**
- ❌ **No** - For player tracking MVP (uses OpenCV)
- ❌ **No** - For most use cases
- ✅ **Yes** - Only if you need hardware-accelerated video decoding for the original ball-action model training

### PyTorch CUDA not available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

## Exporting Environment

To share your exact environment:

```bash
# Export with exact versions
conda env export > environment-lock.yml

# Export for cross-platform
conda env export --no-builds > environment-cross-platform.yml
```

## Removing Environment

```bash
conda deactivate
conda env remove -n ball-action-spotting
```

## VM Setup

For your VM, after pushing changes:

```bash
# On VM
cd /path/to/ball-action-spotting
git checkout feature/player-tracking-mvp
git pull

# Setup environment
./setup_conda_env.sh

# Or manually
conda env create -f environment.yml
conda activate ball-action-spotting

# Test
python mvp/run_mvp.py test_video.mp4
```

## Development

When adding new packages:

1. Add to `environment.yml` (if available in conda)
2. Or add to pip section in `environment.yml`
3. Also update `requirements.txt` for pip users
4. Test: `conda env update -f environment.yml --prune`

## Using on HPC/Cluster

Many HPC systems have conda pre-installed:

```bash
# Load conda module (if needed)
module load anaconda3

# Create environment
conda env create -f environment.yml -p ./env

# Activate
conda activate ./env

# Submit job
sbatch job_script.sh
```
