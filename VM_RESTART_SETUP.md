# VM Container Restart Setup Guide

Quick guide to restore your environment after container restart.

## Quick Start (30 seconds)

```bash
# 1. Navigate to project
cd /workspace/ball-action-spotting

# 2. Activate conda environment
conda activate ball-action-spotting

# 3. Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"

# 4. Run your script
python mvp/run_mvp.py --help
```

That's it! Your environment should be ready.

---

## Step-by-Step (If Environment Missing)

### Check What's Available

```bash
# Check if conda is installed
which conda
conda env list

# Check if ball-action-spotting env exists
conda env list | grep ball-action-spotting
```

### Scenario 1: Conda Environment Exists ✅

```bash
# Just activate it
conda activate ball-action-spotting

# Verify it works
python -c "import torch; print(torch.__version__)"
```

### Scenario 2: Conda Installed, But Environment Missing 🔧

```bash
cd /workspace/ball-action-spotting

# Recreate environment (5-10 minutes)
conda env create -f environment.yml

# Activate
conda activate ball-action-spotting

# Verify
python -c "import torch, cv2, ultralytics; print('All imports OK')"
```

### Scenario 3: Conda Not Installed ⚙️

```bash
# Install Miniconda (if needed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda for shell
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment
cd /workspace/ball-action-spotting
conda env create -f environment.yml
conda activate ball-action-spotting
```

---

## One-Line Auto-Setup Script

Create a script that runs on container start:

```bash
# Create auto-setup script
cat > /workspace/ball-action-spotting/vm_setup.sh << 'EOF'
#!/bin/bash
# VM Container Restart Auto-Setup

echo "🔧 Setting up ball-action-spotting environment..."

# Navigate to project
cd /workspace/ball-action-spotting || exit 1

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda first."
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "ball-action-spotting"; then
    echo "✅ Environment found: ball-action-spotting"
else
    echo "📦 Creating environment (this may take 5-10 minutes)..."
    conda env create -f environment.yml
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ball-action-spotting

# Verify setup
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || echo "❌ PyTorch import failed"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" || echo "❌ OpenCV import failed"
python -c "from ultralytics import YOLO; print('✅ YOLO/Ultralytics')" || echo "❌ Ultralytics import failed"

echo ""
echo "🎉 Setup complete! Environment ready."
echo ""
echo "To use:"
echo "  conda activate ball-action-spotting"
echo "  python mvp/run_mvp.py --help"
EOF

chmod +x /workspace/ball-action-spotting/vm_setup.sh
```

### Run Auto-Setup

```bash
# Run the setup script
bash /workspace/ball-action-spotting/vm_setup.sh

# Or make it automatic on shell startup (optional)
echo 'bash /workspace/ball-action-spotting/vm_setup.sh' >> ~/.bashrc
```

---

## Data Persistence Check

Your container should persist data in `/workspace` and `/netscratch`. Verify:

```bash
# Check if your data is still there
ls /workspace/ball-action-spotting/data/ball_action/experiments/
ls /netscratch/eattar/ds/SoccetNet/

# Check git status
cd /workspace/ball-action-spotting
git status
git branch
```

If data is missing, you may need to:
1. Re-clone repository: `git clone https://github.com/eattar/ball-action-spotting.git`
2. Re-download models/datasets

---

## Quick Test After Restart

```bash
# Activate environment
conda activate ball-action-spotting

# Quick 100-frame test (should take ~10 seconds)
cd /workspace/ball-action-spotting
python mvp/run_mvp.py \
  "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest/224p.mp4" \
  19043 \
  --max-frames 100

# If that works, you're good to go! 🎉
```

---

## Common Issues After Restart

### Issue 1: "conda: command not found"

```bash
# Re-initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Or add to ~/.bashrc permanently
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: "Environment doesn't exist"

```bash
# Recreate it
cd /workspace/ball-action-spotting
conda env create -f environment.yml
```

### Issue 3: "CUDA not available"

```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda activate ball-action-spotting
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: "Repository not found"

```bash
# Re-clone
cd /workspace
git clone https://github.com/eattar/ball-action-spotting.git
cd ball-action-spotting
git checkout feature/player-tracking-mvp
```

### Issue 5: "Model files missing"

Models should be in `/workspace/ball-action-spotting/data/`. If missing:
- Check if `/workspace` is mounted correctly
- May need to re-download or copy from backup

---

## Permanent Setup (Recommended)

Add to your `~/.bashrc` for automatic activation:

```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Auto-activate ball-action-spotting environment
if [ -d "/workspace/ball-action-spotting" ]; then
    # Initialize conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        
        # Auto-activate environment
        conda activate ball-action-spotting 2>/dev/null || echo "Run: conda env create -f environment.yml"
    fi
    
    # Set working directory
    cd /workspace/ball-action-spotting
    
    echo "🎯 ball-action-spotting environment ready!"
    echo "   Run: python mvp/run_mvp.py --help"
fi
EOF

# Reload
source ~/.bashrc
```

Now every time you log in, you'll automatically:
1. Be in the right directory
2. Have conda environment activated
3. Be ready to run scripts

---

## Minimal Restart Checklist

- [ ] `conda activate ball-action-spotting`
- [ ] `cd /workspace/ball-action-spotting`
- [ ] Test: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Test: `python mvp/run_mvp.py --help`
- [ ] Ready to run! ✅

---

## Full Environment Recreation (If Needed)

If everything is broken, start fresh:

```bash
# 1. Remove old environment
conda deactivate
conda env remove -n ball-action-spotting

# 2. Update repository
cd /workspace/ball-action-spotting
git pull origin feature/player-tracking-mvp

# 3. Recreate environment
conda env create -f environment.yml

# 4. Activate and verify
conda activate ball-action-spotting
python -c "import torch, cv2, ultralytics; print('✅ All OK')"

# 5. Test run
python mvp/run_mvp.py --help
```

Time: ~5-10 minutes total

---

## Container Startup Script

If you have access to container configuration, add this to run on startup:

```dockerfile
# Add to Dockerfile or startup script
RUN echo 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ball-action-spotting' >> ~/.bashrc
WORKDIR /workspace/ball-action-spotting
```

---

## Questions?

If something doesn't work:
1. Check `nvidia-smi` - GPU should be visible
2. Check `conda env list` - Environment should exist
3. Check `/workspace/ball-action-spotting/` - Repository should exist
4. Run `bash vm_setup.sh` - Auto-fixes most issues

For fresh start: Just run `conda env create -f environment.yml` (5-10 min)
