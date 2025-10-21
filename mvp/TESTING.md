# Testing on Virtual Machine

## Quick Setup Guide

### 1. Pull Latest Changes from Git

On your VM:

```bash
cd /path/to/ball-action-spotting
git pull origin master
```

### 2. Install New Dependencies

```bash
# Activate your virtual environment if you have one
# source venv/bin/activate

# Install new packages
pip install ultralytics>=8.0.0
pip install lap>=0.4.0

# Or reinstall all requirements
pip install -r requirements.txt
```

### 3. Download YOLO Model (First Time Only)

The YOLO model will auto-download on first run, but you can pre-download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This downloads ~6MB model to `~/.cache/ultralytics/`

### 4. Test with Sample Video

```bash
# Quick test (interactive player selection)
python mvp/run_mvp.py path/to/test_video.mp4

# Or specify player ID
python mvp/run_mvp.py path/to/test_video.mp4 7

# Test on first 500 frames only (faster)
python mvp/player_action_counter.py path/to/test_video.mp4 \
    --max-frames 500 \
    --sample-rate 5
```

### 5. Expected Output

```
Loading YOLO model: yolov8n.pt
✓ Player detector initialized on cuda
Initializing player tracker...
✓ Player action counter initialized

============================================================
Processing: test_video.mp4
============================================================

Step 1: Tracking players...
Tracking players in video: 500 frames @ 25.0 fps
  Processed 0/500 frames...
  Processed 100/500 frames...
  ...
✓ Tracking complete: 100 frames processed, 12 unique players

Step 2: Auto-selected most frequent player: #7
  Player visible in 95 frames
  First appearance: frame 10
  Last appearance: frame 487
  Avg confidence: 0.89

Step 3: Detecting ball actions...
  ⚠️  Using placeholder action detection (integrate real model later)
  Detected 10 total actions in video

Step 4: Matching actions to player...
Matching 10 actions to player visible in 95 frames...
✓ Matched 6 actions to player (removed 0 duplicates)

============================================================
Player #7 Action Summary
============================================================
...
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"

**Solution 1**: Use CPU mode
```bash
python mvp/run_mvp.py video.mp4 --device cpu
```

**Solution 2**: Use nano model + higher sample rate
```bash
python mvp/player_action_counter.py video.mp4 \
    --yolo-model yolov8n.pt \
    --sample-rate 10 \
    --max-frames 500
```

### Issue: "ultralytics not installed"

```bash
pip install ultralytics
```

### Issue: "No module named 'cv2'"

OpenCV should already be in requirements.txt, but if missing:
```bash
pip install opencv-python
```

### Issue: "ImportError: cannot import PlayerTracker"

Make sure you're running from the project root:
```bash
cd /path/to/ball-action-spotting
python mvp/run_mvp.py video.mp4
```

## Git Workflow

### After Testing, Commit Your Results

```bash
# Check what changed
git status

# Add all new files
git add .

# Commit
git commit -m "Add player tracking MVP module"

# Push to remote
git push origin master
```

### If You Want to Make Changes

```bash
# Create a branch
git checkout -b feature/player-tracking-improvements

# Make changes, test, commit
git add .
git commit -m "Improve player tracking accuracy"

# Push branch
git push origin feature/player-tracking-improvements

# Then merge on GitHub or:
git checkout master
git merge feature/player-tracking-improvements
git push origin master
```

## File Structure Added

```
ball-action-spotting/
├── requirements.txt           # ✏️ UPDATED (added ultralytics, lap)
├── src/
│   └── player_tracking/       # 🆕 NEW MODULE
│       ├── __init__.py
│       ├── detector.py
│       ├── tracker.py
│       └── matcher.py
└── mvp/                       # 🆕 NEW DIRECTORY
    ├── README.md
    ├── player_action_counter.py
    ├── run_mvp.py
    └── TESTING.md             # (this file)
```

## Testing Checklist

- [ ] Git pull successful
- [ ] Dependencies installed (ultralytics, lap)
- [ ] YOLO model downloaded
- [ ] Can run `python mvp/run_mvp.py --help`
- [ ] Test on short video clip (500 frames)
- [ ] Player tracking works (shows player IDs)
- [ ] Action detection runs (placeholder data)
- [ ] Results JSON file created
- [ ] Can see summary and timeline output

## Next Steps After MVP Works

1. **Integrate real action model**
   - Load your trained ball-action-spotting model
   - Replace placeholder in `_detect_actions()`

2. **Test on full match video**
   - Remove `--max-frames` limit
   - Use appropriate `--sample-rate`

3. **Optimize performance**
   - Benchmark different YOLO models
   - Tune sample rate vs accuracy

4. **Add visualization**
   - Generate annotated video output
   - Show player tracks + actions

## Quick Test Commands

```bash
# Minimal test
python mvp/run_mvp.py video.mp4 7

# With all options
python mvp/player_action_counter.py video.mp4 \
    --player 7 \
    --yolo-model yolov8n.pt \
    --sample-rate 5 \
    --max-frames 500 \
    --output test_results.json \
    --device cuda

# Check results
cat test_results.json | head -50
```

## Performance Benchmarks (Approximate)

| Configuration | Speed | Accuracy | Use Case |
|--------------|-------|----------|----------|
| yolov8n, sample=10, CPU | ~10 FPS | Medium | Quick testing |
| yolov8n, sample=5, GPU | ~50 FPS | Medium | Fast processing |
| yolov8s, sample=5, GPU | ~30 FPS | Good | Balanced |
| yolov8x, sample=1, GPU | ~5 FPS | Best | High accuracy |

## Questions?

If something doesn't work:
1. Check error message
2. Look in `mvp/README.md` for solutions
3. Check that all files are present: `ls -la src/player_tracking/` `ls -la mvp/`
4. Verify imports work: `python -c "from src.player_tracking import PlayerTracker; print('OK')"`

Good luck testing! 🚀
