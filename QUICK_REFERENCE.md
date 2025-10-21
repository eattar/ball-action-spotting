# Quick Reference Card

## One-Line Commands

```bash
# Most common use cases

# 1. Interactive selection (placeholder actions)
python mvp/run_mvp.py video.mp4

# 2. Specific player (placeholder actions)
python mvp/run_mvp.py video.mp4 7

# 3. REAL MODEL - Recommended (90% accuracy)
python mvp/run_mvp.py video.mp4 7 data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth

# 4. Quick test (first 500 frames)
python mvp/player_action_counter.py video.mp4 --max-frames 500

# 5. Fast mode
python mvp/player_action_counter.py video.mp4 --sample-rate 10 --yolo-model yolov8n.pt

# 6. Accurate mode with real model
python mvp/player_action_counter.py video.mp4 --sample-rate 1 --yolo-model yolov8x.pt --action-model data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth
```

## Installation

```bash
pip install ultralytics lap
```

## Output Location

```
player_{ID}_actions.json       # Auto-generated in current directory
player_{ID}_{video_name}.json  # When using -o option
```

## Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM | `--device cpu` or `--sample-rate 10` |
| Too slow | `--yolo-model yolov8n.pt --sample-rate 10` |
| Can't find module | Run from repo root: `cd /path/to/ball-action-spotting` |
| No players found | Lower confidence in `detector.py` or try different frame |

## YOLO Model Options

| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| yolov8n.pt | 6MB | Fastest | Good | Testing/Fast processing |
| yolov8s.pt | 22MB | Fast | Better | Balanced (default) |
| yolov8m.pt | 52MB | Medium | Good | Production |
| yolov8l.pt | 87MB | Slow | Great | High accuracy |
| yolov8x.pt | 136MB | Slowest | Best | Research/Offline |

## Sample Rates

| Rate | Meaning | Use Case |
|------|---------|----------|
| 1 | Every frame | Highest accuracy, slowest |
| 5 | Every 5th (default) | Balanced |
| 10 | Every 10th | Fast processing |
| 25 | Once per second | Very fast, may miss actions |

## Git Commands

```bash
# Update repo on VM
git pull origin master

# Check changes
git status

# Commit and push
git add .
git commit -m "Test player tracking"
git push origin master
```

## File Locations

```
src/player_tracking/       # Core module
mvp/                       # MVP scripts
  ├── player_action_counter.py   # Main script
  ├── run_mvp.py                 # Simple wrapper
  ├── README.md                  # Full docs
  └── TESTING.md                 # VM setup guide
```

## Python API

```python
from src.player_tracking import PlayerTracker, ActionPlayerMatcher

# Track players
tracker = PlayerTracker(model_name='yolov8n.pt', device='cuda')
tracks = tracker.track_video('video.mp4')

# Get specific player
player_tracks = tracker.get_player_tracks(track_id=7)

# Match actions (after getting actions from model)
matcher = ActionPlayerMatcher(temporal_window=25)
matched = matcher.match_actions_to_player(actions, player_tracks)

# Get statistics
counts = matcher.count_actions(matched)
summary = matcher.create_summary(matched, track_id=7)
```

## Expected Performance

**GPU (RTX 3080-level):**
- yolov8n + sample=5: ~50 FPS
- yolov8s + sample=5: ~30 FPS
- yolov8x + sample=1: ~5 FPS

**CPU:**
- yolov8n + sample=10: ~3-5 FPS
- Not recommended for full videos

## Testing Checklist

- [ ] `git pull` successful
- [ ] `pip install ultralytics lap` done
- [ ] Can run `python mvp/run_mvp.py --help`
- [ ] Test on 500 frames works
- [ ] Player IDs show in output
- [ ] JSON file created
- [ ] Can see action summary

## Debug Commands

```bash
# Test imports
python -c "from src.player_tracking import PlayerTracker; print('OK')"

# Test YOLO
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('OK')"

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# List files
ls -la src/player_tracking/
ls -la mvp/
```

## Need Help?

1. Check `mvp/README.md` - Full documentation
2. Check `mvp/TESTING.md` - VM setup guide
3. Check `IMPLEMENTATION_SUMMARY.md` - Architecture details
4. Check error message and google it
5. Try CPU mode: `--device cpu`

---

**Ready to test? Run:**
```bash
cd /path/to/ball-action-spotting
git pull
pip install ultralytics lap
python mvp/run_mvp.py your_video.mp4
```
