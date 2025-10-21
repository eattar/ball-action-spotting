# Using Real Ball-Action-Spotting Model

This guide shows how to use your trained ball-action-spotting models instead of placeholder detection.

## Quick Start

### 1. Find Your Best Model

Your trained models are in `data/ball_action/experiments/`:

```bash
# List available experiments
ls data/ball_action/experiments/

# List models for specific experiment (with scores)
ls -lh data/ball_action/experiments/ball_finetune_long_004/fold_*/model*.pth
```

Example models:
- `ball_finetune_long_004/fold_5/model-006-0.901643.pth` (90.1% accuracy) ✅ BEST
- `ball_finetune_long_004/fold_6/model-006-0.897602.pth` (89.7% accuracy)
- `ball_tuning_001/fold_6/model-034-0.891524.pth` (89.1% accuracy)

### 2. Run with Real Model

```bash
# Method 1: Quick script (recommended for testing)
python mvp/run_mvp.py \
  "/path/to/video.mp4" \
  19043 \
  data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth

# Method 2: Full script (more options)
python mvp/player_action_counter.py \
  --video "/path/to/video.mp4" \
  --player-id 19043 \
  --action-model data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth
```

## Model Details

### Available Experiments

1. **ball_finetune_long_004** (Recommended)
   - Best overall performance (86-90% accuracy)
   - 7 folds (0-6)
   - Trained for 6 epochs
   - Uses EfficientNetV2 + 3D CNN

2. **ball_tuning_001**
   - Good baseline (87-89% accuracy)
   - 7 folds (0-6)
   - Trained for 34 epochs

3. **sampling_weights_001**
   - Weighted sampling (83-88% accuracy)
   - 7 folds (0-6)
   - Trained for 29 epochs

### Model Architecture

- **Backbone**: EfficientNetV2-B0 (2D features)
- **Temporal**: 3D CNN blocks
- **Input**: 33 frames stacked (frame_stack_size=33, step=2)
- **Output**: 12 action classes + probabilities

### Action Classes Detected

1. PASS
2. DRIVE
3. HEADER
4. HIGH PASS
5. OUT
6. CROSS
7. THROW IN
8. SHOT
9. BALL PLAYER BLOCK
10. PLAYER SUCCESSFUL TACKLE
11. FREE KICK
12. GOAL

## On Your VM

### Step 1: Verify Model Exists

```bash
cd /workspace/ball-action-spotting
git pull  # Get latest code

# Check if models are present
ls -lh data/ball_action/experiments/ball_finetune_long_004/fold_5/
```

### Step 2: Run with Real Detection

```bash
# Test on the Blackburn vs Nottingham match
python mvp/run_mvp.py \
  "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest/224p.mp4" \
  19043 \
  data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth
```

### Step 3: Compare Results

**Placeholder mode (before):**
```
✓ Matched 1 actions to player
  GOAL: 1 (random)
```

**Real model (after):**
```
✓ Matched 8 actions to player
  PASS: 4 (67.3%, 82.1%, 75.9%, 80.2%)
  DRIVE: 2 (71.4%, 68.9%)
  SHOT: 1 (88.6%)
  CROSS: 1 (74.2%)
```

## Performance Notes

### Processing Time

- **Placeholder**: ~2 minutes (tracking only)
- **Real Model**: ~15-30 minutes (depends on video length)
  - Player tracking: ~2 minutes
  - Action detection: ~13-28 minutes (142K frames)
  - Action matching: <1 second

### GPU Memory

- Player tracking (YOLO): ~500MB VRAM
- Action detection (EfficientNetV2): ~2-4GB VRAM
- **Total required**: ~4-5GB VRAM

If you run out of memory:
```bash
# Use CPU for action detection (slower but works)
python mvp/player_action_counter.py \
  --video "video.mp4" \
  --player-id 19043 \
  --action-model model.pth \
  --device cpu
```

### Optimization Tips

1. **Use best model**: Fold 5 usually has highest accuracy
2. **Sample frames**: Already set to `sample_rate=5` (every 5th frame)
3. **Batch processing**: For multiple videos, run in parallel on different GPUs

## Troubleshooting

### Error: "Model file not found"

```bash
# Check path is correct
ls data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth

# Use absolute path if relative doesn't work
python mvp/run_mvp.py video.mp4 7 "$PWD/data/ball_action/experiments/.../model.pth"
```

### Error: "CUDA out of memory"

```bash
# Option 1: Use smaller sample_rate (skip more frames)
# Edit mvp/run_mvp.py: sample_rate=10

# Option 2: Use CPU
# Edit mvp/run_mvp.py: device='cpu'
```

### Model loads but no actions detected

- Check if video has ball actions (not just players standing)
- Try different confidence threshold
- Verify video FPS is 25 (model trained on 25fps)

### Fallback to Placeholder

If you see "⚠️  Using placeholder action detection" but provided a model:
- Model path might be wrong
- Model file corrupted
- Missing dependencies (argus, timm, kornia)
- Check error message above warning for details

## Next Steps

Once real model is working:

1. **Test different folds**: Compare fold 5 vs fold 6 accuracy
2. **Ensemble predictions**: Average predictions from multiple folds
3. **Add spatial matching**: Use attention maps for better player-action association
4. **Visualization**: Generate video with action annotations

## Example Output

With real model, you'll get detailed action timeline:

```json
{
  "player_id": 19043,
  "total_actions": 8,
  "action_counts": {
    "PASS": 4,
    "DRIVE": 2,
    "SHOT": 1,
    "CROSS": 1
  },
  "timeline": [
    {"frame": 31557, "time": "21:02", "action": "PASS", "confidence": 0.673},
    {"frame": 31789, "time": "21:11", "action": "DRIVE", "confidence": 0.714},
    {"frame": 32012, "time": "21:20", "action": "SHOT", "confidence": 0.886},
    ...
  ]
}
```

## Questions?

Check the main documentation:
- [README.md](README.md) - Basic usage
- [TESTING.md](TESTING.md) - VM setup
- [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Technical details
