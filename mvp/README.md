# Player Action Tracking MVP

Track specific players in soccer videos and count their ball actions.

## Features

- ✅ **Player Detection & Tracking**: Uses YOLOv8 to detect and track players across frames
- ✅ **Action Detection**: Detects 12 types of ball actions (PASS, SHOT, DRIVE, etc.)
- ✅ **Action-Player Matching**: Associates detected actions with specific tracked players
- ✅ **Action Statistics**: Counts and summarizes actions per player

## Quick Start

### 1. Installation

```bash
# Install new dependencies
pip install ultralytics lap

# Or install all requirements
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Interactive mode - select player from first frame
python mvp/run_mvp.py path/to/video.mp4

# Specify player ID directly
python mvp/run_mvp.py path/to/video.mp4 7

# Advanced usage with more options
python mvp/player_action_counter.py path/to/video.mp4 \
    --player 7 \
    --yolo-model yolov8s.pt \
    --sample-rate 5 \
    --output results.json
```

### 3. Example Output

```
============================================================
Player #7 Action Summary
============================================================

Total Actions: 15

Action Breakdown:
  PASS                     :  8 ( 53.3%)
  DRIVE                    :  3 ( 20.0%)
  SHOT                     :  2 ( 13.3%)
  CROSS                    :  2 ( 13.3%)

Temporal Range:
  First action: 02:25 (frame 145)
  Last action:  47:28 (frame 2847)
  Duration:     45:03
  Rate:         0.3 actions/min

Action Timeline:
------------------------------------------------------------
   1. [02:25] PASS                 (87.3%)
   2. [04:12] DRIVE                (92.1%)
   3. [06:45] PASS                 (78.9%)
   ...

✓ Results saved to: player_7_actions.json
```

## Architecture

```
Video → YOLO Detection → ByteTrack → Action Detection → Matching → Statistics
         (players)        (track IDs)   (ball actions)    (associate)  (count)
```

## Components

### 1. Player Tracking (`src/player_tracking/`)

- **`detector.py`**: Player detection using YOLOv8
- **`tracker.py`**: Multi-object tracking across frames
- **`matcher.py`**: Action-player association logic

### 2. MVP Scripts (`mvp/`)

- **`player_action_counter.py`**: Main integration pipeline
- **`run_mvp.py`**: Simple CLI wrapper for quick testing

## Command-Line Options

```bash
python mvp/player_action_counter.py --help

Arguments:
  video                Path to video file

Options:
  --player INT         Player ID to track (optional)
  --interactive, -i    Show frame to select player interactively
  --yolo-model STR     YOLO model (default: yolov8n.pt)
                       Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt,
                                yolov8l.pt, yolov8x.pt (most accurate)
  --action-model PATH  Path to ball action model (optional)
  --output, -o PATH    Output JSON file path
  --max-frames INT     Maximum frames to process (for testing)
  --sample-rate INT    Process every Nth frame (default: 5)
  --device STR         Device: cuda or cpu (default: cuda)
```

## Output Format

Results are saved as JSON:

```json
{
  "video": "match.mp4",
  "player_id": 7,
  "track_statistics": {
    "track_id": 7,
    "num_frames": 450,
    "first_frame": 10,
    "last_frame": 2847,
    "avg_confidence": 0.87
  },
  "total_actions": 15,
  "action_counts": {
    "PASS": 8,
    "DRIVE": 3,
    "SHOT": 2,
    "CROSS": 2
  },
  "actions": [
    {
      "frame": 145,
      "action": "PASS",
      "confidence": 0.873,
      "player_frame": 145,
      "frame_offset": 0,
      "player_bbox": [245.5, 123.2, 312.8, 387.4],
      "player_center": [279.15, 255.3],
      "player_confidence": 0.91
    },
    ...
  ]
}
```

## Current Limitations (MVP)

⚠️ **Action detection is currently using placeholder data**

To integrate real action detection:

1. Load trained ball-action-spotting model
2. Replace `_detect_actions()` in `player_action_counter.py`:

```python
def _detect_actions(self, video_path: str) -> List[Dict]:
    from scripts.ball_action.predict import predict_video
    
    # Load your trained model
    model = load_model(self.action_model_path)
    
    # Run prediction
    predictions = predict_video(model, video_path, self.device)
    
    # Convert to required format
    actions = []
    for frame_idx, preds in predictions.items():
        for action_class, conf in preds.items():
            if conf > 0.5:
                actions.append({
                    'frame': frame_idx,
                    'action': action_class,
                    'confidence': float(conf)
                })
    
    return actions
```

## Performance Tips

- **Faster processing**: Use `yolov8n.pt` (nano model) and higher `--sample-rate`
- **Better accuracy**: Use `yolov8x.pt` (extra-large) and `--sample-rate 1`
- **GPU memory**: If OOM errors, reduce batch size or use smaller YOLO model
- **Testing**: Use `--max-frames 500` to test on first 20 seconds

## Examples

```bash
# Quick test on first 500 frames
python mvp/player_action_counter.py test.mp4 --max-frames 500

# High accuracy (slower)
python mvp/player_action_counter.py match.mp4 \
    --yolo-model yolov8x.pt \
    --sample-rate 1 \
    --player 10

# Fast processing (lower accuracy)
python mvp/player_action_counter.py match.mp4 \
    --yolo-model yolov8n.pt \
    --sample-rate 10 \
    --interactive

# CPU-only mode
python mvp/player_action_counter.py match.mp4 \
    --device cpu \
    --yolo-model yolov8n.pt
```

## Next Steps

- [ ] Integrate real action detection model
- [ ] Add spatial proximity checking (use action location from attention maps)
- [ ] Add video visualization output
- [ ] Add team classification
- [ ] Add jersey number recognition
- [ ] Add ball tracking for better action-player association
- [ ] Optimize for real-time processing

## Troubleshooting

**"ultralytics not found"**
```bash
pip install ultralytics
```

**"CUDA out of memory"**
- Use smaller YOLO model: `--yolo-model yolov8n.pt`
- Increase sample rate: `--sample-rate 10`
- Use CPU: `--device cpu`

**"No players detected"**
- Check video quality/resolution
- Adjust confidence threshold in `detector.py`
- Try different YOLO model

**"ImportError: cannot import name 'PlayerTracker'"**
```bash
# Make sure you're in the project root
cd /path/to/ball-action-spotting
python mvp/run_mvp.py video.mp4
```

## Development

To add new features:

1. **Player tracking improvements**: Edit `src/player_tracking/`
2. **Action detection integration**: Edit `mvp/player_action_counter.py`
3. **Matching logic**: Edit `src/player_tracking/matcher.py`
4. **Add tests**: Create `tests/test_player_tracking.py`

## License

Same as ball-action-spotting project.
