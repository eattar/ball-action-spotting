# Player Tracking Integration Test Guide

This guide shows how to test the complete player tracking + ball action detection pipeline.

## Prerequisites

On your VM, ensure you have:
- ✅ Ball action predictions already computed
- ✅ YOLOv8 model downloaded (will auto-download if missing)
- ✅ Video file accessible
- ✅ Environment activated: `conda activate ball-action-spotting`

## Quick Start

### 1. Pull Latest Changes

```bash
cd /workspace/ball-action-spotting
git pull origin master
```

### 2. Set Environment Variables

```bash
export SOCCERNET_DIR=/netscratch/eattar/ds/SoccetNet
export DATA_DIR=/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/data
```

### 3. Run Integration Test

**Using pre-computed predictions** (fastest):

```bash
python test_player_tracking_integration.py \
  --video "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/720p.mp4" \
  --predictions "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/data/ball_action/predictions/sampling_weights_001/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/results_spotting.json" \
  --output player_stats.json
```

## What This Does

1. **Loads ball actions** from results_spotting.json (4,055 actions)
2. **Runs player tracking** on the video using YOLOv8
3. **Shows player selection UI** (or text-based on headless server)
4. **Matches actions to player** using temporal proximity
5. **Generates statistics** showing how many PASS/DRIVE actions the player performed

## Expected Output

```
============================================================
Player Tracking + Ball Action Integration Test
============================================================

Loaded 4055 ball actions from ...

Running player tracking on ...
Tracked 146450 frames
Selected player ID: 42

Matching actions to player...
Matched 287 actions to player 42

============================================================
Results for Player 42
============================================================

Total actions: 287

Actions by type:
  PASS: 162 (56.4%)
  DRIVE: 125 (43.6%)

First 5 matched actions:
  1. Frame   1520 - PASS       (conf: 0.822, time: 1 - 00:01)
  2. Frame   3640 - PASS       (conf: 0.911, time: 1 - 00:03)
  3. Frame   7840 - DRIVE      (conf: 0.875, time: 1 - 00:05)
  ...

Results saved to player_stats.json

============================================================
Integration test complete!
============================================================
```

## Shorter Test (First 5 Minutes)

To test quickly without processing the full 98-minute video:

```bash
# Extract first 5 minutes (7500 frames at 25fps)
ffmpeg -i "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/720p.mp4" \
  -t 300 -c copy test_5min.mp4

# Run on shorter video
python test_player_tracking_integration.py \
  --video test_5min.mp4 \
  --predictions "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/data/ball_action/predictions/sampling_weights_001/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/results_spotting.json" \
  --output test_stats.json
```

## Alternative: Use Different Video

Test on a different fold's video:

```bash
# First run predictions for that fold
python scripts/ball_action/predict.py --experiment sampling_weights_001 --folds 1

# Then run integration test
python test_player_tracking_integration.py \
  --video "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday/720p.mp4" \
  --predictions "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/data/ball_action/predictions/sampling_weights_001/cv/fold_1/england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday/results_spotting.json" \
  --output hull_stats.json
```

## Troubleshooting

### "No display available" Error
- The script auto-detects headless mode and falls back to text-based player selection
- You'll see a list of player IDs to choose from

### YOLOv8 Model Download
- First run will download yolov8n.pt (~6MB)
- Stored in `~/.cache/ultralytics/`

### Memory Issues
- Use `--yolo-model yolov8n.pt` (nano model, default)
- Test on shorter video segment first

### Player ID Not Found
- Player IDs change between runs (ByteTrack assigns them dynamically)
- Script will show top 20 available IDs
- Pick one from the list

## Output Files

### player_stats.json
Contains:
- Selected player ID
- Total matched actions
- Per-action-type statistics
- Full list of matched actions with timestamps

```json
{
  "video": "...",
  "player_id": 42,
  "total_actions": 287,
  "statistics": {
    "PASS": 162,
    "DRIVE": 125
  },
  "matched_actions": [...]
}
```

## Next Steps

After successful test:
1. ✅ Verify statistics are reasonable
2. ✅ Try different players
3. ✅ Test on multiple videos
4. 🎯 Add visualization (annotated video output)
5. 🎯 Batch process multiple videos
6. 🎯 Generate team-wide statistics

## Performance

- **Player tracking**: ~120 fps (OpenCV) or ~300 fps (NvDec)
- **Full 98-min video**: ~20 minutes processing time
- **5-min segment**: ~2.5 minutes processing time
