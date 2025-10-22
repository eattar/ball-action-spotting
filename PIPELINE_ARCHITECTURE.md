# Pipeline Architecture Documentation

## Current Implementation Flow

### Step-by-Step Breakdown

```python
# STEP 1: Load Pre-computed Actions
# File: test_player_tracking_fast.py, line ~25
actions = load_ball_actions(predictions_json)
# Output: 4,055 action dicts with frame numbers

# STEP 2: Create Action Windows  
# File: test_player_tracking_fast.py, line ~60
action_windows = get_action_windows(actions, window_before=50, window_after=25)
# Output: 314 time segments to track
# Example: [(118174, 118249), (118205, 118280), ...]

# STEP 3: Track Players (Only in Windows)
# File: test_player_tracking_fast.py, line ~105
tracking_data = track_players_fast(video, action_windows, frame_skip=2)
# Processes: ~117k frames (80% of video)
# Output: {
#   'all_detections': {frame_idx: [player_dets, ...]},
#   'selected_player_id': 48633,
#   'top_players': [(48633, 569), ...]
# }

# STEP 4: Filter to Selected Player
# File: test_player_tracking_fast.py, line ~195
player_tracks = {}  # Only frames with selected player
for frame_idx, detections in all_detections.items():
    for det in detections:
        if det['track_id'] == selected_player_id:
            player_tracks[frame_idx] = PlayerDetection(det)

# STEP 5: Temporal Matching
# File: src/player_tracking/matcher.py, line ~60
for action in actions:
    action_frame = action['frame']
    
    # Search ±50 frames (2 seconds)
    for offset in range(-50, 51):
        check_frame = action_frame + offset
        if check_frame in player_tracks:
            # Player was visible near this action time
            matched_actions.append(action)
            break

# STEP 6: Remove Duplicates
# File: src/player_tracking/matcher.py, line ~201
# If same action detected in consecutive frames, keep best one

# OUTPUT: 42 matched actions for Player 48633
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│  PRE-COMPUTED (scripts/predict.py)  │
│  ─────────────────────────────────  │
│  Video (146k frames)                │
│         ↓                            │
│  Ball Action Model (EfficientNet)   │
│         ↓                            │
│  results_spotting.json (4,055)      │
└─────────────────────────────────────┘
         ↓
         ↓ [Load actions]
         ↓
┌─────────────────────────────────────┐
│  ACTION WINDOW EXTRACTION           │
│  ─────────────────────────────────  │
│  For each action at frame F:        │
│    window = [F-50, F+25]            │
│  Merge overlapping windows          │
│  Result: 314 segments                │
└─────────────────────────────────────┘
         ↓
         ↓ [Time ranges to process]
         ↓
┌─────────────────────────────────────┐
│  PLAYER TRACKING                    │
│  ─────────────────────────────────  │
│  YOLOv8 + ByteTrack                 │
│  Only process action windows        │
│  Skip every 2nd frame               │
│  Result: 57,594 frames processed    │
│          11,679 unique track IDs    │
└─────────────────────────────────────┘
         ↓
         ↓ [All player detections]
         ↓
┌─────────────────────────────────────┐
│  PLAYER SELECTION                   │
│  ─────────────────────────────────  │
│  Count appearances per track_id     │
│  Select most frequent (or manual)   │
│  Result: Player 48633               │
└─────────────────────────────────────┘
         ↓
         ↓ [Selected player ID]
         ↓
┌─────────────────────────────────────┐
│  TEMPORAL MATCHING                  │
│  ─────────────────────────────────  │
│  For each action:                   │
│    Is player visible ±2 sec?        │
│    If yes → Match!                  │
│  Remove duplicates                  │
│  Result: 42 matched actions         │
└─────────────────────────────────────┘
         ↓
         ↓ [Final results]
         ↓
┌─────────────────────────────────────┐
│  OUTPUT                             │
│  ─────────────────────────────────  │
│  player_stats.json                  │
│  - Player 48633                     │
│  - 23 DRIVE, 19 PASS                │
│  - Timestamps, confidence, etc.     │
└─────────────────────────────────────┘
```

---

## Why Not Track → Actions?

### Option A: Current (Actions → Track)
**Time**: ~20 minutes
```
Ball action detection:  Already done (pre-computed)
Player tracking:        20 min (action windows only)
Matching:               <1 sec
───────────────────────────────────────────
Total:                  ~20 minutes
```

### Option B: Alternative (Track → Actions)
**Time**: ~50+ minutes
```
Player tracking:        30 min (full video)
Ball action detection:  20 min (player-centric frames)
Matching:               <1 sec
───────────────────────────────────────────
Total:                  ~50 minutes
```

**Decision**: Option A is 2.5x faster for similar results!

---

## Optimization Opportunities

### 1. Add Ball Detection
Track the ball itself:
- Know exactly where ball is at each frame
- Match actions to closest player to ball
- More accurate than temporal proximity alone

### 2. Add Spatial Filtering
Use action attention maps:
- Ball action model can show WHERE in frame action occurred
- Match to player whose bbox overlaps action region
- Reduces false matches

### 3. Cache Player Tracks
Save tracking results:
```python
# First run: Track and save
track_and_save(video, output='player_tracks.pkl')

# Subsequent runs: Just load
player_tracks = load_cached_tracks('player_tracks.pkl')
```
Instant results for different players!

### 4. Parallel Processing
Process multiple videos simultaneously:
```bash
parallel -j 4 python test_player_tracking_fast.py \
  --video {} \
  --predictions {}.json \
  ::: video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

---

## Summary

**Current Pipeline Order:**
1. ✅ Ball actions detected first (pre-computed)
2. ✅ Action windows extracted
3. ✅ Players tracked only in those windows
4. ✅ Actions matched to selected player

**Why This Order:**
- Leverages pre-computed ball actions
- Minimizes tracking time (only ~80% of video)
- Fast iteration for different players
- Good accuracy with temporal matching

**Future Improvements:**
- Add ball tracking for spatial matching
- Cache player tracks for instant re-runs
- Add attention-based spatial filtering
