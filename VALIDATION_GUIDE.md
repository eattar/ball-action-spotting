# Validating Player Tracking + Action Detection Results

This guide explains how to validate the accuracy of player-action matching.

## 1. Visual Validation (Most Reliable)

### A. Generate Annotated Video Clips

Create short video clips showing matched actions to manually verify:

```python
import cv2
import json

def extract_action_clips(video_path, matched_actions, output_dir, window=50):
    """Extract video clips around matched actions for manual review"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    for i, action in enumerate(matched_actions[:10]):  # First 10 actions
        frame_num = action['frame']
        start_frame = max(0, frame_num - window)
        end_frame = frame_num + window
        
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            f'{output_dir}/action_{i}_{action["label"]}_frame_{frame_num}.mp4',
            fourcc, fps, (1280, 720)
        )
        
        for f in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw bounding box if available
            if 'player_bbox' in action and f == frame_num:
                bbox = action['player_bbox']
                cv2.rectangle(frame, 
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0), 3)
                cv2.putText(frame, f"{action['label']} ({action['confidence']:.2f})",
                    (int(bbox[0]), int(bbox[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Saved clip {i+1}/10")
    
    cap.release()

# Usage
with open('player_stats.json') as f:
    data = json.load(f)

extract_action_clips(
    '/path/to/video.mp4',
    data['matched_actions'],
    'validation_clips'
)
```

Then manually review the clips to see if the correct player is highlighted.

---

## 2. Statistical Validation

### A. Compare with Ground Truth (if available)

If you have manual annotations for specific players:

```bash
python -c "
import json

# Load your results
with open('player_stats.json') as f:
    results = json.load(f)

# Load ground truth (if available)
with open('ground_truth_player_X.json') as f:
    ground_truth = json.load(f)

# Calculate metrics
true_positives = len(set(results['matched_frames']) & set(ground_truth['frames']))
false_positives = len(set(results['matched_frames']) - set(ground_truth['frames']))
false_negatives = len(set(ground_truth['frames']) - set(results['matched_frames']))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.2%}')
print(f'Recall: {recall:.2%}')
print(f'F1 Score: {f1:.2%}')
"
```

### B. Sanity Checks

```bash
python -c "
import json

with open('player_stats.json') as f:
    data = json.load(f)

print('Sanity Checks:')
print('='*60)

# 1. Action distribution should be reasonable
stats = data['statistics']
total = sum(stats.values())
print(f'Total actions: {total}')
print(f'Action distribution: {stats}')

# 2. Confidence distribution
confidences = [a['confidence'] for a in data['matched_actions']]
print(f'\nConfidence stats:')
print(f'  Mean: {sum(confidences)/len(confidences):.3f}')
print(f'  Min: {min(confidences):.3f}')
print(f'  Max: {max(confidences):.3f}')

# 3. Temporal distribution (should be spread across game)
frames = [a['frame'] for a in data['matched_actions']]
print(f'\nTemporal spread:')
print(f'  First action: frame {min(frames):,}')
print(f'  Last action: frame {max(frames):,}')
print(f'  Span: {(max(frames)-min(frames))/25/60:.1f} minutes')

# 4. Frame offsets (should be small if matching is good)
offsets = [abs(a.get('frame_offset', 0)) for a in data['matched_actions']]
print(f'\nFrame offset stats (0 = perfect match):')
print(f'  Mean: {sum(offsets)/len(offsets):.1f} frames')
print(f'  Max: {max(offsets)} frames')
print(f'  Offsets > 25 frames: {sum(1 for o in offsets if o > 25)}')
"
```

---

## 3. Cross-Validation Methods

### A. Compare Multiple Players

Track top 3 players and check if action counts make sense:

```bash
# Modify script to output top 3 players
python test_player_tracking_fast.py ... --top-n-players 3
```

Expected patterns:
- ✅ More defensive players: fewer actions
- ✅ Midfielders: balanced PASS/DRIVE
- ✅ Forwards: more DRIVE actions
- ❌ Goalkeeper: should have very few actions

### B. Team-Level Validation

```python
# Sum all player actions and compare to total detections
# Should account for ~80-90% of total actions (rest are off-screen players)
```

---

## 4. Confidence Threshold Analysis

Test different confidence thresholds to see trade-offs:

```bash
python -c "
import json

with open('player_stats.json') as f:
    data = json.load(f)

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    filtered = [a for a in data['matched_actions'] if a['confidence'] >= threshold]
    print(f'Threshold {threshold}: {len(filtered)} actions')
    
    # Show distribution
    stats = {}
    for a in filtered:
        stats[a['label']] = stats.get(a['label'], 0) + 1
    print(f'  {stats}')
    print()
"
```

High threshold = fewer but more accurate matches  
Low threshold = more matches but potentially noisy

---

## 5. Known Issues to Check

### Issue 1: Track ID Drift
**Problem**: ByteTrack may reassign IDs when players leave/enter frame  
**Check**: Look for sudden jumps in temporal distribution  
**Fix**: Use appearance-based re-identification

### Issue 2: False Matches During Crowded Scenes
**Problem**: Multiple players near ball during corners/throw-ins  
**Check**: Count actions during set pieces (frame ranges from annotations)  
**Fix**: Add spatial filtering using ball location

### Issue 3: Low Visibility Periods
**Problem**: Player partially occluded or off-screen  
**Check**: Review actions with large frame_offset values  
**Fix**: Increase temporal window or add appearance tracking

---

## 6. Quick Validation Script

Run this on your VM to get immediate quality metrics:

```bash
python -c "
import json
import numpy as np

with open('player_stats.json') as f:
    data = json.load(f)

actions = data['matched_actions']

print('='*60)
print('VALIDATION REPORT')
print('='*60)

# Quality indicators
confidences = [a['confidence'] for a in actions]
offsets = [abs(a.get('frame_offset', 0)) for a in actions]

print(f'\n1. Match Quality:')
print(f'   High confidence (>0.5): {sum(1 for c in confidences if c > 0.5)}/{len(confidences)} ({sum(1 for c in confidences if c > 0.5)/len(confidences)*100:.1f}%)')
print(f'   Perfect temporal match (offset=0): {sum(1 for o in offsets if o == 0)}/{len(offsets)} ({sum(1 for o in offsets if o == 0)/len(offsets)*100:.1f}%)')

print(f'\n2. Distribution Check:')
stats = data['statistics']
total = sum(stats.values())
for label, count in stats.items():
    print(f'   {label}: {count} ({count/total*100:.1f}%)')

print(f'\n3. Temporal Spread:')
frames = [a['frame'] for a in actions]
game_minutes = (max(frames) - min(frames)) / 25 / 60
print(f'   First action: {min(frames)/25/60:.1f} min into game')
print(f'   Last action: {max(frames)/25/60:.1f} min into game')
print(f'   Active period: {game_minutes:.1f} minutes')

print(f'\n4. Confidence by Action Type:')
by_type = {}
for a in actions:
    label = a['label']
    if label not in by_type:
        by_type[label] = []
    by_type[label].append(a['confidence'])

for label, confs in by_type.items():
    print(f'   {label}: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}')

print(f'\n5. Red Flags:')
flags = []
if np.mean(confidences) < 0.3:
    flags.append('⚠️  Very low average confidence')
if max(offsets) > 50:
    flags.append('⚠️  Some matches are >2 seconds away')
if game_minutes < 5:
    flags.append('⚠️  Player only active for <5 minutes')
if len(actions) < 10:
    flags.append('⚠️  Very few actions matched (<10)')

if flags:
    for flag in flags:
        print(f'   {flag}')
else:
    print('   ✅ No obvious issues detected')

print('\n' + '='*60)
"
```

---

## 7. Recommended Validation Workflow

1. **Quick check** (1 min): Run validation script above
2. **Visual spot check** (5 min): Extract 5 random clips and manually verify
3. **Statistical analysis** (2 min): Check sanity metrics
4. **Confidence filtering** (2 min): Test different thresholds

Total time: ~10 minutes for thorough validation

---

## Expected Accuracy Ranges

Based on similar systems:

- **Precision**: 70-85% (matched actions are correct)
- **Recall**: 60-80% (% of player's actual actions found)
- **Confidence > 0.5**: Usually 80%+ accurate
- **Temporal offset**: <1 second (25 frames) for 90%+ of matches

Your results (42 actions, mean conf likely ~0.5) suggest moderate-to-good quality.
