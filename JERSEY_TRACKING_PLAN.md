# Enhanced Pipeline: Jersey Number Recognition → Player Tracking → Action Detection

## Overview

New pipeline with jersey number recognition for better UX:

```
1. Player Detection (YOLOv8)
2. Jersey Number Recognition (OCR/CNN)
3. User selects jersey number (e.g., #10)
4. Track selected player throughout video
5. Match ball actions to that player
6. Output: "Player #10 performed 42 actions (23 DRIVE, 19 PASS)"
```

## Implementation Plan

### Phase 1: Jersey Number Recognition

**Option A: Use SoccerNet Jersey Challenge Model**
- SoccerNet has a jersey number recognition challenge
- Pre-trained models available: https://github.com/SoccerNet/sn-jersey

**Option B: EasyOCR (Quick Start)**
- Lightweight OCR library
- Can read numbers from jersey crops
- Easier to integrate initially

**Option C: Custom CNN**
- Train on jersey number dataset
- More accurate for soccer context
- Requires training data

### Phase 2: Integration Architecture

```python
class JerseyBasedPlayerTracker:
    def __init__(self):
        self.player_detector = YOLO('yolov8n.pt')  # Person detection
        self.jersey_recognizer = JerseyNumberRecognizer()  # New component
        
    def detect_and_recognize(self, frame):
        # Detect all players
        players = self.player_detector(frame)
        
        # For each player, extract jersey crop and recognize number
        for player in players:
            bbox = player.bbox
            jersey_crop = extract_torso_region(frame, bbox)
            jersey_number = self.jersey_recognizer(jersey_crop)
            player.jersey_number = jersey_number
        
        return players
    
    def track_by_jersey(self, video_path, target_jersey):
        # Track player with specific jersey number
        all_frames = []
        
        for frame in video:
            players = self.detect_and_recognize(frame)
            
            # Find player with target jersey
            for player in players:
                if player.jersey_number == target_jersey:
                    all_frames.append({
                        'frame_idx': frame_idx,
                        'bbox': player.bbox,
                        'jersey': target_jersey
                    })
        
        return all_frames
```

### Phase 3: User Flow

```bash
# Step 1: Quick scan to find available jersey numbers
python detect_jerseys.py --video video.mp4 --sample-rate 100

# Output:
# Found jersey numbers: #1, #5, #7, #10, #11, #14, #17, #23
# 
# Most frequent:
#   #10: 450 frames
#   #7:  420 frames
#   #11: 380 frames

# Step 2: User selects jersey number
python track_by_jersey.py \
  --video video.mp4 \
  --jersey 10 \
  --predictions results_spotting.json \
  --output player_10_stats.json

# Output:
# Tracking player #10...
# Found in 450 frames
# Matched 42 actions
#   PASS: 19
#   DRIVE: 23
```

## Quick Implementation (EasyOCR)

### Install Dependencies

```bash
pip install easyocr opencv-python
```

### Jersey Number Recognition Script

```python
import easyocr
import cv2
import numpy as np

class JerseyNumberRecognizer:
    def __init__(self):
        # Initialize EasyOCR (one-time setup)
        self.reader = easyocr.Reader(['en'], gpu=True)
    
    def extract_jersey_region(self, frame, bbox):
        """Extract torso region (likely location of jersey number)"""
        x1, y1, x2, y2 = bbox
        
        # Focus on upper-center part of bbox (where jersey number usually is)
        height = y2 - y1
        width = x2 - x1
        
        # Take top 40% of height, center 60% of width
        torso_y1 = int(y1 + height * 0.1)
        torso_y2 = int(y1 + height * 0.5)
        torso_x1 = int(x1 + width * 0.2)
        torso_x2 = int(x2 - width * 0.2)
        
        jersey_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        return jersey_crop
    
    def recognize_number(self, jersey_crop):
        """Recognize jersey number using OCR"""
        if jersey_crop.size == 0:
            return None
        
        # Enhance image for better OCR
        gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR
        results = self.reader.readtext(thresh)
        
        # Filter for numbers only
        for (bbox, text, prob) in results:
            # Remove non-digit characters
            cleaned = ''.join(filter(str.isdigit, text))
            if cleaned and prob > 0.5:  # Confidence threshold
                return int(cleaned)
        
        return None
```

## Alternative: SoccerNet Jersey Model

Use pre-trained model from SoccerNet:

```bash
# Clone SoccerNet jersey recognition repo
git clone https://github.com/SoccerNet/sn-jersey.git

# Install dependencies
cd sn-jersey
pip install -r requirements.txt

# Use their model
from sn_jersey import JerseyRecognizer

recognizer = JerseyRecognizer(weights='path/to/weights.pth')
jersey_number = recognizer.predict(player_crop)
```

## Challenges & Solutions

### Challenge 1: Jersey Not Always Visible
- **Problem**: Player facing away, occluded, or blurry
- **Solution**: Track across multiple frames, use most common number detected

### Challenge 2: Different Teams Same Numbers
- **Problem**: Both teams might have #10
- **Solution**: 
  - Also detect jersey color/team
  - Let user specify "Home #10" vs "Away #10"
  - Use spatial zones (left half = team A, right half = team B)

### Challenge 3: OCR Accuracy
- **Problem**: Numbers misread (8 vs 3, 6 vs 5, etc.)
- **Solution**:
  - Use multiple frames and vote
  - Add jersey color context
  - Fine-tune OCR for soccer fonts

### Challenge 4: Goalkeepers Different Colors
- **Problem**: Goalkeeper jersey looks different
- **Solution**: Detect by position (usually near goal) + different color

## Complete New Pipeline Script

I'll create a full implementation with jersey recognition:

```python
#!/usr/bin/env python3
"""
Jersey-based player tracking with ball action detection.

Usage:
    python track_by_jersey.py --video video.mp4 --jersey 10
"""
```

Would you like me to:
1. **Implement EasyOCR version** (quick, works now)
2. **Integrate SoccerNet jersey model** (more accurate, needs setup)
3. **Build hybrid** (OCR + color detection for team identification)

Which approach do you prefer?
