# Ball-Action-Spotting Model Information

## Available Models

### 1. Ball Action Models (`ball_action/experiments/`)

Located in: `data/ball_action/experiments/`

These models detect ball-related actions:

#### `ball_finetune_long_004`
- **Type**: Binary classifier (2 classes)
- **Classes**: PASS (index 0), DRIVE (index 1)
- **Purpose**: Detect when PASS or DRIVE actions occur
- **Accuracy**: ~86-90% (best fold: fold_5 with 90.1%)
- **Training**: 6 epochs
- **Note**: Does NOT detect all 12 action types, only PASS and DRIVE

#### `ball_tuning_001`
- **Type**: Binary classifier (2 classes)
- **Classes**: PASS, DRIVE
- **Accuracy**: ~87-89%
- **Training**: 34 epochs

#### `sampling_weights_001`
- **Type**: Binary classifier (2 classes)  
- **Classes**: PASS, DRIVE
- **Accuracy**: ~83-88%
- **Training**: 29 epochs with weighted sampling

### 2. Action Models (`action/experiments/`)

Located in: `data/action/experiments/`

These models detect general soccer events (NOT ball actions):

#### `action_sampling_weights_002`
- **Type**: Multi-class classifier (15 classes)
- **Classes**: Penalty, Kick-off, Goal, Substitution, Offside, Shots on target, Shots off target, Clearance, Ball out of play, Throw-in, Foul, Indirect free-kick, Direct free-kick, Corner, Card
- **Purpose**: Detect general game events (referee decisions, game flow)
- **Accuracy**: ~80%
- **Note**: Different from ball actions - these are game events, not player ball touches

## The 12 Ball Action Classes

According to `src/ball_action/constants.py`, the full set of ball actions is:

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

## Current Limitation

⚠️ **Important**: The available `ball_action` models only detect **2 out of 12** classes (PASS and DRIVE).

To get all 12 action types, you would need:
- A model trained on all 12 classes, OR
- Separate binary classifiers for each action type, OR
- A different approach (e.g., two-stage: detect ball touch, then classify type)

## For Player Tracking Integration

**Current approach** (working):
- Use `ball_finetune_long_004` model
- Detects PASS and DRIVE actions only
- Match these to tracked players
- Output shows which player performed PASS or DRIVE actions

**Future enhancement options**:
1. Train a 12-class ball action model
2. Use ensemble of binary classifiers
3. Combine ball detection + action classification in two stages
4. Use attention maps for spatial matching

## Model Files

All models are ~53MB each:

```
data/ball_action/experiments/ball_finetune_long_004/
├── fold_0/model-006-0.864002.pth
├── fold_1/model-006-0.862339.pth  
├── fold_2/model-006-0.758246.pth
├── fold_3/model-006-0.887217.pth
├── fold_4/model-006-0.869269.pth
├── fold_5/model-006-0.901643.pth  ← Best model (90.1%)
└── fold_6/model-006-0.897602.pth
```

## Usage

### Test ball action detection (PASS + DRIVE only):
```bash
python test_ball_action_prediction.py \
  data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth \
  video.mp4
```

### Full player tracking + action detection:
```bash
python mvp/run_mvp.py \
  video.mp4 \
  player_id \
  data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth
```

This will show which PASS and DRIVE actions were performed by the specified player.
