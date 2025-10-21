#!/usr/bin/env python3
"""
Test script for integrating player tracking with existing ball action predictions.

This script:
1. Loads pre-computed ball action predictions (results_spotting.json)
2. Runs player tracking on the same video
3. Matches actions to players
4. Generates per-player statistics
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.player_tracking import PlayerTracker, ActionPlayerMatcher


def load_ball_actions(json_path: str) -> List[Dict]:
    """Load ball actions from results_spotting.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    actions = []
    for pred in data['predictions']:
        # Convert position (milliseconds) to frame number (at 25 fps)
        frame = int(int(pred['position']) / 1000 * 25)
        
        actions.append({
            'frame': frame,
            'label': pred['label'],
            'confidence': float(pred['confidence']),
            'game_time': pred['gameTime'],
            'position_ms': int(pred['position'])
        })
    
    print(f"Loaded {len(actions)} ball actions from {json_path}")
    return actions


def track_players(video_path: str, yolo_model: str = 'yolov8n.pt') -> Dict:
    """Run player tracking on video"""
    print(f"\nRunning player tracking on {video_path}...")
    
    tracker = PlayerTracker(model_name=yolo_model, device='cuda')
    
    # Track players (user will select player interactively or via text mode)
    tracking_data = tracker.track_video(video_path)
    
    print(f"Tracked {len(tracking_data['all_detections'])} frames")
    print(f"Selected player ID: {tracking_data['selected_player_id']}")
    
    return tracking_data


def match_actions_to_player(actions: List[Dict], tracking_data: Dict) -> List[Dict]:
    """Match ball actions to the tracked player"""
    print("\nMatching actions to player...")
    
    matcher = ActionPlayerMatcher(temporal_window=25)  # 1 second at 25fps
    
    matched_actions = matcher.match_actions_to_player(
        actions,
        tracking_data['all_detections'],
        tracking_data['selected_player_id']
    )
    
    print(f"Matched {len(matched_actions)} actions to player {tracking_data['selected_player_id']}")
    
    return matched_actions


def generate_statistics(matched_actions: List[Dict]) -> Dict:
    """Generate per-player action statistics"""
    stats = {}
    
    for action in matched_actions:
        label = action['label']
        stats[label] = stats.get(label, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Test player tracking with ball action predictions')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--predictions', required=True, help='Path to results_spotting.json')
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='YOLO model to use')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not Path(args.predictions).exists():
        print(f"Error: Predictions file not found: {args.predictions}")
        return 1
    
    print("="*60)
    print("Player Tracking + Ball Action Integration Test")
    print("="*60)
    
    # Step 1: Load ball actions
    actions = load_ball_actions(args.predictions)
    
    # Step 2: Track players
    tracking_data = track_players(args.video, args.yolo_model)
    
    # Step 3: Match actions to player
    matched_actions = match_actions_to_player(actions, tracking_data)
    
    # Step 4: Generate statistics
    stats = generate_statistics(matched_actions)
    
    # Display results
    print("\n" + "="*60)
    print(f"Results for Player {tracking_data['selected_player_id']}")
    print("="*60)
    print(f"\nTotal actions: {len(matched_actions)}")
    print("\nActions by type:")
    for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(matched_actions) * 100) if matched_actions else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Show some examples
    print("\nFirst 5 matched actions:")
    for i, action in enumerate(matched_actions[:5], 1):
        print(f"  {i}. Frame {action['frame']:6d} - {action['label']:10s} "
              f"(conf: {action['confidence']:.3f}, time: {action['game_time']})")
    
    # Save results if requested
    if args.output:
        output_data = {
            'video': args.video,
            'player_id': tracking_data['selected_player_id'],
            'total_actions': len(matched_actions),
            'statistics': stats,
            'matched_actions': matched_actions
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "="*60)
    print("Integration test complete!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
