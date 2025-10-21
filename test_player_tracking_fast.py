#!/usr/bin/env python3
"""
FAST version: Player tracking integration with frame skipping and optimizations.

Key optimizations:
1. Skip frames (process every Nth frame)
2. Only track during action windows
3. Use smaller YOLO model
4. Batch processing
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys
import cv2
from tqdm import tqdm

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


def get_action_windows(actions: List[Dict], window_before: int = 50, window_after: int = 25) -> List[tuple]:
    """
    Get frame ranges around actions (only track during these windows)
    
    Args:
        actions: List of action dicts with 'frame' key
        window_before: Frames to track before action (2 seconds at 25fps)
        window_after: Frames to track after action (1 second at 25fps)
    
    Returns:
        List of (start_frame, end_frame) tuples
    """
    windows = []
    for action in actions:
        frame = action['frame']
        windows.append((frame - window_before, frame + window_after))
    
    # Merge overlapping windows
    if not windows:
        return []
    
    windows.sort()
    merged = [windows[0]]
    
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping, merge
            merged[-1] = (last_start, max(end, last_end))
        else:
            merged.append((start, end))
    
    total_frames = sum(end - start for start, end in merged)
    print(f"Action windows: {len(merged)} segments, {total_frames:,} total frames to process")
    
    return merged


def track_players_fast(
    video_path: str, 
    action_windows: List[tuple],
    yolo_model: str = 'yolov8n.pt',
    frame_skip: int = 2
) -> Dict:
    """
    Run fast player tracking only during action windows
    
    Args:
        video_path: Path to video
        action_windows: List of (start_frame, end_frame) tuples
        yolo_model: YOLO model (use 'n' for fastest)
        frame_skip: Process every Nth frame (2 = every other frame)
    """
    print(f"\nRunning FAST player tracking...")
    print(f"Frame skip: {frame_skip} (processing {100/frame_skip:.0f}% of frames)")
    
    from ultralytics import YOLO
    
    # Load YOLO model
    model = YOLO(yolo_model)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Track only in action windows
    all_detections = {}
    frame_count = 0
    processed_count = 0
    
    with tqdm(total=total_frames, desc="Tracking") as pbar:
        for start_frame, end_frame in action_windows:
            # Seek to start of window
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, min(end_frame, total_frames), frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO tracking
                results = model.track(frame, persist=True, verbose=False, classes=[0])  # class 0 = person
                
                if results and results[0].boxes is not None:
                    detections = []
                    boxes = results[0].boxes
                    
                    for i, box in enumerate(boxes):
                        if hasattr(box, 'id') and box.id is not None:
                            bbox = box.xyxy[0].cpu().numpy()
                            track_id = int(box.id.cpu().numpy()[0])
                            conf = float(box.conf.cpu().numpy()[0])
                            
                            detections.append({
                                'bbox': bbox.tolist(),
                                'track_id': track_id,
                                'confidence': conf
                            })
                    
                    if detections:
                        all_detections[frame_idx] = detections
                        processed_count += 1
                
                frame_count += 1
                pbar.update(frame_skip)
    
    cap.release()
    
    print(f"Processed {processed_count:,} frames (skipped {frame_count - processed_count:,})")
    print(f"Found detections in {len(all_detections):,} frames")
    
    # Get all unique track IDs
    all_track_ids = set()
    for detections in all_detections.values():
        for det in detections:
            all_track_ids.add(det['track_id'])
    
    print(f"Found {len(all_track_ids)} unique players")
    
    # Show top players by appearance count
    track_counts = {}
    for detections in all_detections.values():
        for det in detections:
            tid = det['track_id']
            track_counts[tid] = track_counts.get(tid, 0) + 1
    
    top_players = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 players by appearance:")
    for tid, count in top_players:
        print(f"  Player {tid}: {count} frames")
    
    # Select player (simplified - just pick most frequent)
    selected_player_id = top_players[0][0]
    print(f"\nAuto-selecting most frequent player: {selected_player_id}")
    
    return {
        'all_detections': all_detections,
        'selected_player_id': selected_player_id,
        'frame_skip': frame_skip
    }


def match_actions_to_player(actions: List[Dict], tracking_data: Dict) -> List[Dict]:
    """Match ball actions to the tracked player"""
    print("\nMatching actions to player...")
    
    from src.player_tracking.detector import PlayerDetection
    
    # Convert raw detections to PlayerDetection objects for the selected player
    selected_id = tracking_data['selected_player_id']
    player_tracks = {}
    
    for frame_idx, detections in tracking_data['all_detections'].items():
        for det in detections:
            if det['track_id'] == selected_id:
                # Convert dict to PlayerDetection object
                bbox = det['bbox']
                player_tracks[frame_idx] = PlayerDetection(
                    bbox=bbox,
                    confidence=det['confidence'],
                    track_id=det['track_id']
                )
                break  # Only need the selected player
    
    print(f"Player {selected_id} visible in {len(player_tracks)} frames")
    
    matcher = ActionPlayerMatcher(temporal_window=50)  # 2 seconds at 25fps
    
    matched_actions = matcher.match_actions_to_player(
        actions,
        player_tracks,
        selected_id
    )
    
    print(f"Matched {len(matched_actions)} actions to player {selected_id}")
    
    return matched_actions


def generate_statistics(matched_actions: List[Dict]) -> Dict:
    """Generate per-player action statistics"""
    stats = {}
    
    for action in matched_actions:
        label = action['label']
        stats[label] = stats.get(label, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='FAST player tracking integration')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--predictions', required=True, help='Path to results_spotting.json')
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='YOLO model (n/s/m/l/x)')
    parser.add_argument('--frame-skip', type=int, default=2, help='Process every Nth frame (default: 2)')
    parser.add_argument('--window-before', type=int, default=50, help='Frames before action (default: 50)')
    parser.add_argument('--window-after', type=int, default=25, help='Frames after action (default: 25)')
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
    print("FAST Player Tracking + Ball Action Integration")
    print("="*60)
    
    # Step 1: Load ball actions
    actions = load_ball_actions(args.predictions)
    
    # Step 2: Get action windows (only track during these periods)
    action_windows = get_action_windows(actions, args.window_before, args.window_after)
    
    # Step 3: Track players (fast mode)
    tracking_data = track_players_fast(
        args.video, 
        action_windows,
        args.yolo_model,
        args.frame_skip
    )
    
    # Step 4: Match actions to player
    matched_actions = match_actions_to_player(actions, tracking_data)
    
    # Step 5: Generate statistics
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
            'matched_actions': matched_actions,
            'optimization': {
                'frame_skip': tracking_data['frame_skip'],
                'windows': len(action_windows)
            }
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
