#!/usr/bin/env python3
"""
Simple CLI runner for player action counting

Quick wrapper around player_action_counter.py for easy testing
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from player_action_counter import PlayerActionCounter


def quick_test(video_path: str, player_id: int = None, model_path: str = None):
    """
    Quick test function - minimal arguments
    
    Usage:
        python run_mvp.py video.mp4
        python run_mvp.py video.mp4 7
        python run_mvp.py video.mp4 7 model.pth
    """
    counter = PlayerActionCounter(
        yolo_model='yolov8n.pt',  # Fastest model
        action_model_path=model_path,  # Real action detection model (optional)
        device='cuda'
    )
    
    results = counter.process_video(
        video_path,
        player_id=player_id,
        interactive=(player_id is None),
        max_frames=None,  # Process all frames
        sample_rate=5      # Every 5th frame
    )
    
    if results:
        counter.print_results(results)
        
        # Auto-save
        output_path = f"player_{results['player_id']}_actions.json"
        counter.save_results(results, output_path)


if __name__ == '__main__':
    # Check for help flag
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("=" * 60)
        print("Player Action Counter - Quick Test Script")
        print("=" * 60)
        print("\nUsage: python run_mvp.py <video_path> [player_id] [model_path]")
        print("\nArguments:")
        print("  video_path    Path to video file (required)")
        print("  player_id     Player track ID to analyze (optional)")
        print("                If not provided, will show interactive selection")
        print("  model_path    Path to trained action model .pth file (optional)")
        print("                If not provided, uses placeholder detection")
        print("\nExamples:")
        print("  python run_mvp.py match.mp4")
        print("    → Shows first frame with player IDs, lets you select")
        print("    → Uses placeholder action detection")
        print()
        print("  python run_mvp.py match.mp4 7")
        print("    → Directly tracks player #7")
        print("    → Uses placeholder action detection")
        print()
        print("  python run_mvp.py match.mp4 7 data/ball_action/experiments/ball_finetune_long_004/fold_0/model-006-0.864002.pth")
        print("    → Tracks player #7 using real trained model")
        print()
        print("\nOptions:")
        print("  Settings: Uses yolov8n.pt, CUDA, sample_rate=5")
        print("  Output:   Saves to player_<ID>_actions.json")
        print()
        print("For advanced options (different YOLO model, CPU mode, etc):")
        print("  python mvp/player_action_counter.py --help")
        print("=" * 60)
        sys.exit(0)
    
    video = sys.argv[1]
    
    # Parse player_id with better error handling
    player = None
    model = None
    
    if len(sys.argv) > 2 and sys.argv[2]:  # Check if not empty string
        try:
            player = int(sys.argv[2])
        except ValueError:
            print(f"❌ Error: Invalid player ID '{sys.argv[2]}'")
            print(f"   Player ID must be a number (e.g., 7)")
            print()
            print("💡 Tip: If your video path has spaces, use quotes:")
            print(f'   python run_mvp.py "/path/with spaces/video.mp4"')
            print()
            print(f"   It looks like you might have unquoted spaces in your path.")
            print(f"   Received {len(sys.argv)} arguments: {sys.argv[1:]}")
            print()
            print("💡 To see available players, just run:")
            print(f"   python run_mvp.py video.mp4")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        model = sys.argv[3]
        if not os.path.exists(model):
            print(f"❌ Error: Model file not found: {model}")
            sys.exit(1)
    
    # Check if video file exists
    if not os.path.exists(video):
        print(f"❌ Error: Video file not found: {video}")
        print()
        print("💡 Tip: If your path has spaces, use quotes:")
        print(f'   python run_mvp.py "/path/with spaces/video.mp4"')
        sys.exit(1)
    
    quick_test(video, player, model)
