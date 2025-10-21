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


def quick_test(video_path: str, player_id: int = None):
    """
    Quick test function - minimal arguments
    
    Usage:
        python run_mvp.py video.mp4
        python run_mvp.py video.mp4 7
    """
    counter = PlayerActionCounter(
        yolo_model='yolov8n.pt',  # Fastest model
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
        print("\nUsage: python run_mvp.py <video_path> [player_id]")
        print("\nArguments:")
        print("  video_path    Path to video file (required)")
        print("  player_id     Player track ID to analyze (optional)")
        print("                If not provided, will show interactive selection")
        print("\nExamples:")
        print("  python run_mvp.py match.mp4")
        print("    → Shows first frame with player IDs, lets you select")
        print()
        print("  python run_mvp.py match.mp4 7")
        print("    → Directly tracks player #7")
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
    player = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    quick_test(video, player)
