"""
MVP: Player Action Counter

End-to-end pipeline that combines player tracking with action detection
to count actions performed by a specific player.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.player_tracking import PlayerTracker, ActionPlayerMatcher


class PlayerActionCounter:
    """
    Main integration class that combines player tracking and action detection
    """
    
    def __init__(
        self,
        yolo_model: str = 'yolov8n.pt',
        action_model_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize player action counter
        
        Args:
            yolo_model: YOLO model for player detection
            action_model_path: Path to trained ball action model (optional for now)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.action_model_path = action_model_path
        
        # Initialize player tracker
        print("Initializing player tracker...")
        self.tracker = PlayerTracker(
            model_name=yolo_model,
            device=device
        )
        
        # Initialize matcher
        self.matcher = ActionPlayerMatcher(
            temporal_window=25,  # 1 second at 25fps
            spatial_threshold=150.0  # pixels
        )
        
        print("✓ Player action counter initialized")
    
    def process_video(
        self,
        video_path: str,
        player_id: Optional[int] = None,
        interactive: bool = False,
        max_frames: Optional[int] = None,
        sample_rate: int = 5
    ) -> Dict:
        """
        Process video to track player and count their actions
        
        Args:
            video_path: Path to video file
            player_id: Player track ID (if None and interactive=False, will use most frequent)
            interactive: Show frame to let user select player
            max_frames: Maximum frames to process (None = all)
            sample_rate: Process every Nth frame for efficiency
            
        Returns:
            Results dictionary with player actions
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}\n")
        
        # Step 1: Track all players in video
        print("Step 1: Tracking players...")
        all_tracks = self.tracker.track_video(
            video_path,
            max_frames=max_frames,
            sample_rate=sample_rate
        )
        
        if not all_tracks:
            print("❌ No players tracked in video")
            return {}
        
        # Step 2: Select target player
        if interactive:
            print("\nStep 2: Interactive player selection...")
            player_id = self.tracker.select_player_interactive(video_path)
            if player_id is None:
                print("❌ No player selected")
                return {}
        elif player_id is None:
            # Use most frequently appearing player
            available_ids = self.tracker.get_available_track_ids()
            if not available_ids:
                print("❌ No players found")
                return {}
            player_id = available_ids[0]
            print(f"\nStep 2: Auto-selected most frequent player: #{player_id}")
        else:
            print(f"\nStep 2: Using specified player: #{player_id}")
        
        # Get player-specific tracks
        player_tracks = self.tracker.get_player_tracks(player_id)
        
        if not player_tracks:
            print(f"❌ Player #{player_id} not found in video")
            return {}
        
        track_stats = self.tracker.get_track_statistics(player_id)
        print(f"  Player visible in {track_stats['num_frames']} frames")
        print(f"  First appearance: frame {track_stats['first_frame']}")
        print(f"  Last appearance: frame {track_stats['last_frame']}")
        print(f"  Avg confidence: {track_stats['avg_confidence']:.2f}")
        
        # Step 3: Detect ball actions
        print("\nStep 3: Detecting ball actions...")
        all_actions = self._detect_actions(video_path)
        
        if not all_actions:
            print("⚠️  No actions detected")
            return {
                'video': os.path.basename(video_path),
                'player_id': player_id,
                'total_actions': 0,
                'action_counts': {},
                'actions': []
            }
        
        print(f"  Detected {len(all_actions)} total actions in video")
        
        # Step 4: Match actions to player
        print("\nStep 4: Matching actions to player...")
        player_actions = self.matcher.match_actions_to_player(
            all_actions,
            player_tracks,
            verbose=True
        )
        
        # Step 5: Create results
        action_counts = self.matcher.count_actions(player_actions)
        summary = self.matcher.create_summary(player_actions, player_id)
        timeline = self.matcher.create_timeline(player_actions)
        
        results = {
            'video': os.path.basename(video_path),
            'video_path': video_path,
            'player_id': player_id,
            'track_statistics': track_stats,
            'total_actions': len(player_actions),
            'action_counts': action_counts,
            'actions': player_actions,
            'summary': summary,
            'timeline': timeline
        }
        
        return results
    
    def _detect_actions(self, video_path: str) -> List[Dict]:
        """
        Detect ball actions in video
        
        For MVP, this uses placeholder data. Replace with actual model inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of action dictionaries
        """
        # TODO: Integrate with actual ball-action-spotting model
        # from scripts.ball_action.predict import predict_video
        
        print("  ⚠️  Using placeholder action detection (integrate real model later)")
        
        import cv2
        import random
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Generate placeholder actions for testing
        # Make them appear near tracked players for better testing
        random.seed(42)
        
        # Use the 12 actual action classes from ball-action-spotting model
        action_types = [
            'PASS', 'DRIVE', 'HEADER', 'HIGH PASS', 'OUT', 'CROSS',
            'THROW IN', 'SHOT', 'BALL PLAYER BLOCK', 'PLAYER SUCCESSFUL TACKLE',
            'FREE KICK', 'GOAL'
        ]
        
        # Get frames where players are tracked
        tracked_frames = sorted(self.tracker.tracks.keys())
        
        if tracked_frames:
            # Generate actions near tracked frames for better testing
            num_actions = min(30, len(tracked_frames) // 10)
            actions = []
            
            for _ in range(num_actions):
                # Pick a random tracked frame
                base_frame = random.choice(tracked_frames)
                # Add small offset (±10 frames) to simulate action timing
                frame = max(0, min(total_frames - 1, base_frame + random.randint(-10, 10)))
                action = random.choice(action_types)
                confidence = random.uniform(0.6, 0.95)
                
                actions.append({
                    'frame': frame,
                    'action': action,
                    'confidence': confidence
                })
        else:
            # Fallback to random distribution if no tracks
            num_actions = min(15, total_frames // 50)
            actions = []
            
            for _ in range(num_actions):
                frame = random.randint(0, total_frames - 1)
                action = random.choice(action_types)
                confidence = random.uniform(0.6, 0.95)
                
                actions.append({
                    'frame': frame,
                    'action': action,
                    'confidence': confidence
                })
        
        return sorted(actions, key=lambda x: x['frame'])
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        # Convert non-serializable objects
        results_copy = results.copy()
        if 'actions' in results_copy:
            for action in results_copy['actions']:
                # Remove non-JSON-serializable fields if any
                if 'player_center' in action:
                    action['player_center'] = list(action['player_center'])
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_results(self, results: Dict):
        """Print results to console"""
        if not results:
            return
        
        print(f"\n{'='*60}")
        print(results['summary'])
        print(results['timeline'])
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='MVP: Count actions performed by a specific player in soccer video'
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--player', type=int, help='Player ID to track (optional)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Show frame to select player interactively')
    parser.add_argument('--yolo-model', default='yolov8n.pt',
                       help='YOLO model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--action-model', help='Path to ball action model')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--sample-rate', type=int, default=5,
                       help='Process every Nth frame (default: 5)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = PlayerActionCounter(
        yolo_model=args.yolo_model,
        action_model_path=args.action_model,
        device=args.device
    )
    
    # Process video
    results = counter.process_video(
        args.video,
        player_id=args.player,
        interactive=args.interactive,
        max_frames=args.max_frames,
        sample_rate=args.sample_rate
    )
    
    if results:
        # Print results
        counter.print_results(results)
        
        # Save if requested
        if args.output:
            counter.save_results(results, args.output)
        else:
            # Auto-generate output filename
            video_name = Path(args.video).stem
            output_path = f"player_{results['player_id']}_{video_name}.json"
            counter.save_results(results, output_path)


if __name__ == '__main__':
    main()
