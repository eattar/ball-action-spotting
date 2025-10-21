"""
Match detected actions to tracked players

This module associates ball actions detected by the action spotting model
with specific tracked players based on temporal and spatial proximity.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import Counter
from .detector import PlayerDetection


class ActionPlayerMatcher:
    """
    Associate detected actions with tracked players
    
    Uses temporal proximity (action detected when player is visible) and
    spatial proximity (action location near player bbox) to match actions to players.
    """
    
    def __init__(
        self,
        temporal_window: int = 25,
        spatial_threshold: float = 150.0,
        fps: float = 25.0
    ):
        """
        Initialize matcher
        
        Args:
            temporal_window: Frames before/after action to check for player visibility
                           Default 25 frames = 1 second at 25fps
            spatial_threshold: Maximum distance (pixels) from player center to consider match
            fps: Frames per second (for timestamp calculations)
        """
        self.temporal_window = temporal_window
        self.spatial_threshold = spatial_threshold
        self.fps = fps
    
    def match_actions_to_player(
        self,
        actions: List[Dict],
        player_tracks: Dict[int, PlayerDetection],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Match detected actions to a specific tracked player
        
        Args:
            actions: List of action dicts with keys: frame, action, confidence
            player_tracks: Dict mapping frame_idx -> PlayerDetection for target player
            verbose: Print matching statistics
            
        Returns:
            List of matched actions (subset of input actions that match the player)
        """
        if verbose:
            print(f"Matching {len(actions)} actions to player visible in {len(player_tracks)} frames...")
        
        matched_actions = []
        
        for action in actions:
            action_frame = action['frame']
            
            # Check if player is visible in nearby frames (temporal proximity)
            matched_frame = None
            min_offset = float('inf')
            
            for offset in range(-self.temporal_window, self.temporal_window + 1):
                check_frame = action_frame + offset
                
                if check_frame in player_tracks:
                    if abs(offset) < abs(min_offset):
                        matched_frame = check_frame
                        min_offset = offset
            
            if matched_frame is not None:
                # Action detected when player was visible
                matched_action = action.copy()
                matched_action['player_frame'] = matched_frame
                matched_action['frame_offset'] = min_offset
                matched_action['player_bbox'] = player_tracks[matched_frame].bbox
                matched_action['player_center'] = player_tracks[matched_frame].center
                matched_action['player_confidence'] = player_tracks[matched_frame].confidence
                
                matched_actions.append(matched_action)
        
        # Remove duplicate detections (same action detected in consecutive frames)
        unique_actions = self._remove_duplicate_actions(matched_actions)
        
        if verbose:
            print(f"✓ Matched {len(unique_actions)} actions to player (removed {len(matched_actions) - len(unique_actions)} duplicates)")
        
        return unique_actions
    
    def match_actions_to_multiple_players(
        self,
        actions: List[Dict],
        all_player_tracks: Dict[int, Dict[int, PlayerDetection]],
        action_locations: Optional[Dict[int, tuple]] = None
    ) -> Dict[int, List[Dict]]:
        """
        Match actions to multiple players (determine which player performed each action)
        
        Args:
            actions: List of action dicts
            all_player_tracks: Dict mapping frame_idx -> {track_id: PlayerDetection}
            action_locations: Optional dict mapping action_frame -> (x, y) location
            
        Returns:
            Dict mapping track_id -> List[matched_actions]
        """
        player_actions = {}
        
        for action in actions:
            action_frame = action['frame']
            action_loc = action_locations.get(action_frame) if action_locations else None
            
            # Find all players visible near this action
            candidate_players = []
            
            for offset in range(-self.temporal_window, self.temporal_window + 1):
                check_frame = action_frame + offset
                
                if check_frame in all_player_tracks:
                    for track_id, detection in all_player_tracks[check_frame].items():
                        # Calculate proximity score
                        if action_loc:
                            distance = np.sqrt(
                                (detection.center[0] - action_loc[0])**2 +
                                (detection.center[1] - action_loc[1])**2
                            )
                        else:
                            # No action location, just use temporal proximity
                            distance = abs(offset) * 10  # Penalize temporal distance
                        
                        candidate_players.append({
                            'track_id': track_id,
                            'frame': check_frame,
                            'offset': offset,
                            'distance': distance,
                            'detection': detection
                        })
            
            if candidate_players:
                # Select closest player
                best_match = min(candidate_players, key=lambda x: x['distance'])
                
                # Only match if within threshold
                if best_match['distance'] <= self.spatial_threshold:
                    track_id = best_match['track_id']
                    
                    matched_action = action.copy()
                    matched_action['player_frame'] = best_match['frame']
                    matched_action['frame_offset'] = best_match['offset']
                    matched_action['player_bbox'] = best_match['detection'].bbox
                    matched_action['player_center'] = best_match['detection'].center
                    matched_action['distance'] = best_match['distance']
                    
                    if track_id not in player_actions:
                        player_actions[track_id] = []
                    player_actions[track_id].append(matched_action)
        
        # Remove duplicates for each player
        for track_id in player_actions:
            player_actions[track_id] = self._remove_duplicate_actions(player_actions[track_id])
        
        return player_actions
    
    def _remove_duplicate_actions(
        self,
        actions: List[Dict],
        frame_threshold: int = 25
    ) -> List[Dict]:
        """
        Remove duplicate detections of the same action
        
        Same action type detected within frame_threshold are considered duplicates.
        Keeps the one with highest confidence.
        
        Args:
            actions: List of action dicts
            frame_threshold: Frames within which same action is considered duplicate
            
        Returns:
            Filtered list of unique actions
        """
        if not actions:
            return []
        
        # Sort by frame
        sorted_actions = sorted(actions, key=lambda x: x['frame'])
        
        unique = [sorted_actions[0]]
        
        for action in sorted_actions[1:]:
            last_action = unique[-1]
            
            # Check if this is a different action or far enough in time
            if (action['action'] != last_action['action'] or
                abs(action['frame'] - last_action['frame']) > frame_threshold):
                unique.append(action)
            # Same action within threshold - keep higher confidence
            elif action['confidence'] > last_action['confidence']:
                unique[-1] = action
        
        return unique
    
    def count_actions(self, actions: List[Dict]) -> Dict[str, int]:
        """
        Count actions by type
        
        Args:
            actions: List of matched actions
            
        Returns:
            Dictionary mapping action_type -> count
        """
        action_types = [a['action'] for a in actions]
        return dict(Counter(action_types))
    
    def create_summary(
        self,
        actions: List[Dict],
        track_id: Optional[int] = None
    ) -> str:
        """
        Create human-readable summary of player actions
        
        Args:
            actions: List of matched actions
            track_id: Optional player track ID for header
            
        Returns:
            Formatted summary string
        """
        if not actions:
            return "No actions detected for this player."
        
        action_counts = self.count_actions(actions)
        
        lines = []
        
        if track_id is not None:
            lines.append(f"Player #{track_id} Action Summary")
            lines.append("=" * 60)
        
        lines.append(f"\nTotal Actions: {len(actions)}")
        lines.append("\nAction Breakdown:")
        
        for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(actions)) * 100
            lines.append(f"  {action_type:25s}: {count:2d} ({percentage:5.1f}%)")
        
        # Temporal info
        if actions:
            first_action = min(actions, key=lambda x: x['frame'])
            last_action = max(actions, key=lambda x: x['frame'])
            
            first_time = first_action['frame'] / self.fps
            last_time = last_action['frame'] / self.fps
            duration = last_time - first_time
            
            lines.append(f"\nTemporal Range:")
            lines.append(f"  First action: {self._format_time(first_time)} (frame {first_action['frame']})")
            lines.append(f"  Last action:  {self._format_time(last_time)} (frame {last_action['frame']})")
            lines.append(f"  Duration:     {self._format_time(duration)}")
            
            if duration > 0:
                actions_per_min = (len(actions) / duration) * 60
                lines.append(f"  Rate:         {actions_per_min:.1f} actions/min")
        
        return "\n".join(lines)
    
    def create_timeline(
        self,
        actions: List[Dict],
        max_items: int = 20
    ) -> str:
        """
        Create timeline of actions
        
        Args:
            actions: List of matched actions
            max_items: Maximum number of actions to show
            
        Returns:
            Formatted timeline string
        """
        if not actions:
            return "No actions in timeline."
        
        sorted_actions = sorted(actions, key=lambda x: x['frame'])
        
        lines = []
        lines.append("\nAction Timeline:")
        lines.append("-" * 60)
        
        for i, action in enumerate(sorted_actions[:max_items], 1):
            time_str = self._format_time(action['frame'] / self.fps)
            conf_str = f"{action['confidence']:.1%}"
            
            lines.append(f"  {i:2d}. [{time_str}] {action['action']:20s} ({conf_str})")
        
        if len(sorted_actions) > max_items:
            lines.append(f"  ... and {len(sorted_actions) - max_items} more")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
