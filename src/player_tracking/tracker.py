"""
Player tracking across video frames

This module tracks players across frames using YOLO's built-in tracking
or a simple centroid-based tracker as fallback.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from .detector import PlayerDetector, PlayerDetection


class PlayerTracker:
    """
    Track players across video frames
    
    Uses YOLO's built-in tracking (BoT-SORT/ByteTrack) for robust multi-object tracking.
    Maintains track history and provides utilities for accessing specific player tracks.
    """
    
    def __init__(
        self,
        detector: Optional[PlayerDetector] = None,
        model_name: str = 'yolov8n.pt',
        conf_threshold: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize player tracker
        
        Args:
            detector: Optional PlayerDetector instance (will create one if None)
            model_name: YOLO model to use if detector is None
            conf_threshold: Confidence threshold
            device: 'cuda' or 'cpu'
        """
        if detector is None:
            self.detector = PlayerDetector(
                model_name=model_name,
                conf_threshold=conf_threshold,
                device=device
            )
        else:
            self.detector = detector
        
        # Track storage: {frame_idx: {track_id: PlayerDetection}}
        self.tracks = defaultdict(dict)
        self.track_history = defaultdict(list)  # {track_id: [centers]}
    
    def track_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        sample_rate: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Dict[int, Dict[int, PlayerDetection]]:
        """
        Track all players in a video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)
            sample_rate: Process every Nth frame (1 = all frames)
            progress_callback: Optional callback function(frame_idx, total_frames)
            
        Returns:
            Dictionary mapping frame_idx -> {track_id: PlayerDetection}
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Tracking players in video: {total_frames} frames @ {fps:.1f} fps")
        
        frame_idx = 0
        
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                # Run YOLO with tracking
                results = self.detector.model.track(
                    frame,
                    persist=True,  # Enable tracking across frames
                    conf=self.detector.conf_threshold,
                    iou=self.detector.iou_threshold,
                    classes=[0],
                    verbose=False,
                    device=self.detector.device
                )
                
                # Process tracked detections
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    # Check if tracking IDs are available
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        
                        for box, track_id in zip(boxes, track_ids):
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0])
                            
                            # Filter by size
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            
                            if height >= self.detector.min_height and width >= self.detector.min_width:
                                detection = PlayerDetection(
                                    bbox=bbox,
                                    confidence=conf,
                                    track_id=track_id
                                )
                                
                                self.tracks[frame_idx][track_id] = detection
                                self.track_history[track_id].append(detection.center)
                
                # Progress callback
                if progress_callback and frame_idx % 100 == 0:
                    progress_callback(frame_idx, total_frames)
                elif frame_idx % 100 == 0:
                    print(f"  Processed {frame_idx}/{total_frames} frames...")
            
            frame_idx += 1
        
        cap.release()
        print(f"✓ Tracking complete: {len(self.tracks)} frames processed, {len(self.track_history)} unique players")
        
        return dict(self.tracks)
    
    def get_player_tracks(self, track_id: int) -> Dict[int, PlayerDetection]:
        """
        Get all detections for a specific player track
        
        Args:
            track_id: Player track ID
            
        Returns:
            Dictionary mapping frame_idx -> PlayerDetection
        """
        player_tracks = {}
        
        for frame_idx, players in self.tracks.items():
            if track_id in players:
                player_tracks[frame_idx] = players[track_id]
        
        return player_tracks
    
    def get_available_track_ids(self) -> List[int]:
        """
        Get list of all track IDs that appear in the video
        
        Returns:
            List of track IDs sorted by frequency (most common first)
        """
        from collections import Counter
        
        track_counts = Counter()
        for frame_players in self.tracks.values():
            for track_id in frame_players.keys():
                track_counts[track_id] += 1
        
        return [tid for tid, _ in track_counts.most_common()]
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """
        Get statistics for a specific track
        
        Args:
            track_id: Player track ID
            
        Returns:
            Dictionary with track statistics
        """
        player_tracks = self.get_player_tracks(track_id)
        
        if not player_tracks:
            return {
                'track_id': track_id,
                'num_frames': 0,
                'first_frame': None,
                'last_frame': None,
                'avg_confidence': 0.0
            }
        
        frames = sorted(player_tracks.keys())
        confidences = [det.confidence for det in player_tracks.values()]
        
        return {
            'track_id': track_id,
            'num_frames': len(player_tracks),
            'first_frame': frames[0],
            'last_frame': frames[-1],
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    def select_player_interactive(
        self,
        video_path: str,
        frame_idx: int = 0
    ) -> Optional[int]:
        """
        Show a frame and let user select which player to track
        
        Args:
            video_path: Path to video
            frame_idx: Frame to show for selection
            
        Returns:
            Selected track_id or None
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Cannot read frame {frame_idx}")
            return None
        
        # Run tracking on this frame
        results = self.detector.model.track(
            frame,
            persist=True,
            conf=self.detector.conf_threshold,
            classes=[0],
            verbose=False,
            device=self.detector.device
        )
        
        if len(results) == 0 or results[0].boxes is None or results[0].boxes.id is None:
            print("No tracked players found in frame")
            return None
        
        boxes = results[0].boxes
        track_ids = boxes.id.cpu().numpy().astype(int)
        
        # Draw bounding boxes with track IDs
        display_frame = frame.copy()
        
        for box, track_id in zip(boxes, track_ids):
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            cv2.rectangle(
                display_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                f"ID: {track_id}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Show frame
        cv2.imshow("Select Player - Press ESC when done", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Get user input
        print(f"\nAvailable player IDs: {sorted(track_ids.tolist())}")
        try:
            player_id = int(input("Enter player ID to track: "))
            if player_id in track_ids:
                return player_id
            else:
                print(f"Invalid ID. Must be one of: {sorted(track_ids.tolist())}")
                return None
        except ValueError:
            print("Invalid input")
            return None
    
    def visualize_tracks(
        self,
        video_path: str,
        output_path: str,
        track_ids: Optional[List[int]] = None,
        max_frames: Optional[int] = None
    ):
        """
        Create visualization video with tracked players
        
        Args:
            video_path: Input video path
            output_path: Output video path
            track_ids: Optional list of track IDs to highlight (None = all)
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw tracks for this frame
            if frame_idx in self.tracks:
                for track_id, detection in self.tracks[frame_idx].items():
                    # Skip if filtering by track_ids
                    if track_ids is not None and track_id not in track_ids:
                        continue
                    
                    bbox = [int(x) for x in detection.bbox]
                    
                    # Draw bbox
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        2
                    )
                    
                    # Draw ID and confidence
                    label = f"ID:{track_id} {detection.confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"✓ Visualization saved to: {output_path}")
