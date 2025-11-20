"""
Player detection using YOLOv8

This module uses ultralytics YOLO for detecting players (person class) in soccer videos.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("⚠️  ultralytics not installed. Run: pip install ultralytics")


@dataclass
class PlayerDetection:
    """Single player detection"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def width(self) -> float:
        """Get width of bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return self.width * self.height


class PlayerDetector:
    """
    Player detector using YOLOv8
    
    Detects players (person class) in video frames and filters by size constraints
    to focus on actual players rather than distant spectators or referees.
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        min_height: int = 30,
        min_width: int = 15,
        device: str = 'cuda',
        classes: List[int] = None
    ):
        """
        Initialize player detector
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
                       n=nano (fastest), s=small, m=medium, l=large, x=xlarge (most accurate)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IOU threshold for non-maximum suppression
            min_height: Minimum bbox height in pixels (filters out distant/small detections)
            min_width: Minimum bbox width in pixels
            device: 'cuda' or 'cpu'
            classes: List of COCO class IDs to detect (default: [0] for person)
        """
        if not HAS_YOLO:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_height = min_height
        self.min_width = min_width
        self.device = device
        self.classes = classes if classes is not None else [0]
        
        # Warm up model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False, device=device)
        print(f"✓ Detector initialized on {device} (classes: {self.classes})")
    
    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        Detect objects in a single frame
        
        Args:
            frame: BGR image (opencv format)
            
        Returns:
            List of PlayerDetection objects
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
            device=self.device
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                
                # Filter by size (players should be reasonable size, not distant spectators)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if height >= self.min_height and width >= self.min_width:
                    detections.append(PlayerDetection(
                        bbox=bbox,
                        confidence=conf
                    ))
        
        return detections
    
    def detect_batch(
        self, 
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[PlayerDetection]]:
        """
        Detect players in multiple frames (batch processing for efficiency)
        
        Args:
            frames: List of BGR images
            batch_size: Number of frames to process at once
            
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            # Run batch inference
            results = self.model(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                verbose=False,
                device=self.device
            )
            
            # Process each result
            for result in results:
                frame_detections = []
                
                if result.boxes is not None:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0])
                        
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        if height >= self.min_height and width >= self.min_width:
                            frame_detections.append(PlayerDetection(
                                bbox=bbox,
                                confidence=conf
                            ))
                
                all_detections.append(frame_detections)
        
        return all_detections
    
    def visualize(
        self, 
        frame: np.ndarray, 
        detections: List[PlayerDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detections on frame for visualization
        
        Args:
            frame: BGR image
            detections: List of PlayerDetection objects
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence
            label = f"{det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
        
        return annotated
