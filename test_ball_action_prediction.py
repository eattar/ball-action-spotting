#!/usr/bin/env python3
"""
Test ball-action-spotting model prediction

Tests the original ball action detection on a video to verify it works.
"""

import sys
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

from src.utils import get_video_info, post_processing
from src.predictors import MultiDimStackerPredictor
from src.frame_fetchers import OpencvFrameFetcher, NvDecFrameFetcher
from src.ball_action import constants


def test_prediction(model_path: str, video_path: str, use_nvdec: bool = False):
    """
    Test ball action prediction on a video
    
    Args:
        model_path: Path to trained model .pth file
        video_path: Path to video file
        use_nvdec: Use NvDec if available (GPU accelerated decode)
    """
    print("=" * 60)
    print("Ball-Action-Spotting Model Test")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Video: {video_path}")
    
    # Load model
    print("\nLoading model...")
    predictor = MultiDimStackerPredictor(
        model_path,
        device="cuda:0",
        tta=True  # Test-time augmentation
    )
    print(f"✓ Model loaded on: {predictor.device}")
    
    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info["frame_count"]
    fps = video_info["fps"]
    print(f"\nVideo info:")
    print(f"  Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    # Create frame fetcher
    if use_nvdec:
        try:
            print("\nUsing NvDec frame fetcher (GPU accelerated)...")
            frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=0)
        except Exception as e:
            print(f"  NvDec failed: {e}")
            print("  Falling back to OpenCV...")
            frame_fetcher = OpencvFrameFetcher(video_path, gpu_id=0)
    else:
        print("\nUsing OpenCV frame fetcher (CPU decode, GPU inference)...")
        frame_fetcher = OpencvFrameFetcher(video_path, gpu_id=0)
    
    frame_fetcher.num_frames = total_frames
    
    # Run inference
    print(f"\nRunning inference on {total_frames} frames...")
    indexes_generator = predictor.indexes_generator
    INDEX_SAVE_ZONE = 1
    min_frame_index = indexes_generator.clip_index(0, total_frames, INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(total_frames, total_frames, INDEX_SAVE_ZONE)
    
    frame_index2prediction = dict()
    predictor.reset_buffers()
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            frame = frame_fetcher.fetch_frame()
            frame_index = frame_fetcher.current_index
            prediction, predict_index = predictor.predict(frame, frame_index)
            
            if predict_index < min_frame_index:
                pbar.update(1)
                continue
            
            if prediction is not None:
                frame_index2prediction[predict_index] = prediction.cpu().numpy()
            
            pbar.update(1)
            
            if predict_index == max_frame_index:
                break
    
    predictor.reset_buffers()
    
    # Convert predictions to actions
    print("\nProcessing predictions...")
    frame_indexes = sorted(frame_index2prediction.keys())
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    
    print(f"  Prediction shape: {raw_predictions.shape}")
    print(f"  Number of classes: {raw_predictions.shape[1]}")
    
    # Check if binary or multi-class
    if raw_predictions.shape[1] == 2:
        print("\n  Model type: BINARY (ball action vs no action)")
        action_scores = raw_predictions[:, 1]  # Positive class
        
        # Use post-processing to find peaks
        action_frames, action_confidences = post_processing(
            frame_indexes,
            action_scores,
            **constants.postprocess_params
        )
        
        print(f"\n✓ Detected {len(action_frames)} ball action events")
        
        # Show first 10 actions
        print("\nFirst 10 detected actions:")
        print(f"{'Frame':<10} {'Time':<12} {'Confidence':<12}")
        print("-" * 40)
        for i, (frame, conf) in enumerate(zip(action_frames[:10], action_confidences[:10])):
            seconds = frame / fps
            time_str = f"{int(seconds//60)}:{int(seconds%60):02d}"
            print(f"{frame:<10} {time_str:<12} {conf:<12.3f}")
        
        if len(action_frames) > 10:
            print(f"... and {len(action_frames) - 10} more actions")
    
    else:
        print(f"\n  Model type: MULTI-CLASS ({raw_predictions.shape[1]} classes)")
        
        # Process each class
        from src.ball_action.annotations import raw_predictions_to_actions
        class2actions = raw_predictions_to_actions(frame_indexes, raw_predictions)
        
        total_actions = sum(len(actions[0]) for actions in class2actions.values())
        print(f"\n✓ Detected {total_actions} total actions across all classes")
        
        print("\nActions by class:")
        for class_name, (frames, confidences) in sorted(class2actions.items()):
            if len(frames) > 0:
                avg_conf = np.mean(confidences)
                print(f"  {class_name:<25}: {len(frames):>4} actions (avg conf: {avg_conf:.3f})")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test ball-action-spotting model on a video"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model .pth file"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--nvdec",
        action="store_true",
        help="Use NvDec for GPU accelerated video decoding"
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        sys.exit(1)
    
    test_prediction(args.model_path, args.video_path, args.nvdec)


if __name__ == "__main__":
    main()
