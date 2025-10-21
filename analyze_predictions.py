#!/usr/bin/env python3
"""
Quick script to analyze prediction outputs and show what classes are detected.
"""
import json
import numpy as np
from pathlib import Path
import sys

def analyze_npz(npz_path):
    """Analyze raw predictions NPZ file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {npz_path}")
    print(f"{'='*60}")
    
    data = np.load(npz_path)
    frame_indexes = data['frame_indexes']
    raw_predictions = data['raw_predictions']
    
    print(f"\nFrame indexes shape: {frame_indexes.shape}")
    print(f"Raw predictions shape: {raw_predictions.shape}")
    print(f"Number of classes: {raw_predictions.shape[1]}")
    print(f"\nFirst 5 frame indexes: {frame_indexes[:5]}")
    print(f"Last 5 frame indexes: {frame_indexes[-5:]}")
    
    # Show prediction statistics
    print(f"\nPrediction statistics:")
    print(f"  Min: {raw_predictions.min():.4f}")
    print(f"  Max: {raw_predictions.max():.4f}")
    print(f"  Mean: {raw_predictions.mean():.4f}")
    
    # Show per-class max predictions
    print(f"\nPer-class max predictions:")
    for i in range(raw_predictions.shape[1]):
        print(f"  Class {i}: {raw_predictions[:, i].max():.4f}")


def analyze_json(json_path):
    """Analyze results JSON file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {json_path}")
    print(f"{'='*60}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count predictions by label
    label_counts = {}
    all_predictions = data['predictions']
    
    for pred in all_predictions:
        label = pred['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nGame: {data['UrlLocal']}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Unique labels: {list(label_counts.keys())}")
    print(f"\nPredictions by label:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    
    # Show confidence statistics
    if all_predictions:
        confidences = [float(p['confidence']) for p in all_predictions]
        print(f"\nConfidence statistics:")
        print(f"  Min: {min(confidences):.4f}")
        print(f"  Max: {max(confidences):.4f}")
        print(f"  Mean: {sum(confidences)/len(confidences):.4f}")
        
    # Show some example predictions
    print(f"\nFirst 5 predictions:")
    for i, pred in enumerate(all_predictions[:5]):
        print(f"  {i+1}. {pred['gameTime']} - {pred['label']} (conf: {float(pred['confidence']):.4f}, pos: {pred['position']})")
    
    print(f"\nLast 5 predictions:")
    for i, pred in enumerate(all_predictions[-5:]):
        print(f"  {len(all_predictions)-4+i}. {pred['gameTime']} - {pred['label']} (conf: {float(pred['confidence']):.4f}, pos: {pred['position']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_predictions.py <npz_or_json_file>")
        print("\nOr analyze entire experiment:")
        print("python analyze_predictions.py /path/to/predictions/sampling_weights_001/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_dir():
        # Analyze all files in directory
        for npz_file in path.rglob("*_raw_predictions.npz"):
            analyze_npz(npz_file)
        for json_file in path.rglob("results_spotting.json"):
            analyze_json(json_file)
    elif path.suffix == '.npz':
        analyze_npz(path)
    elif path.suffix == '.json':
        analyze_json(path)
    else:
        print(f"Unknown file type: {path}")
        sys.exit(1)
