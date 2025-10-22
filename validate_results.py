#!/usr/bin/env python3
"""
Quick validation script for player tracking results.
Analyzes match quality, distribution, and flags potential issues.
"""

import json
import sys
import numpy as np
from pathlib import Path


def validate_results(json_path: str):
    """Validate player tracking results"""
    
    with open(json_path) as f:
        data = json.load(f)
    
    actions = data['matched_actions']
    
    if not actions:
        print("❌ No matched actions found!")
        return
    
    print('='*60)
    print('PLAYER TRACKING VALIDATION REPORT')
    print('='*60)
    print(f'\nPlayer ID: {data["player_id"]}')
    print(f'Video: {data["video"]}')
    
    # Quality indicators
    confidences = [a['confidence'] for a in actions]
    offsets = [abs(a.get('frame_offset', 0)) for a in actions]
    
    print(f'\n1. MATCH QUALITY:')
    print(f'   Total matched actions: {len(actions)}')
    high_conf = sum(1 for c in confidences if c > 0.5)
    print(f'   High confidence (>0.5): {high_conf}/{len(confidences)} ({high_conf/len(confidences)*100:.1f}%)')
    perfect_match = sum(1 for o in offsets if o == 0)
    print(f'   Perfect temporal match (offset=0): {perfect_match}/{len(offsets)} ({perfect_match/len(offsets)*100:.1f}%)')
    print(f'   Mean confidence: {np.mean(confidences):.3f}')
    print(f'   Mean frame offset: {np.mean(offsets):.1f} frames ({np.mean(offsets)/25:.2f}s)')
    
    print(f'\n2. ACTION DISTRIBUTION:')
    stats = data['statistics']
    total = sum(stats.values())
    for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f'   {label}: {count} ({count/total*100:.1f}%)')
    
    print(f'\n3. TEMPORAL SPREAD:')
    frames = [a['frame'] for a in actions]
    game_minutes = (max(frames) - min(frames)) / 25 / 60
    print(f'   First action: frame {min(frames):,} ({min(frames)/25/60:.1f} min)')
    print(f'   Last action: frame {max(frames):,} ({max(frames)/25/60:.1f} min)')
    print(f'   Active period: {game_minutes:.1f} minutes')
    print(f'   Actions per minute: {len(actions)/game_minutes:.2f}')
    
    print(f'\n4. CONFIDENCE BY ACTION TYPE:')
    by_type = {}
    for a in actions:
        label = a['label']
        if label not in by_type:
            by_type[label] = []
        by_type[label].append(a['confidence'])
    
    for label, confs in sorted(by_type.items()):
        print(f'   {label}: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}, '
              f'min={min(confs):.3f}, max={max(confs):.3f}')
    
    print(f'\n5. FRAME OFFSET ANALYSIS:')
    print(f'   Min offset: {min(offsets)} frames')
    print(f'   Max offset: {max(offsets)} frames ({max(offsets)/25:.2f}s)')
    print(f'   Within 1 sec: {sum(1 for o in offsets if o <= 25)}/{len(offsets)} ({sum(1 for o in offsets if o <= 25)/len(offsets)*100:.1f}%)')
    
    print(f'\n6. QUALITY FLAGS:')
    flags = []
    warnings = []
    
    if np.mean(confidences) < 0.3:
        flags.append('❌ Very low average confidence (<0.3)')
    elif np.mean(confidences) < 0.4:
        warnings.append('⚠️  Low average confidence (0.3-0.4)')
    
    if max(offsets) > 50:
        flags.append('❌ Some matches are >2 seconds away')
    elif max(offsets) > 25:
        warnings.append('⚠️  Some matches are 1-2 seconds away')
    
    if game_minutes < 5:
        flags.append('❌ Player only active for <5 minutes (possible tracking loss)')
    elif game_minutes < 10:
        warnings.append('⚠️  Player active for <10 minutes')
    
    if len(actions) < 10:
        flags.append('❌ Very few actions matched (<10)')
    elif len(actions) < 20:
        warnings.append('⚠️  Few actions matched (10-20)')
    
    if perfect_match / len(offsets) < 0.3:
        warnings.append('⚠️  Less than 30% perfect temporal matches')
    
    if high_conf / len(confidences) < 0.4:
        warnings.append('⚠️  Less than 40% high-confidence matches')
    
    if flags:
        for flag in flags:
            print(f'   {flag}')
    if warnings:
        for warning in warnings:
            print(f'   {warning}')
    if not flags and not warnings:
        print('   ✅ No obvious issues detected')
    
    # Overall assessment
    print(f'\n7. OVERALL ASSESSMENT:')
    score = 0
    max_score = 5
    
    if np.mean(confidences) >= 0.4:
        score += 1
    if high_conf / len(confidences) >= 0.5:
        score += 1
    if np.mean(offsets) <= 15:
        score += 1
    if len(actions) >= 20:
        score += 1
    if game_minutes >= 10:
        score += 1
    
    rating = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    rating_idx = min(int(score / max_score * len(rating)), len(rating) - 1)
    
    print(f'   Score: {score}/{max_score}')
    print(f'   Rating: {rating[rating_idx]}')
    
    if score >= 4:
        print(f'   ✅ Results look reliable')
    elif score >= 3:
        print(f'   ⚠️  Results are acceptable but could be improved')
    else:
        print(f'   ❌ Results may need review or parameter tuning')
    
    print('\n' + '='*60)
    
    # Suggestions
    if flags or warnings:
        print('\nSUGGESTIONS FOR IMPROVEMENT:')
        if np.mean(confidences) < 0.4:
            print('  • Increase confidence threshold to filter low-quality matches')
        if max(offsets) > 25:
            print('  • Reduce temporal_window to get tighter matches')
        if len(actions) < 20:
            print('  • Try increasing temporal_window to catch more actions')
            print('  • Check if player was actually on screen during the game')
        if game_minutes < 10:
            print('  • Player may be a substitute - check game time')
            print('  • Track ID may have changed (ByteTrack limitation)')
        print()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python validate_results.py <player_stats.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    validate_results(json_path)
