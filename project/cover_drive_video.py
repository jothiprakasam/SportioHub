#!/usr/bin/env python3
"""
Cricket Cover Drive Pose Analyzer (CLI Tool)
Analyzes cricket batting pose from video footage against a reference pro shot.
Provides real-time skeleton overlays, angle calculations, and actionable coaching recommendations.
"""

import os
import sys
import argparse

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cricket_pose_engine import analyze_cricket_videos

def main():
    parser = argparse.ArgumentParser(
        description="SportioHub Cricket Pose Analyzer: Analyze batting form against reference videos."
    )
    parser.add_argument(
        "--default", "-d",
        default=os.path.join(BASE_DIR, "default.mp4"),
        help="Path to the reference/pro video (default: project/default.mp4)"
    )
    parser.add_argument(
        "--user", "-u",
        default=os.path.join(BASE_DIR, "comparison.mp4"),
        help="Path to the user/athlete video (default: project/comparison.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(BASE_DIR, "static", "outputs", "cricket_analysis_output.mp4"),
        help="Path for saving analyzed comparison video"
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum frames to process (useful for quick testing)"
    )

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    snapshots_dir = os.path.join(BASE_DIR, "static", "outputs")
    os.makedirs(snapshots_dir, exist_ok=True)

    print("=" * 65)
    print("🏏 SPORTIOHUB CRICKET POSE ANALYZER (COVER DRIVE) 🏏")
    print("=" * 65)
    print(f"Reference Video : {args.default}")
    print(f"User Video      : {args.user}")
    print(f"Output Video    : {args.output}")
    print("-" * 65)

    if not os.path.exists(args.default):
        print(f"Error: Reference video not found at '{args.default}'")
        sys.exit(1)
    if not os.path.exists(args.user):
        print(f"Error: User video not found at '{args.user}'")
        sys.exit(1)

    print("Processing video frames and computing biomechanical angles...")

    def progress_callback(current, total):
        pct = (current / total) * 100
        bar = "█" * int(pct // 5) + "-" * (20 - int(pct // 5))
        sys.stdout.write(f"\rProgress: [{bar}] {pct:.1f}% ({current}/{total} frames)")
        sys.stdout.flush()

    report = analyze_cricket_videos(
        args.default,
        args.user,
        args.output,
        snapshots_dir=snapshots_dir,
        max_frames=args.max_frames,
        progress_callback=progress_callback
    )

    print("\n" + "=" * 65)
    print("📊 COACHING PERFORMANCE REPORT CARD")
    print("=" * 65)
    print(f"Overall Form Score : {report['overall_score']}%  [{report['grade']}]")
    print(f"Lead Elbow Delta   : ~{report['lead_elbow_diff']}° deviation from pro reference")
    print(f"Front Knee Delta   : ~{report['front_knee_diff']}° deviation from pro reference")
    print(f"Frames Analyzed    : {report['total_frames_analyzed']}")
    print(f"\nVerdict: {report['summary']}")

    if report["strengths"]:
        print("\n✅ Key Strengths:")
        for s in report["strengths"]:
            print(f"  • {s}")

    if report["flaws"]:
        print("\n⚠️ Areas to Correct:")
        for f in report["flaws"]:
            print(f"  • {f}")

    if report["drills"]:
        print("\n🏏 Recommended Practice Drills:")
        for d in report["drills"]:
            print(f"  • {d['name']}: {d['description']}")

    print("\n📁 Keyframe Snapshots Saved:")
    for phase, fname in report["keyframes"].items():
        print(f"  • {phase}: {fname}")

    print(f"\n🎥 Analyzed Video Ready: {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
