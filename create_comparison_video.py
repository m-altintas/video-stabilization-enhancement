"""
Create a side-by-side comparison video from three input videos.

Usage:
    python create_comparison_video.py -i video1.mp4 video2.mp4 video3.mp4 -o output.mp4
    python create_comparison_video.py --labels "Original" "DSTNet" "RealESRGAN" -o comparison.mp4
"""

import cv2 as cv
import numpy as np
import argparse
import os
from typing import List, Tuple


def add_text_label(frame: np.ndarray, text: str, position: str = 'top-left') -> np.ndarray:
    """
    Add a text label to a frame.
    
    Args:
        frame: Input frame
        text: Text to add
        position: Position of text ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    
    Returns:
        Frame with text label
    """
    frame_copy = frame.copy()
    height, width = frame.shape[:2]
    
    # Font settings
    font = cv.FONT_HERSHEY_DUPLEX  # Bold-looking font
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background
    
    # Get text size
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    
    # Determine position
    padding = 10
    if position == 'top-left':
        x = padding
        y = text_height + padding
    elif position == 'top-right':
        x = width - text_width - padding
        y = text_height + padding
    elif position == 'bottom-left':
        x = padding
        y = height - baseline - padding
    elif position == 'bottom-right':
        x = width - text_width - padding
        y = height - baseline - padding
    else:
        x = padding
        y = text_height + padding
    
    # Draw background rectangle
    cv.rectangle(
        frame_copy,
        (x - padding//2, y - text_height - padding//2),
        (x + text_width + padding//2, y + baseline + padding//2),
        bg_color,
        -1
    )
    
    # Draw text
    cv.putText(frame_copy, text, (x, y), font, font_scale, color, thickness)
    
    return frame_copy


def create_comparison_video(
    video_paths: List[str],
    output_path: str,
    labels: List[str] = None,
    fps: float = None
) -> bool:
    """
    Create a side-by-side comparison video from multiple input videos.
    
    Args:
        video_paths: List of paths to input videos
        output_path: Path to save comparison video
        labels: Optional list of labels for each video
        fps: Output FPS (uses first video's FPS if not specified)
    
    Returns:
        True if successful, False otherwise
    """
    if len(video_paths) < 2:
        print("Error: Need at least 2 videos for comparison")
        return False
    
    # Check if all videos exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            return False
    
    # Open all video files
    caps = []
    video_props = []
    
    for video_path in video_paths:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv.CAP_PROP_FPS)
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        caps.append(cap)
        video_props.append({
            'width': width,
            'height': height,
            'fps': video_fps,
            'n_frames': n_frames,
            'path': video_path
        })
        
        print(f"Loaded: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height}, FPS: {video_fps:.2f}, Frames: {n_frames}")
    
    # Use FPS from first video or specified FPS
    output_fps = fps if fps is not None else video_props[0]['fps']
    
    # Use dimensions from first video (assume all are same size)
    frame_height = video_props[0]['height']
    frame_width = video_props[0]['width']
    
    # Calculate output dimensions (side-by-side)
    output_width = frame_width * len(video_paths)
    output_height = frame_height
    
    print(f"\nOutput video: {output_width}x{output_height} @ {output_fps:.2f} FPS")
    
    # Create output video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        for cap in caps:
            cap.release()
        return False
    
    # Default labels
    if labels is None:
        labels = [f"Video {i+1}" for i in range(len(video_paths))]
    elif len(labels) < len(video_paths):
        # Extend labels if not enough provided
        labels.extend([f"Video {i+1}" for i in range(len(labels), len(video_paths))])
    
    # Process frames
    frame_count = 0
    min_frames = min(vp['n_frames'] for vp in video_props)
    
    print(f"\nProcessing {min_frames} frames...")
    
    while frame_count < min_frames:
        frames = []
        
        # Read frame from each video
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # If we can't read a frame, break the loop
                print(f"Warning: Could not read frame {frame_count} from video {i+1}")
                break
            
            # Resize frame if needed (shouldn't be necessary, but just in case)
            if frame.shape[:2] != (frame_height, frame_width):
                frame = cv.resize(frame, (frame_width, frame_height))
            
            # Add label
            if labels and i < len(labels):
                frame = add_text_label(frame, labels[i], position='top-left')
            
            frames.append(frame)
        
        # If we didn't get all frames, break
        if len(frames) != len(caps):
            break
        
        # Combine frames horizontally
        combined_frame = np.hstack(frames)
        
        # Write combined frame
        out.write(combined_frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{min_frames} frames ({100*frame_count/min_frames:.1f}%)")
    
    # Cleanup
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"\nDone! Comparison video saved to: {output_path}")
    print(f"Total frames: {frame_count}")
    
    return True


def main():
    """Command-line interface for creating comparison videos."""
    parser = argparse.ArgumentParser(
        description='Create a side-by-side comparison video from multiple input videos'
    )
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Paths to input videos (2 or more)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output comparison video'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Labels for each video (optional)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output FPS (uses first video FPS if not specified)'
    )
    
    args = parser.parse_args()
    
    # Create comparison video
    success = create_comparison_video(
        video_paths=args.input,
        output_path=args.output,
        labels=args.labels,
        fps=args.fps
    )
    
    if success:
        print("\nComparison video created successfully!")
    else:
        print("\nFailed to create comparison video. Check error messages above.")


if __name__ == "__main__":
    main()

