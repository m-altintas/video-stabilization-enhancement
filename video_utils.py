"""
Video utility functions for reading and preserving video properties.
"""

import cv2 as cv
import subprocess
import json
import os
from typing import Dict, Optional, Tuple


def get_video_properties_ffprobe(video_path: str) -> Optional[Dict]:
    """
    Get detailed video properties using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties including bitrate, codec, etc.
        Returns None if ffprobe is not available or fails.
    """
    try:
        # Use ffprobe to get detailed video information
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # Extract properties
        format_info = data.get('format', {})
        
        # Parse FPS (format: "30/1" -> 30.0)
        fps_str = video_stream.get('r_frame_rate', '0/1')
        try:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        
        properties = {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': fps,
            'codec': video_stream.get('codec_name', 'unknown'),
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'bitrate': int(format_info.get('bit_rate', 0)),  # in bits per second
            'duration': float(format_info.get('duration', 0)),
            'size': int(format_info.get('size', 0)),
        }
        
        # Try to get average bitrate from video stream if format bitrate is 0
        if properties['bitrate'] == 0:
            bitrate_str = video_stream.get('bit_rate')
            if bitrate_str:
                properties['bitrate'] = int(bitrate_str)
        
        return properties
        
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not get video properties with ffprobe: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error reading video properties: {e}")
        return None


def get_video_properties_opencv(video_path: str) -> Dict:
    """
    Get basic video properties using OpenCV.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with basic video properties
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    properties = {
        'width': int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
    }
    
    cap.release()
    return properties


def encode_video_with_ffmpeg(
    input_video: str,
    output_video: str,
    target_bitrate: Optional[int] = None,
    codec: str = 'libx264',
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    preset: str = 'medium',
    crf: Optional[int] = None
) -> bool:
    """
    Encode video using ffmpeg with specified bitrate and codec settings.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        target_bitrate: Target bitrate in bits per second (e.g., 5000000 for 5 Mbps)
                       If None, uses CRF mode instead
        codec: Video codec to use (default: libx264)
        fps: Frame rate (uses input if None)
        width: Output width (uses input if None)
        height: Output height (uses input if None)
        preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        crf: Constant Rate Factor (18-28, lower = better quality). Used if target_bitrate is None.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Cannot encode with bitrate control.")
        return False
    
    cmd = ['ffmpeg', '-i', input_video, '-y']  # -y to overwrite
    
    # Try to use GPU acceleration for encoding if available (NVENC)
    # Check for NVIDIA GPU encoder
    use_nvenc = False
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                              capture_output=True, text=True, timeout=2)
        if 'h264_nvenc' in result.stdout:
            # Use NVENC for faster encoding on NVIDIA GPUs
            codec = 'h264_nvenc'
            use_nvenc = True
            print("Using GPU-accelerated encoding (NVENC)")
    except:
        pass  # Fallback to CPU encoding
    
    # Set codec
    cmd.extend(['-c:v', codec])
    
    # Set resolution if specified
    if width and height:
        cmd.extend(['-s', f'{width}x{height}'])
    
    # Set FPS if specified
    if fps:
        cmd.extend(['-r', str(fps)])
    
    # Set bitrate or CRF
    if target_bitrate:
        # Use bitrate mode
        bitrate_kbps = target_bitrate // 1000
        cmd.extend(['-b:v', f'{bitrate_kbps}k'])
        if not use_nvenc:
            cmd.extend(['-maxrate', f'{bitrate_kbps}k'])
            cmd.extend(['-bufsize', f'{bitrate_kbps * 2}k'])
        if use_nvenc:
            # NVENC uses different bitrate control
            cmd.extend(['-maxrate', f'{int(bitrate_kbps * 1.2)}k'])
            cmd.extend(['-bufsize', f'{bitrate_kbps * 2}k'])
    elif crf is not None:
        # Use CRF mode (constant quality)
        if not use_nvenc:
            cmd.extend(['-crf', str(crf)])
        else:
            # NVENC doesn't support CRF, use CQP instead
            cmd.extend(['-cq', str(crf)])
    else:
        # Default CRF for good quality
        if not use_nvenc:
            cmd.extend(['-crf', '23'])
        else:
            cmd.extend(['-cq', '23'])
    
    # Set preset
    if use_nvenc:
        # NVENC presets: p1 (fastest) to p7 (slowest, best quality)
        cmd.extend(['-preset', 'p4'])  # p4 = medium quality/speed
        cmd.extend(['-rc', 'vbr'])  # Variable bitrate
    else:
        cmd.extend(['-preset', preset])
    
    # Copy audio if present
    cmd.extend(['-c:a', 'copy'])
    
    # Pixel format for compatibility
    cmd.extend(['-pix_fmt', 'yuv420p'])
    
    # Output file
    cmd.append(output_video)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error encoding video with ffmpeg: {e.stderr}")
        return False


def get_video_properties(video_path: str) -> Tuple[Dict, bool]:
    """
    Get video properties, trying ffprobe first, then falling back to OpenCV.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (properties_dict, has_bitrate_info)
        has_bitrate_info is True if bitrate information was successfully retrieved
    """
    # Try ffprobe first for detailed info including bitrate
    props = get_video_properties_ffprobe(video_path)
    has_bitrate = False
    
    if props and props.get('bitrate', 0) > 0:
        has_bitrate = True
        return props, has_bitrate
    
    # Fall back to OpenCV for basic properties
    if props:
        # We got some info from ffprobe but no bitrate
        opencv_props = get_video_properties_opencv(video_path)
        props.update(opencv_props)
    else:
        # No ffprobe info, use OpenCV only
        props = get_video_properties_opencv(video_path)
    
    return props, has_bitrate

