#!/usr/bin/env python3
"""
Video Stabilization and Sharpening Pipeline

This script combines L1 optimal video stabilization with Real-ESRGAN sharpening.
The pipeline:
1. Stabilizes input video using L1 optimal paths with a given crop_ratio
2. Sharpens the stabilized video and upscales it back to original resolution
3. Outputs a stabilized and sharpened video

Usage:
    python pipeline.py -i input_video.mp4 -o output_video.mp4 --crop-ratio 0.8
"""

import argparse
import os
import shutil
import tempfile
import time
from datetime import timedelta
from stabilization import stabilize_video
from sharpen import sharpen_and_upscale_video


def main():
    parser = argparse.ArgumentParser(
        description='Stabilize and sharpen video using L1 optimal paths and Real-ESRGAN'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input video'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output video (stabilized and sharpened)'
    )
    parser.add_argument(
        '--crop-ratio',
        type=float,
        required=True,
        help='Crop ratio for stabilization (float < 1.0, e.g., 0.8). '
             'The video will be cropped during stabilization and then upscaled back to original resolution.'
    )
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=30,
        help='Print progress every N frames during sharpening (default: 30)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary stabilized video file (for debugging)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for sharpening: cuda (GPU), cpu, or auto (default: auto)'
    )
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Disable half precision (FP16) even on GPU (use FP32 instead)'
    )
    parser.add_argument(
        '--traj-plot',
        action='store_true',
        help='Plot camera trajectories before/after stabilization (saves to plots/ folder)'
    )
    parser.add_argument(
        '--face-enhance',
        action='store_true',
        help='Enable face enhancement using GFPGAN (requires: pip install facexlib gfpgan)'
    )
    parser.add_argument(
        '--tile',
        type=int,
        default=0,
        help='Tile size for processing large images. 0 = no tiling. '
             'Set to 400-800 if you run out of memory (default: 0)'
    )
    parser.add_argument(
        '--no-preserve-bitrate',
        action='store_true',
        help='Do not preserve original video bitrate (may result in larger files)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Number of frames to load in batches for I/O optimization (default: 4). '
             'The pipeline will auto-adjust tile size to better utilize available GPU memory.'
    )
    parser.add_argument(
        '--gpu-stabilization',
        action='store_true',
        help='Use GPU acceleration for stabilization (requires OpenCV with CUDA support)'
    )
    parser.add_argument(
        '--async-sharpening',
        action='store_true',
        help='Use async processing with CUDA streams for sharpening (faster GPU utilization)'
    )
    parser.add_argument(
        '--output-both',
        action='store_true',
        help='Output both stabilized-only and stabilized+sharpened versions. '
             'The stabilized-only version will be saved with "_stabilized_only" suffix.'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Validate crop ratio
    if args.crop_ratio <= 0 or args.crop_ratio >= 1.0:
        print(f"Error: crop-ratio must be between 0 and 1.0, got {args.crop_ratio}")
        return
    
    # Create temporary file for stabilized video
    temp_dir = os.path.dirname(args.output) or '.'
    temp_stabilized = os.path.join(
        temp_dir,
        f"temp_stabilized_{os.path.basename(args.input)}"
    )
    
    # Start overall timer
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Stabilize video
        print("=" * 60)
        print("STEP 1: Video Stabilization")
        print("=" * 60)
        if args.gpu_stabilization:
            print("GPU acceleration for stabilization: ENABLED")
        else:
            print("GPU acceleration for stabilization: DISABLED (use --gpu-stabilization to enable)")
        stabilization_start_time = time.time()
        
        original_width, original_height, fps, video_props = stabilize_video(
            input_path=args.input,
            output_path=temp_stabilized,
            crop_ratio=args.crop_ratio,
            plot_trajectories=args.traj_plot,
            use_gpu=args.gpu_stabilization
        )
        
        stabilization_time = time.time() - stabilization_start_time
        print(f"\n[Timing] Stabilization completed in: {timedelta(seconds=int(stabilization_time))} ({stabilization_time:.2f} seconds)")
        
        # Save stabilized-only version if --output-both is enabled
        stabilized_only_path = None
        if args.output_both:
            # Generate output path for stabilized-only version
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1]
            stabilized_only_path = f"{base_name}_stabilized_only{ext}"
            
            # Copy the stabilized video to the permanent location
            shutil.copy2(temp_stabilized, stabilized_only_path)
            print(f"\n[Output] Stabilized-only video saved to: {stabilized_only_path}")
        
        # Step 2: Sharpen and upscale video
        print("\n" + "=" * 60)
        print("STEP 2: Video Sharpening and Upscaling")
        print("=" * 60)
        if args.async_sharpening:
            print("Async processing for sharpening: ENABLED")
        else:
            print("Async processing for sharpening: DISABLED (use --async-sharpening to enable)")
        print(f"Batch size: {args.batch_size}")
        sharpening_start_time = time.time()
        
        # Determine device settings
        device = None if args.device == 'auto' else args.device
        if args.no_half:
            use_half = False  # Force FP32 if --no-half is specified
        elif args.device == 'auto':
            use_half = None  # Auto-detect
        else:
            use_half = (args.device == 'cuda')  # Use FP16 on GPU, FP32 on CPU
        
        success = sharpen_and_upscale_video(
            input_path=temp_stabilized,
            output_path=args.output,
            original_width=original_width,
            original_height=original_height,
            crop_ratio=args.crop_ratio,
            fps=fps,
            progress_interval=args.progress_interval,
            device=device,
            use_half=use_half,
            tile=args.tile,
            face_enhance=args.face_enhance,
            video_props=video_props,
            preserve_bitrate=not args.no_preserve_bitrate,
            batch_size=args.batch_size,
            async_processing=args.async_sharpening
        )
        
        sharpening_time = time.time() - sharpening_start_time
        print(f"\n[Timing] Sharpening completed in: {timedelta(seconds=int(sharpening_time))} ({sharpening_time:.2f} seconds)")
        
        if success:
            total_time = time.time() - pipeline_start_time
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)
            if args.output_both:
                print(f"Output videos:")
                print(f"  - Stabilized + Sharpened: {args.output}")
                print(f"  - Stabilized only:         {stabilized_only_path}")
            else:
                print(f"Output video: {args.output}")
            print("\n" + "=" * 60)
            print("PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Total pipeline time: {timedelta(seconds=int(total_time))} ({total_time:.2f} seconds)")
            print(f"  - Stabilization: {timedelta(seconds=int(stabilization_time))} ({stabilization_time:.2f}s, {100*stabilization_time/total_time:.1f}%)")
            print(f"  - Sharpening:     {timedelta(seconds=int(sharpening_time))} ({sharpening_time:.2f}s, {100*sharpening_time/total_time:.1f}%)")
            print("=" * 60)
        else:
            print("\nPipeline failed during sharpening step.")
            return
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        # Clean up temporary file
        if not args.keep_temp and os.path.exists(temp_stabilized):
            try:
                os.remove(temp_stabilized)
                print(f"\nCleaned up temporary file: {temp_stabilized}")
            except OSError:
                print(f"\nWarning: Could not remove temporary file: {temp_stabilized}")


if __name__ == "__main__":
    main()

