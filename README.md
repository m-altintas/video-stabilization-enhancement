# Video Stabilization and Sharpening Pipeline

This pipeline combines L1 optimal video stabilization with Real-ESRGAN sharpening to produce high-quality stabilized and enhanced videos.

## Overview

The pipeline performs the following steps:

1. **Stabilization**: Stabilizes the input video using L1 optimal paths with a specified `crop_ratio`
   - The video is cropped during stabilization to avoid black borders
   - Output is a stabilized video at reduced resolution (original × crop_ratio)

2. **Sharpening and Upscaling**: Sharpens the stabilized video and restores it to original resolution
   - Uses Real-ESRGAN for high-quality sharpening
   - Upscales by `1/crop_ratio` to restore original dimensions
   - Output is a stabilized and sharpened video at original resolution

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### System Requirements

- Python 3.7+
- OpenCV (cv2)
- PuLP (for linear programming optimization)
- Real-ESRGAN (automatically downloads model weights on first use)
- **ffmpeg** (for bitrate preservation - install with: `sudo apt install ffmpeg` or `brew install ffmpeg`)
  - If ffmpeg is not available, bitrate preservation will be disabled automatically

## Usage

### Basic Usage

```bash
python pipeline.py -i input_video.mp4 -o output_video.mp4 --crop-ratio 0.8
```

### Arguments

- `-i, --input`: Path to input video (required)
- `-o, --output`: Path to output video (required)
- `--crop-ratio`: Crop ratio for stabilization (required, float < 1.0)
  - Typical values: 0.7-0.9
  - Lower values = more aggressive cropping (less black borders, smaller output during stabilization)
  - Higher values = less cropping (more black borders, larger output during stabilization)
- `--progress-interval`: Print progress every N frames during sharpening (default: 30)
- `--keep-temp`: Keep temporary stabilized video file (for debugging)
- `--device`: Device to use for sharpening: `cuda` (GPU), `cpu`, or `auto` (default: auto)
- `--no-half`: Disable half precision (FP16) even on GPU (use FP32 instead, slower but more accurate)
- `--traj-plot`: Plot camera trajectories before/after stabilization (saves to `plots/` folder)
- `--face-enhance`: Enable face enhancement using GFPGAN (requires: `pip install facexlib gfpgan`)
- `--tile`: Tile size for processing large images. 0 = no tiling. Set to 400-800 if you run out of memory (default: 0)
- `--no-preserve-bitrate`: Do not preserve original video bitrate (may result in larger files). By default, bitrate is preserved.
- `--batch-size`: Number of frames to load in batches for I/O optimization (default: 4). Increase for better GPU utilization.
- `--gpu-stabilization`: Use GPU acceleration for stabilization (requires OpenCV with CUDA support)
- `--async-sharpening`: Use async processing with threading for sharpening (overlaps I/O and GPU computation for faster processing)
- `--output-both`: Output both stabilized-only and stabilized+sharpened versions (stabilized-only version saved with "_stabilized_only" suffix)

### Prefered Usage

```bash
python pipeline.py -i input_video.mp4 -o output_video.mp4 --crop-ratio 0.85 --device cuda --no-preserve-bitrate --batch-size 64 --gpu-stabilization --async-sharpening
```

## How It Works

### Step 1: Stabilization

The stabilization process:
- Tracks feature points between frames using optical flow
- Computes inter-frame transformations
- Solves an L1 optimization problem to find smooth camera paths
- Applies transformations to stabilize the video
- Crops the video to `crop_ratio` size to avoid black borders

### Step 2: Sharpening and Upscaling

The sharpening process:
- Uses Real-ESRGAN to enhance each frame
- Upscales frames by `1/crop_ratio` to restore original resolution
- Outputs a high-quality stabilized and sharpened video

## File Structure

```
video-stabilization-enhancement/
├── L1optimal_lpp.py      # L1 optimization core (stabilization algorithm)
├── stabilization.py      # Video stabilization module
├── sharpen.py            # Video sharpening and upscaling module
├── pipeline.py           # Main pipeline script
├── video_utils.py        # Video utility functions
├── create_comparison_video.py  # Comparison video creation utility
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Notes

- The first run will download Real-ESRGAN model weights (~67 MB)
- Processing time depends on video length and resolution
- **GPU acceleration**: Automatically detects and uses GPU if available
  - GPU processing is typically 5-10x faster than CPU
  - Uses FP16 (half precision) by default on GPU for even faster processing
  - Use `--device cpu` to force CPU usage
  - Use `--no-half` to disable FP16 if you encounter precision issues
- **Bitrate preservation**: By default, the pipeline preserves the original video bitrate
  - Uses ffmpeg for final encoding to maintain original bitrate and codec settings
  - This prevents unnecessary file size increases
  - Sharpening doesn't inherently require higher bitrate - it's just better quality at the same bitrate
  - Use `--no-preserve-bitrate` to disable this feature
- The temporary stabilized video is automatically deleted unless `--keep-temp` is used

## Performance

### GPU Acceleration

The pipeline automatically detects and uses GPU if available. GPU acceleration provides significant speedup:

- **CPU**: ~0.5-2 seconds per frame (depending on resolution)
- **GPU (FP32)**: ~0.1-0.3 seconds per frame
- **GPU (FP16)**: ~0.05-0.15 seconds per frame (2x faster than FP32)
- **GPU (FP16) + Async**: ~0.04-0.12 seconds per frame (additional 10-20% speedup)

**Example**: For a 1000-frame video at 1080p:
- CPU: ~15-30 minutes
- GPU (FP16): ~1-3 minutes
- GPU (FP16) + Async + GPU Stabilization: ~45-90 seconds

### Speedup Tips

1. **Use GPU for both steps**: `--gpu-stabilization --device cuda`
2. **Enable async processing**: `--async-sharpening` (overlaps I/O and GPU computation)
3. **Increase batch size**: `--batch-size 8` or higher (if GPU memory allows)
4. **Use FP16**: Default on GPU, provides 2x speedup over FP32
5. **Use PyTorch 2.0+**: Automatically enables torch.compile() optimizations

### GPU Optimization Features

The pipeline automatically optimizes GPU utilization:

- **Auto tile size adjustment**: If GPU memory usage is low (<60%), automatically increases tile size to better utilize GPU
- **Batch frame loading**: Loads frames in batches to overlap I/O with GPU processing
- **CUDA optimizations**: Enables cuDNN benchmarking for faster processing
- **Memory-aware**: Automatically adjusts settings based on available GPU memory
- **GPU-accelerated stabilization**: Use `--gpu-stabilization` to accelerate feature tracking and warping (requires OpenCV with CUDA support)
- **Async processing**: Use `--async-sharpening` to overlap I/O and GPU computation for faster processing
- **torch.compile()**: Automatically uses PyTorch 2.0+ compilation optimizations when available
- **GPU-accelerated encoding**: Automatically uses NVENC for video encoding when available (requires ffmpeg with NVENC support)

**To maximize GPU utilization:**
- If you see low GPU memory usage, the pipeline will auto-increase tile size
- You can manually set `--tile 800` for maximum GPU utilization (if memory allows)
- Increase `--batch-size` to 8-16 for better frame loading overlap
- Enable `--gpu-stabilization` for faster stabilization (requires OpenCV with CUDA support)
- Enable `--async-sharpening` for better GPU utilization during sharpening

### Memory Usage

- **CPU**: Low memory usage (~2-4 GB RAM)
- **GPU**: Higher memory usage (~4-8 GB VRAM depending on resolution)
  - If you run out of GPU memory, the code will automatically use tiling
  - You can also force CPU with `--device cpu` or use `--tile` to reduce memory usage

## Troubleshooting

### Real-ESRGAN Import Error

If you get import errors for Real-ESRGAN:
```bash
pip install realesrgan basicsr
```

### Memory Issues

If you run out of memory during sharpening:
- Use the `--tile` parameter: `--tile 400` or `--tile 600`
- This processes the image in tiles instead of all at once
- Reduce `--batch-size` to 2 or 1 if needed
- Use `--device cpu` to process on CPU instead of GPU

### GPU Not Detected

If you have a GPU but it's not being used:

1. **Check PyTorch CUDA installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True` if CUDA is available.

2. **Reinstall PyTorch with CUDA support**:
   ```bash
   # Uninstall current PyTorch
   pip uninstall torch torchvision
   
   # Install with CUDA (replace cu118 with your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA version**:
   ```bash
   nvidia-smi
   ```
   Make sure your CUDA version matches PyTorch's CUDA version.

### GPU Out of Memory

If you get CUDA out of memory errors:

1. **Use CPU instead**:
   ```bash
   python pipeline.py -i input.mp4 -o output.mp4 --crop-ratio 0.8 --device cpu
   ```

2. **Use tiling**: Use the `--tile` parameter to process images in tiles:
   ```bash
   python pipeline.py -i input.mp4 -o output.mp4 --crop-ratio 0.8 --tile 400
   ```

3. **Use FP32 instead of FP16**:
   ```bash
   python pipeline.py -i input.mp4 -o output.mp4 --crop-ratio 0.8 --no-half
   ```

### Video Codec Issues

If the output video doesn't play, you may need to re-encode it:
```bash
ffmpeg -i output_video.mp4 -c:v libx264 -c:a copy final_output.mp4
```

## Features Comparison

### Compared to Original Repositories

Our merged pipeline includes all key features from both source repositories:

**From [L1-optimal-paths-Stabilization](https://github.com/ishank-juneja/L1-optimal-paths-Stabilization):**
- ✅ L1 optimal camera path stabilization
- ✅ Crop ratio parameter
- ✅ Camera trajectory plotting (`--traj-plot`)
- ✅ Full affine transform (6 DOF)

**From [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):**
- ✅ Real-ESRGAN_x4plus model
- ✅ Configurable upscaling factor
- ✅ GPU acceleration with FP16/FP32 support
- ✅ Tile-based processing for large images (`--tile`)
- ✅ Face enhancement with GFPGAN (`--face-enhance`)

**Additional Enhancements:**
- ✅ Integrated pipeline combining both methods
- ✅ Automatic resolution restoration
- ✅ Enhanced GPU detection and optimization
- ✅ Better error handling and progress reporting

## References

- L1 Optimal Paths: ["Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths"](https://github.com/ishank-juneja/L1-optimal-paths-Stabilization)
- Real-ESRGAN: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

