# How to Verify Optimizations Are Working

This guide helps you verify that the performance optimizations are actually being used.

## 1. Check GPU Stabilization

**What to look for:**
- When running with `--gpu-stabilization`, you should see: "GPU warping enabled for stabilization"
- If you see "Warning: GPU warping not available", OpenCV CUDA is not available

**Test:**
```bash
python pipeline.py -i test.mp4 -o output.mp4 --crop-ratio 0.8 --gpu-stabilization
```

**Expected output:**
```
GPU acceleration for stabilization: ENABLED
GPU warping enabled for stabilization
```

## 2. Check torch.compile()

**What to look for:**
- When PyTorch 2.0+ is installed, you should see: "Compiling Real-ESRGAN model with torch.compile()..."
- Followed by: "✓ Model compilation complete - torch.compile() optimization active"

**Test:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Has compile: {hasattr(torch, \"compile\")}')"
```

**Expected output:**
```
PyTorch version: 2.x.x
Has compile: True
```

## 3. Check Async Processing

**What to look for:**
- When running with `--async-sharpening`, you should see: "Using async processing mode for better GPU utilization"
- This appears at the start of the sharpening step

**Test:**
```bash
python pipeline.py -i test.mp4 -o output.mp4 --crop-ratio 0.8 --async-sharpening --device cuda
```

**Expected output:**
```
Async processing for sharpening: ENABLED
Using async processing mode for better GPU utilization
```

## 4. Check NVENC Encoding

**What to look for:**
- When NVENC is available, you should see: "Using GPU-accelerated encoding (NVENC)"
- This appears during the final encoding step

**Test:**
```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

**Expected output:**
```
V..... h264_nvenc           NVIDIA NVENC H.264 encoder
```

## 5. Performance Comparison

Run these commands and compare the timing output:

**Baseline (no optimizations):**
```bash
python pipeline.py -i test.mp4 -o output1.mp4 --crop-ratio 0.8 --device cpu
```

**With GPU:**
```bash
python pipeline.py -i test.mp4 -o output2.mp4 --crop-ratio 0.8 --device cuda
```

**With all optimizations:**
```bash
python pipeline.py -i test.mp4 -o output3.mp4 --crop-ratio 0.8 \
    --device cuda --gpu-stabilization --async-sharpening --batch-size 8
```

Compare the "Total pipeline time" at the end of each run.

## Common Issues

### GPU Stabilization Not Working

**Symptom:** No "GPU warping enabled" message

**Causes:**
1. OpenCV not compiled with CUDA support
2. OpenCV CUDA module not available

**Check:**
```python
import cv2 as cv
try:
    print(f"CUDA devices: {cv.cuda.getCudaEnabledDeviceCount()}")
except:
    print("OpenCV CUDA not available")
```

**Solution:** Install OpenCV with CUDA support or use CPU mode

### torch.compile() Not Working

**Symptom:** No compilation messages

**Causes:**
1. PyTorch < 2.0
2. Model structure doesn't match expected format

**Check:**
```python
import torch
print(f"Version: {torch.__version__}")
print(f"Has compile: {hasattr(torch, 'compile')}")
```

**Solution:** Upgrade to PyTorch 2.0+

### Async Processing Not Providing Speedup

**Symptom:** Same speed with/without `--async-sharpening`

**Causes:**
1. I/O is not the bottleneck (GPU processing is already fast enough)
2. Python GIL limiting parallelism
3. Batch size too small

**Solution:** 
- Increase `--batch-size` to 8 or 16
- Async processing helps most when I/O is slow (e.g., reading from slow disk)

### NVENC Not Working

**Symptom:** No "Using GPU-accelerated encoding" message

**Causes:**
1. ffmpeg not compiled with NVENC support
2. No NVIDIA GPU

**Check:**
```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

**Solution:** Install ffmpeg with NVENC support or use CPU encoding

## Debugging Tips

1. **Check what's actually being used:**
   - Look for the diagnostic messages in the output
   - Each optimization prints a message when enabled

2. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - GPU utilization should be high during processing
   - Memory usage should increase when optimizations are active

3. **Check timing breakdown:**
   - The pipeline prints timing for each step
   - Compare stabilization time and sharpening time separately

4. **Test with small video first:**
   - Use a short test video to quickly verify optimizations
   - Then scale up to full videos

## Expected Speedups

If optimizations are working correctly, you should see:

- **GPU vs CPU**: 5-10x faster
- **GPU + torch.compile()**: Additional 15-30% faster
- **GPU + async**: Additional 10-20% faster (when I/O is bottleneck)
- **GPU stabilization**: 20-40% faster stabilization step
- **NVENC encoding**: 5-10x faster encoding

If you're not seeing these improvements, check the diagnostic messages to see which optimizations are actually active.

