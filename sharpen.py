"""
Video Sharpening Module using Real-ESRGAN

This module provides video sharpening and upscaling functionality using Real-ESRGAN.
Supports GPU acceleration with PyTorch for faster processing.
"""

import cv2 as cv
import numpy as np
from typing import Optional, Dict, List
import torch
import os
import tempfile
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
from video_utils import encode_video_with_ffmpeg, get_video_properties

# Try to disable CUDA memory pool recording to avoid conflicts with Real-ESRGAN
# This is a workaround for "beginAllocateToPool: already recording to mempool_id" errors
if torch.cuda.is_available():
    try:
        # Disable memory pool recording if possible
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    except:
        pass

# Fix for torchvision compatibility issue with basicsr
try:
    import torchvision.transforms.functional as F
    import sys
    import types
    import torchvision.transforms as T
    
    # Create the functional_tensor module and inject it
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        functional_tensor = types.ModuleType('functional_tensor')
        functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
        if hasattr(F, 'adjust_brightness'):
            functional_tensor.adjust_brightness = F.adjust_brightness
        if hasattr(F, 'adjust_contrast'):
            functional_tensor.adjust_contrast = F.adjust_contrast
        
        # Inject into both the module and sys.modules
        T.functional_tensor = functional_tensor
        sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
except Exception:
    pass  # If patching fails, we'll handle the error in the function


def get_device_info():
    """
    Detect available device and return device info.
    
    Returns:
        Tuple of (device_name, use_half, device)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = 'cuda'
        # Use half precision (FP16) for faster processing on modern GPUs
        # Check if GPU supports FP16 (most modern GPUs do)
        use_half = True
        print(f"GPU detected: {device_name}")
        print(f"Using GPU acceleration with FP16 precision")
    else:
        device_name = "CPU"
        device = 'cpu'
        use_half = False
        print("No GPU detected, using CPU (slower)")
    return device_name, use_half, device


def enhance_frame_realesrgan(frame: np.ndarray, outscale: float = 1.0, 
                             device: str = None, use_half: bool = None,
                             tile: int = 0, face_enhance: bool = False, 
                             force_reload: bool = False) -> Optional[np.ndarray]:
    """
    Enhance and/or upscale frame using Real-ESRGAN model.
    
    Args:
        frame: Input frame (BGR format)
        outscale: Upscaling factor (1.0 = no upscaling, just enhancement; 
                  2.0 = 2x upscaling, 4.0 = 4x upscaling, etc.)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        use_half: Whether to use half precision (FP16) for faster processing.
                 If None, auto-detects based on device.
        tile: Tile size for processing large images. 0 = no tiling. 
              Set to 400-800 if you run out of memory.
        face_enhance: Whether to use GFPGAN for face enhancement (requires facexlib and gfpgan)
    
    Requires: pip install torch realesrgan basicsr
    For face enhancement: pip install facexlib gfpgan
    Model weights download automatically on first use.
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # Auto-detect device if not specified
        if device is None or use_half is None:
            _, auto_half, auto_device = get_device_info()
            if device is None:
                device = auto_device
            if use_half is None:
                use_half = auto_half and device == 'cuda'
        
        # Check if upsampler is already initialized
        # Use the actual tile value in cache key
        cache_key = f"{device}_{use_half}_{tile}_{face_enhance}"
        if not hasattr(enhance_frame_realesrgan, 'upsamplers'):
            enhance_frame_realesrgan.upsamplers = {}
        
        # Force reload if requested (useful when tile size changes)
        if force_reload and cache_key in enhance_frame_realesrgan.upsamplers:
            del enhance_frame_realesrgan.upsamplers[cache_key]
        
        if cache_key not in enhance_frame_realesrgan.upsamplers:
            print(f"Loading Real-ESRGAN model on {device} (first time may download weights)...")
            
            # Use RealESRGAN_x4plus model (good for general enhancement and upscaling)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            
            # Real-ESRGAN auto-downloads weights from the official release
            # Set up face enhancer if requested
            face_enhancer = None
            if face_enhance:
                try:
                    from gfpgan import GFPGANer
                    face_enhancer = GFPGANer(
                        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                        upscale=1,  # We'll upscale with Real-ESRGAN first
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None,
                        device=torch.device(device)
                    )
                    print("Face enhancement enabled (GFPGAN)")
                except ImportError:
                    print("Warning: GFPGAN not installed. Face enhancement disabled.")
                    print("Install with: pip install facexlib gfpgan")
                    face_enhance = False  # Disable face enhancement if not available
            
            upsampler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=tile,  # Configurable tile size (larger = more GPU utilization)
                tile_pad=10,
                pre_pad=0,
                half=use_half,  # Use FP16 for faster GPU processing
                device=torch.device(device),
                gpu_id=0 if device == 'cuda' else None
            )
            
            # Enable CUDA optimizations if available
            if device == 'cuda' and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Store face enhancer with upsampler
            if face_enhancer is not None:
                upsampler.face_enhancer = face_enhancer
            
            # Try to use torch.compile() with workarounds for CUDA memory pool conflicts
            if device == 'cuda' and torch.cuda.is_available():
                try:
                    if hasattr(torch, 'compile'):
                        torch_version = torch.__version__.split('.')
                        if len(torch_version) >= 2 and int(torch_version[0]) >= 2:
                            # Try to compile the underlying model with a workaround
                            # Use 'default' mode instead of 'reduce-overhead' to avoid memory pool issues
                            compiled = False
                            
                            # Try to access and compile the model
                            if hasattr(upsampler, 'model') and upsampler.model is not None:
                                print("Attempting to compile Real-ESRGAN model with torch.compile() (default mode)...")
                                # Use 'default' mode which is more compatible with memory pools
                                upsampler.model = torch.compile(upsampler.model, mode='default', fullgraph=False)
                                compiled = True
                            elif hasattr(upsampler, 'nets') and upsampler.nets:
                                for i, net in enumerate(upsampler.nets):
                                    if hasattr(net, 'net') and net.net is not None:
                                        print(f"Attempting to compile Real-ESRGAN net {i} with torch.compile()...")
                                        net.net = torch.compile(net.net, mode='default', fullgraph=False)
                                        compiled = True
                                    elif hasattr(net, 'module'):
                                        print(f"Attempting to compile Real-ESRGAN net {i} (module)...")
                                        net.module = torch.compile(net.module, mode='default', fullgraph=False)
                                        compiled = True
                            
                            if compiled:
                                print("✓ Model compilation attempted - will test during first frame")
                                # Note: We'll see if it works during actual inference
                except Exception as e:
                    print(f"Note: torch.compile() setup failed: {e}")
                    pass
            
            enhance_frame_realesrgan.upsamplers[cache_key] = upsampler
        
        upsampler = enhance_frame_realesrgan.upsamplers[cache_key]
        
        # Convert BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Enhance and/or upscale using outscale parameter
        # Use torch.no_grad() to prevent memory pool issues and disable gradient tracking
        with torch.no_grad():
            try:
                output, _ = upsampler.enhance(frame_rgb, outscale=outscale)
            except RuntimeError as e:
                error_str = str(e)
                if "beginAllocateToPool" in error_str or "mempool" in error_str.lower():
                    # CUDA memory pool conflict - clear cache, synchronize, and retry once
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        time.sleep(0.001)  # Small delay to let memory pool reset
                    try:
                        output, _ = upsampler.enhance(frame_rgb, outscale=outscale)
                    except Exception as e2:
                        if "beginAllocateToPool" in str(e2) or "mempool" in str(e2).lower():
                            print(f"Warning: CUDA memory pool conflict. If using async processing, ensure proper locking.")
                        print(f"Error using Real-ESRGAN (retry failed): {e2}")
                        return None
                else:
                    print(f"Error using Real-ESRGAN: {e}")
                    return None
            except Exception as e:
                print(f"Error using Real-ESRGAN: {e}")
                return None
        
        # Apply face enhancement if enabled and available
        if hasattr(upsampler, 'face_enhancer') and upsampler.face_enhancer is not None:
            try:
                # Face enhancement expects RGB input
                _, _, output = upsampler.face_enhancer.enhance(
                    output, has_aligned=False, only_center_face=False, paste_back=True
                )
            except Exception as e:
                print(f"Warning: Face enhancement failed: {e}")
        
        # Convert back to BGR
        output_bgr = cv.cvtColor(output, cv.COLOR_RGB2BGR)
        
        return output_bgr
        
    except ImportError as e:
        print(f"Real-ESRGAN dependencies not installed: {e}")
        print("Install with: pip install realesrgan basicsr")
        return None
    except Exception as e:
        print(f"Error using Real-ESRGAN: {e}")
        return None


def enhance_frames_batch(frames: List[np.ndarray], outscale: float, device: str, 
                         use_half: bool, tile: int, face_enhance: bool) -> List[Optional[np.ndarray]]:
    """
    Enhance a batch of frames sequentially (Real-ESRGAN processes one at a time).
    This function helps with frame organization and allows for future parallelization.
    
    Args:
        frames: List of frames (BGR format)
        outscale: Upscaling factor
        device: Device to use
        use_half: Whether to use FP16
        tile: Tile size
        face_enhance: Whether to use face enhancement
    
    Returns:
        List of enhanced frames
    """
    enhanced_frames = []
    for frame in frames:
        enhanced = enhance_frame_realesrgan(frame, outscale=outscale, 
                                           device=device, use_half=use_half,
                                           tile=tile, face_enhance=face_enhance)
        enhanced_frames.append(enhanced)
    return enhanced_frames


def load_frames_batch(cap, batch_size: int) -> List[Optional[np.ndarray]]:
    """
    Load a batch of frames from video.
    
    Args:
        cap: OpenCV VideoCapture object
        batch_size: Number of frames to load
    
    Returns:
        List of frames (None if end of video)
    """
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(None)
            break
    return frames


def sharpen_and_upscale_video(input_path, output_path, original_width, original_height, 
                              crop_ratio, fps, progress_interval=30, device=None, use_half=None,
                              tile=0, face_enhance=False, video_props=None, preserve_bitrate=True,
                              batch_size=4, async_processing=False):
    """
    Sharpen video and upscale it back to original resolution.
    
    Args:
        input_path: Path to stabilized (cropped) video
        output_path: Path to save sharpened and upscaled video
        original_width: Original video width
        original_height: Original video height
        crop_ratio: Crop ratio used during stabilization (to reverse it)
        fps: Video frame rate
        progress_interval: Print progress every N frames
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        use_half: Whether to use half precision (FP16) for faster processing.
                 If None, auto-detects based on device.
        tile: Tile size for processing large images. 0 = no tiling.
        face_enhance: Whether to use GFPGAN for face enhancement
        video_props: Dictionary with original video properties (bitrate, codec, etc.)
        preserve_bitrate: If True, try to preserve original bitrate using ffmpeg
        batch_size: Number of frames to process in parallel (default: 4, increase for more GPU utilization)
    
    Returns:
        True if successful, False otherwise
    """
    # Calculate upscale factor to restore original resolution
    upscale_factor = 1.0 / crop_ratio
    
    print(f"Sharpening and upscaling video...")
    print(f"Input resolution: {int(original_width * crop_ratio)}x{int(original_height * crop_ratio)}")
    print(f"Output resolution: {original_width}x{original_height}")
    print(f"Upscale factor: {upscale_factor:.3f}")
    
    # Open input video
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    input_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {n_frames} frames...")
    
    # Get initial GPU memory state
    initial_memory_used = 0
    gpu_memory_gb = 0
    if device == 'cuda' or (device is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    used_mb, total_mb = map(int, result.stdout.strip().split(', '))
                    initial_memory_used = used_mb / 1024.0
                    memory_usage_percent = (used_mb / total_mb) * 100
                    print(f"Initial GPU memory: {initial_memory_used:.2f}GB / {total_mb/1024:.2f}GB ({memory_usage_percent:.1f}%)")
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                initial_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                memory_usage_percent = (initial_memory_used / gpu_memory_gb) * 100 if gpu_memory_gb > 0 else 0
    
    # Determine optimal tile setting BEFORE loading model
    optimized_tile = tile
    if device == 'cuda' or (device is None and torch.cuda.is_available()):
        if torch.cuda.is_available() and gpu_memory_gb > 0:
            # IMPORTANT: tile=0 = process entire image = MAXIMUM memory usage
            # tile > 0 = process in tiles = LESS memory per operation
            # 
            # If memory usage is low (<65%), use tile=0 to maximize GPU utilization
            if tile == 0:
                if memory_usage_percent < 65:
                    # We have headroom - use tile=0 for maximum GPU utilization
                    optimized_tile = 0
                    print(f"Memory usage low ({memory_usage_percent:.1f}%), using tile=0 (full image) for maximum GPU utilization")
                else:
                    # Memory usage is moderate-high, still try tile=0 but monitor
                    optimized_tile = 0
                    print(f"Memory usage: {memory_usage_percent:.1f}%, using tile=0 (full image processing)")
            else:
                optimized_tile = tile
                if memory_usage_percent < 65:
                    print(f"Note: Using tile={tile} (user specified). For maximum GPU utilization with {memory_usage_percent:.1f}% usage, consider tile=0")
    
    # Pre-load model with optimized tile to ensure correct configuration
    # This also allows us to check actual memory usage after model load
    # Force reload if tile setting changed to ensure we use the correct configuration
    if optimized_tile != tile:
        print(f"Tile setting changed from {tile} to {optimized_tile}, forcing model reload...")
        # Clear any cached upsampler with old tile setting
        cache_key_old = f"{device if device else 'auto'}_{use_half if use_half is not None else 'auto'}_{tile}_{face_enhance}"
        if hasattr(enhance_frame_realesrgan, 'upsamplers') and cache_key_old in enhance_frame_realesrgan.upsamplers:
            del enhance_frame_realesrgan.upsamplers[cache_key_old]
            print(f"Cleared cached model with tile={tile}")
    
    print(f"Pre-loading Real-ESRGAN model with tile={optimized_tile}...")
    ret, test_frame = cap.read()
    if ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        # Load model by processing a test frame with optimized tile
        # This ensures the model is configured correctly before batch processing
        test_enhanced = enhance_frame_realesrgan(
            test_frame, 
            outscale=upscale_factor,
            device=device,
            use_half=use_half,
            tile=optimized_tile,  # Use optimized tile - this will create/use correct upsampler
            face_enhance=face_enhance,
            force_reload=(optimized_tile != tile)  # Force reload if tile changed
        )
        
        # Check memory AFTER model is loaded
        if device == 'cuda' or (device is None and torch.cuda.is_available()):
            if torch.cuda.is_available():
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        used_mb, total_mb = map(int, result.stdout.strip().split(', '))
                        after_load_memory = used_mb / 1024.0
                        after_load_percent = (used_mb / total_mb) * 100
                        memory_increase = after_load_memory - initial_memory_used
                        print(f"GPU memory after model load: {after_load_memory:.2f}GB / {total_mb/1024:.2f}GB ({after_load_percent:.1f}%)")
                        print(f"Memory increase: {memory_increase:.2f}GB")
                        
                        # If memory usage is still low, we could potentially use even more
                        if after_load_percent < 70 and optimized_tile == 0:
                            print(f"Memory usage still low ({after_load_percent:.1f}%), tile=0 is maximizing GPU utilization")
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass
    
    # Adjust batch size for frame loading
    if device == 'cuda' or (device is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            if gpu_memory_gb > 10:  # High-end GPU
                batch_size = max(batch_size, 8)
            elif gpu_memory_gb > 6:  # Mid-range GPU
                batch_size = max(batch_size, 6)
            
            if batch_size > 1:
                print(f"Using batch size: {batch_size} for frame loading optimization")
    else:
        batch_size = 1  # CPU processing, no batching
    
    # Determine if we should use ffmpeg for encoding (if bitrate preservation is requested)
    use_ffmpeg = preserve_bitrate and video_props and video_props.get('bitrate', 0) > 0
    
    if use_ffmpeg:
        # Create temporary file for OpenCV output
        temp_output = output_path.replace('.mp4', '_temp.mp4').replace('.avi', '_temp.avi')
        final_output = output_path
        target_bitrate = video_props.get('bitrate')
        print(f"Will preserve bitrate: {target_bitrate / 1_000_000:.2f} Mbps using ffmpeg")
    else:
        temp_output = output_path
        final_output = None
        target_bitrate = None
    
    # Create output video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(temp_output, fourcc, fps, (original_width, original_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video {temp_output}")
        cap.release()
        return False
    
    # Process frames in batches for better GPU utilization
    frame_count = 0
    success_count = 0
    processing_start_time = time.time()
    first_frame_time = None
    
    # Async processing with lock-based serialization to work around CUDA memory pool conflicts
    # We use a lock to ensure only one thread accesses Real-ESRGAN at a time,
    # but still allow I/O to overlap with GPU processing
    processing_lock = None
    if async_processing and (device == 'cuda' or (device is None and torch.cuda.is_available())):
        import threading
        processing_lock = threading.Lock()
        print("Using async processing with lock-based serialization for Real-ESRGAN access")
    
    if batch_size > 1 and processing_lock is not None:
        # Async processing with lock-based serialization
        import threading
        frame_queue = Queue(maxsize=batch_size * 3)  # Buffer for async loading
        result_queue = Queue()
        stop_loading = threading.Event()
        loader_exception = [None]
        
        def frame_loader():
            """Thread function to load frames asynchronously"""
            try:
                while not stop_loading.is_set():
                    frames_batch = load_frames_batch(cap, batch_size)
                    if not frames_batch or all(f is None for f in frames_batch):
                        frame_queue.put(None)  # Signal end
                        break
                    frame_queue.put(frames_batch)
            except Exception as e:
                loader_exception[0] = e
                frame_queue.put(None)
        
        def frame_processor():
            """Thread function to process frames on GPU with lock"""
            try:
                while True:
                    frames_batch = frame_queue.get()
                    if frames_batch is None:
                        result_queue.put(None)  # Signal end
                        break
                    
                    valid_frames = [f for f in frames_batch if f is not None]
                    if not valid_frames:
                        result_queue.put([])
                        continue
                    
                    # Use lock to serialize Real-ESRGAN access (prevents memory pool conflicts)
                    with processing_lock:
                        enhanced_batch = enhance_frames_batch(
                            valid_frames, 
                            outscale=upscale_factor,
                            device=device,
                            use_half=use_half,
                            tile=optimized_tile,
                            face_enhance=face_enhance
                        )
                    result_queue.put((valid_frames, enhanced_batch))
            except Exception as e:
                result_queue.put(None)
                if loader_exception[0] is None:
                    loader_exception[0] = e
        
        # Start threads
        loader_thread = Thread(target=frame_loader, daemon=True)
        processor_thread = Thread(target=frame_processor, daemon=True)
        loader_thread.start()
        processor_thread.start()
        
        # Process results
        try:
            while True:
                result = result_queue.get()
                if result is None:
                    break
                
                if not result:
                    continue
                
                valid_frames, enhanced_batch = result
                
                # Write enhanced frames
                for i, enhanced in enumerate(enhanced_batch):
                    if enhanced is not None:
                        if enhanced.shape[1] != original_width or enhanced.shape[0] != original_height:
                            enhanced = cv.resize(enhanced, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                        out.write(enhanced)
                        success_count += 1
                    else:
                        original_frame = valid_frames[i]
                        frame_resized = cv.resize(original_frame, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                        print(f"Warning: Enhancement failed for frame {frame_count + i}, using resized original")
                        out.write(frame_resized)
                
                frame_count += len(valid_frames)
                
                if first_frame_time is None and frame_count > 0:
                    first_frame_time = time.time() - processing_start_time
                    print(f"[Timing] First frame processed in: {first_frame_time:.2f} seconds (includes model warmup)")
                
                if frame_count % progress_interval == 0:
                    elapsed = time.time() - processing_start_time
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (n_frames - frame_count) / fps_current if fps_current > 0 else 0
                    eta_str = timedelta(seconds=int(eta_seconds)) if eta_seconds > 0 else "calculating..."
                    print(f"Processed {frame_count}/{n_frames} frames ({100*frame_count/n_frames:.1f}%) | "
                          f"Speed: {fps_current:.2f} fps | ETA: {eta_str}")
        finally:
            stop_loading.set()
            loader_thread.join(timeout=5.0)
            processor_thread.join(timeout=5.0)
            if loader_exception[0] is not None:
                print(f"Warning: Exception in async processing: {loader_exception[0]}")
    
    elif batch_size > 1:
        # Synchronous batch processing
        while True:
            # Load batch of frames
            frames_batch = load_frames_batch(cap, batch_size)
            if not frames_batch or all(f is None for f in frames_batch):
                break
            
            # Filter out None frames
            valid_frames = [f for f in frames_batch if f is not None]
            if not valid_frames:
                break
            
            # Process batch with optimized tile
            enhanced_batch = enhance_frames_batch(
                valid_frames, 
                outscale=upscale_factor,
                device=device,
                use_half=use_half,
                tile=optimized_tile,
                face_enhance=face_enhance
            )
            
            # Write enhanced frames
            for i, enhanced in enumerate(enhanced_batch):
                if enhanced is not None:
                    if enhanced.shape[1] != original_width or enhanced.shape[0] != original_height:
                        enhanced = cv.resize(enhanced, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                    out.write(enhanced)
                    success_count += 1
                else:
                    original_frame = valid_frames[i]
                    frame_resized = cv.resize(original_frame, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                    print(f"Warning: Enhancement failed for frame {frame_count + i}, using resized original")
                    out.write(frame_resized)
            
            frame_count += len(valid_frames)
            
            if first_frame_time is None and frame_count > 0:
                first_frame_time = time.time() - processing_start_time
                print(f"[Timing] First frame processed in: {first_frame_time:.2f} seconds (includes model warmup)")
            
            if frame_count % progress_interval == 0:
                elapsed = time.time() - processing_start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                eta_seconds = (n_frames - frame_count) / fps_current if fps_current > 0 else 0
                eta_str = timedelta(seconds=int(eta_seconds)) if eta_seconds > 0 else "calculating..."
                print(f"Processed {frame_count}/{n_frames} frames ({100*frame_count/n_frames:.1f}%) | "
                      f"Speed: {fps_current:.2f} fps | ETA: {eta_str}")
    else:
        # Single frame processing (CPU or small GPU)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance and upscale frame using Real-ESRGAN
            enhanced = enhance_frame_realesrgan(frame, outscale=upscale_factor, 
                                               device=device, use_half=use_half,
                                               tile=optimized_tile, face_enhance=face_enhance)
            
            if enhanced is not None:
                # Verify dimensions match (in case of rounding differences)
                if enhanced.shape[1] != original_width or enhanced.shape[0] != original_height:
                    # Resize to match expected dimensions
                    enhanced = cv.resize(enhanced, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                out.write(enhanced)
                success_count += 1
            else:
                # If enhancement failed, resize original frame
                frame_resized = cv.resize(frame, (original_width, original_height), interpolation=cv.INTER_LINEAR)
                print(f"Warning: Enhancement failed for frame {frame_count}, using resized original")
                out.write(frame_resized)
            
            frame_count += 1
            
            # Track first frame time for warmup measurement
            if first_frame_time is None and frame_count > 0:
                first_frame_time = time.time() - processing_start_time
                print(f"[Timing] First frame processed in: {first_frame_time:.2f} seconds (includes model warmup)")
            
            # Progress update with timing
            if frame_count % progress_interval == 0:
                elapsed = time.time() - processing_start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                eta_seconds = (n_frames - frame_count) / fps_current if fps_current > 0 else 0
                eta_str = timedelta(seconds=int(eta_seconds)) if eta_seconds > 0 else "calculating..."
                print(f"Processed {frame_count}/{n_frames} frames ({100*frame_count/n_frames:.1f}%) | "
                      f"Speed: {fps_current:.2f} fps | ETA: {eta_str}")
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - processing_start_time
    processing_fps = frame_count / processing_time if processing_time > 0 else 0
    avg_time_per_frame = processing_time / frame_count if frame_count > 0 else 0
    
    print(f"\n[Timing] Frame processing: {timedelta(seconds=int(processing_time))} ({processing_time:.2f} seconds)")
    print(f"[Timing] Average processing rate: {processing_fps:.2f} frames/second")
    print(f"[Timing] Average time per frame: {avg_time_per_frame:.3f} seconds")
    if first_frame_time:
        print(f"[Timing] First frame (with warmup): {first_frame_time:.2f}s | Subsequent frames: ~{avg_time_per_frame:.3f}s each")
    
    # If using ffmpeg, re-encode with original bitrate
    ffmpeg_time = 0  # Initialize to 0 in case ffmpeg is not used
    if use_ffmpeg and final_output:
        print(f"\nRe-encoding with preserved bitrate...")
        ffmpeg_start = time.time()
        success = encode_video_with_ffmpeg(
            input_video=temp_output,
            output_video=final_output,
            target_bitrate=target_bitrate,
            codec='libx264',
            fps=fps,
            width=original_width,
            height=original_height,
            preset='medium'
        )
        
        if success:
            # Remove temporary file
            try:
                os.remove(temp_output)
                print(f"Temporary file cleaned up.")
            except OSError:
                print(f"Warning: Could not remove temporary file: {temp_output}")
        else:
            print(f"Warning: ffmpeg encoding failed. Using OpenCV output.")
            try:
                os.rename(temp_output, final_output)
            except OSError:
                print(f"Error: Could not rename temporary file.")
        
        ffmpeg_time = time.time() - ffmpeg_start
        print(f"[Timing] FFmpeg re-encoding: {timedelta(seconds=int(ffmpeg_time))} ({ffmpeg_time:.2f} seconds)")
    
    total_sharpening_time = processing_time + (ffmpeg_time if use_ffmpeg and final_output else 0)
    print(f"\n[Timing] Total sharpening step: {timedelta(seconds=int(total_sharpening_time))} ({total_sharpening_time:.2f} seconds)")
    print(f"Sharpening complete! Enhanced {success_count}/{frame_count} frames")
    print(f"Output saved to: {output_path}")
    
    return True

