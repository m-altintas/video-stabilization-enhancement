"""
Video Stabilization Module using L1 Optimal Paths

This module provides L1 optimal video stabilization functionality.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
from os.path import join
from L1optimal_lpp import stabilize
from video_utils import get_video_properties

# Check for OpenCV CUDA support
HAS_CUDA = False
try:
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CUDA = True
except:
    pass


def get_inter_frame_transforms(cap, F_transforms, prev_gray, use_gpu=False):
    """
    Find the inter-frame transformations array F_transforms.
    Updates F_transforms array inplace.
    
    Args:
        cap: VideoCapture object
        F_transforms: Array to store transforms (n_frames x 3 x 3)
        prev_gray: Previous frame in grayscale
        use_gpu: If True, use CUDA acceleration when available
    """
    n_frames = F_transforms.shape[0]
    
    # Initialize CUDA objects if available and requested
    cuda_prev_gray = None
    cuda_curr_gray = None
    lk_params = None
    
    if use_gpu and HAS_CUDA:
        try:
            # Upload initial frame to GPU
            cuda_prev_gray = cv.cuda_GpuMat()
            cuda_prev_gray.upload(prev_gray)
            
            # Create CUDA optical flow tracker
            # Note: OpenCV CUDA doesn't have direct SparsePyrLK equivalent,
            # but we can use CPU for feature detection and CUDA for other operations
            lk_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        except Exception as e:
            print(f"Warning: CUDA initialization failed, falling back to CPU: {e}")
            use_gpu = False
    
    for i in range(n_frames):
        # Detect feature points in previous frame (or 1st frame in 1st iteration)
        # Feature detection is typically fast on CPU, so we keep it there
        prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                          minDistance=30, blockSize=3)
        
        if prev_pts is None or len(prev_pts) == 0:
            # Read next frame even if no features found
            success, curr = cap.read()
            if not success:
                break
            curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
            prev_gray = curr_gray
            if use_gpu and HAS_CUDA and cuda_prev_gray is not None:
                cuda_prev_gray.upload(prev_gray)
            continue
        
        # Read next frame
        success, curr = cap.read()
        # If there is no next frame, stream/video has ended
        if not success:
            break
        
        # Convert to grayscale
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        
        # Calculate optical flow (i.e. track feature points)
        if use_gpu and HAS_CUDA and cuda_prev_gray is not None:
            try:
                # Upload current frame to GPU
                if cuda_curr_gray is None:
                    cuda_curr_gray = cv.cuda_GpuMat()
                cuda_curr_gray.upload(curr_gray)
                
                # Use CPU optical flow (OpenCV CUDA doesn't have SparsePyrLK)
                # But we can use GPU for other operations if needed
                curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            except Exception as e:
                # Fallback to CPU
                curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        else:
            # CPU optical flow
            curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        
        # Sanity check, we should at least get the status of all the features from the previous frame
        # Even if they are no longer being tracked (as indicated by the status array)
        assert prev_pts.shape == curr_pts.shape
        # Filter out and use only valid points
        idx = np.where(status == 1)[0]
        # Update which points we should continue to maintain state for
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Find transformation matrix for full 6 DOF affine transform
        if len(prev_pts) >= 3:  # Need at least 3 points for affine transform
            m, _ = cv.estimateAffine2D(curr_pts, prev_pts)
            # Add current transformation matrix $F_t$ to array
            # $F_t$ is a right multiplied homogeneous affine transform, last column is untouched
            # We start from index 1 since index 0 is the identity matrix by construction/definition
            F_transforms[i + 1, :, :2] = m.T
        
        # Move to next frame
        prev_gray = curr_gray
        if use_gpu and HAS_CUDA and cuda_prev_gray is not None:
            # Swap buffers for next iteration
            cuda_prev_gray, cuda_curr_gray = cuda_curr_gray, cuda_prev_gray
    return


def plot_trajectory(og, stb, out_path_plot):
    """
    Plot the original and the corrected/stabilized x and y components of the camera trajectory.
    
    Args:
        og: Original camera trajectory (n_frames x 2 array)
        stb: Stabilized camera trajectory (n_frames x 2 array)
        out_path_plot: Base path for output plots (without extension)
    """
    # Create plots directory if it doesn't exist
    plot_dir = os.path.dirname(out_path_plot)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # x-coordinate trajectory
    plt.figure()
    plt.plot(og[:, 0])
    plt.plot(stb[:, 0])
    plt.xlabel('Frame Number')
    plt.ylabel('2D Camera x coord. (pixels)')
    plt.title('Original vs Stab x')
    plt.legend(['Original', 'Stabilized'])
    plt.savefig(out_path_plot + "_traj_x.png")
    plt.close()
    
    # y-coordinate trajectory
    plt.figure()
    plt.plot(og[:, 1])
    plt.plot(stb[:, 1])
    plt.xlabel('Frame Number')
    plt.ylabel('2D Camera y coord. (pixels)')
    plt.title('Original vs Stab y')
    plt.legend(['Original', 'Stabilized'])
    plt.savefig(out_path_plot + "_traj_y.png")
    plt.close()
    return


def stabilize_video(input_path, output_path, crop_ratio, plot_trajectories=False, use_gpu=False):
    """
    Stabilize a video using L1 optimal paths.
    
    Args:
        input_path: Path to input video
        output_path: Path to save stabilized video (will be cropped to crop_ratio size)
        crop_ratio: Crop ratio for stabilization (float < 1.0, e.g., 0.8)
        plot_trajectories: If True, plot camera trajectories before/after stabilization
        use_gpu: If True, use GPU acceleration when available (default: False)
    
    Returns:
        Tuple of (original_width, original_height, fps, video_properties_dict)
        video_properties_dict contains codec, bitrate, etc. if available
    """
    # Read input video
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties (try ffprobe first for bitrate, then OpenCV)
    video_props, has_bitrate = get_video_properties(input_path)
    
    # Get basic properties
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_props.get('fps', cap.get(cv.CAP_PROP_FPS))
    if isinstance(fps, float):
        fps = int(round(fps))
    
    print(f"Number of frames: {n_frames}")
    print(f"Original resolution: {w}x{h}, FPS: {fps}")
    if has_bitrate and video_props.get('bitrate'):
        bitrate_mbps = video_props['bitrate'] / 1_000_000
        print(f"Original bitrate: {bitrate_mbps:.2f} Mbps ({video_props.get('codec', 'unknown')} codec)")
    print(f"Crop ratio: {crop_ratio}")
    
    # Pre-define transformation-store array
    # Uses 3 parameters since it is purely a coordinate transform
    # A collection of n_frames number of homography matrices
    F_transforms = np.zeros((n_frames, 3, 3), np.float32)
    # Initialise all transformations with Identity matrix
    F_transforms[:, :, :] = np.eye(3)
    
    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale for feature tracking using openCV
    prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    
    # Find the inter-frame transformations array F_transforms
    print("Computing inter-frame transformations...")
    if use_gpu and HAS_CUDA:
        print("Using GPU acceleration for stabilization (where available)")
    transform_start = time.time()
    get_inter_frame_transforms(cap, F_transforms, prev_gray, use_gpu=use_gpu)
    transform_time = time.time() - transform_start
    print(f"[Timing] Inter-frame transforms: {transform_time:.2f} seconds")
    
    # Get stabilization transforms B_t by processing motion transition transforms F_t
    print("Computing L1 optimal stabilization transforms...")
    l1_start = time.time()
    B_transforms = stabilize(F_transforms, prev.shape, True, None, crop_ratio)
    l1_time = time.time() - l1_start
    print(f"[Timing] L1 optimization: {l1_time:.2f} seconds")
    
    # Calculate camera trajectories for plotting if requested
    if plot_trajectories:
        # Accumulate by right multiplication into C_trajectory
        # $C_{t + 1} = C_t x F_t$
        C_trajectory = F_transforms.copy()
        for i in range(1, n_frames):
            # Right multiply transformations to accumulate all the changes into camera trajectory
            C_trajectory[i, :, :] = C_trajectory[i - 1, :, :] @ F_transforms[i, :, :]
        
        # Repeat right multiplication procedure to obtain stabilized camera trajectory P
        P_trajectory = C_trajectory.copy()
        # Apply transform to C_trajectory to get P_trajectory
        for i in range(n_frames):
            P_trajectory[i, :, :] = C_trajectory[i, :, :] @ B_transforms[i, :, :]
        
        # Starting coordinate (0, 0) in homogeneous system
        origin = np.array([0, 0, 1])
        # Evolution of coordinate of camera trajectory under original scheme
        evolution_og = origin @ C_trajectory
        # Evolution of origin under stabilized trajectory
        evolution_stab = origin @ P_trajectory
        
        # Get output path for plots
        in_name = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = os.path.dirname(output_path) or '.'
        out_path_camera_plot = join(out_dir, 'plots', in_name)
        plot_trajectory(evolution_og, evolution_stab, out_path_camera_plot)
        print(f"Trajectory plots saved to: {out_path_camera_plot}_traj_x.png and _traj_y.png")
    
    # Calculate the cropped dimensions (the size of the stabilized content)
    crop_w = round(w * crop_ratio)
    crop_h = round(h * crop_ratio)
    
    print(f"Stabilized output resolution: {crop_w}x{crop_h}")
    
    # Create output video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Reset stream to first frame
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    
    # Write stabilized frames
    print("Writing stabilized frames...")
    write_start = time.time()
    
    # Initialize CUDA objects for warping if available
    cuda_frame = None
    cuda_warped = None
    gpu_warping_enabled = False
    if use_gpu and HAS_CUDA:
        try:
            cuda_frame = cv.cuda_GpuMat()
            cuda_warped = cv.cuda_GpuMat()
            # Test if CUDA warping actually works
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            cuda_frame.upload(test_frame)
            cv.cuda.warpAffine(cuda_frame, test_matrix, (100, 100), cuda_warped)
            gpu_warping_enabled = True
            print("GPU warping enabled for stabilization")
        except Exception as e:
            print(f"Warning: GPU warping not available, using CPU: {e}")
            use_gpu = False
            gpu_warping_enabled = False
    
    for i in range(n_frames):
        success, frame = cap.read()
        if not success:
            break
        
        # Apply affine wrapping to the given frame
        # Warp to the cropped size (not full size) to preserve the stabilized content
        scale_x = 1 / crop_ratio
        scale_y = 1 / crop_ratio

        scaling_matrix = np.eye(3, dtype=float)
        scaling_matrix[0][0] = scale_x
        scaling_matrix[1][1] = scale_y

        shifting_to_center_matrix = np.eye(3, dtype=float)
        shifting_to_center_matrix[0][2] = -w / 2.0
        shifting_to_center_matrix[1][2] = -h / 2.0

        shifting_back_matrix = np.eye(3, dtype=float)
        shifting_back_matrix[0][2] = w / 2.0
        shifting_back_matrix[1][2] = h / 2.0

        B_matrix = np.eye(3, dtype=float)
        B_matrix[:2][:] = B_transforms[i, :, :2].T
        final_matrix = shifting_back_matrix @ scaling_matrix @ shifting_to_center_matrix @ np.linalg.inv(B_matrix)
        
        # Warp to cropped size
        if gpu_warping_enabled and cuda_frame is not None and cuda_warped is not None:
            try:
                # Upload frame to GPU
                cuda_frame.upload(frame)
                # Create warp matrix (2x3 for warpAffine) - must be float32
                warp_matrix = final_matrix[:2, :].astype(np.float32)
                # Perform GPU warping
                cv.cuda.warpAffine(cuda_frame, warp_matrix, (crop_w, crop_h), cuda_warped, 
                                  flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
                # Download result
                frame_stabilized = cuda_warped.download()
            except Exception as e:
                # Fallback to CPU with error message
                if i == 0:  # Only print once
                    print(f"Warning: GPU warping failed, falling back to CPU: {e}")
                frame_stabilized = cv.warpAffine(frame, final_matrix[:2, :], (crop_w, crop_h))
        else:
            frame_stabilized = cv.warpAffine(frame, final_matrix[:2, :], (crop_w, crop_h))
        
        out.write(frame_stabilized)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{n_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    write_time = time.time() - write_start
    total_stab_time = transform_time + l1_time + write_time
    fps_processing = n_frames / total_stab_time if total_stab_time > 0 else 0
    
    print(f"\n[Timing] Frame writing: {write_time:.2f} seconds")
    print(f"[Timing] Processing rate: {fps_processing:.2f} frames/second")
    print(f"Stabilization complete. Output saved to: {output_path}")
    
    return w, h, fps, video_props

