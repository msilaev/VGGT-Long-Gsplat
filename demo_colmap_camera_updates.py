# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
from tomlkit import datetime
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap

from alternative_tracking.colmap_tracking import predict_tracks_with_colmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

from scipy.spatial.transform import Rotation as R


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    #parser.add_argument("--output_dir", type=str, required=True, help="Directory containing VGGT-Long output files (depth_maps.npy, extrinsic.npy, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=16.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.1, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=False, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )

    parser.add_argument(
        "--predict_tracks_type", type=str, default="vggsfm", help="Type of track prediction method: colmap or vggsfm"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Clear GPU memory before starting (important for long sequences)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"GPU memory free: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

    # # Run VGGT for camera and depth estimation
    #model = VGGT()
    #_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    #model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    #model.eval()
    #model = model.to(device)
    #print(f"Model loaded")
    ######################

    # # Get image paths and preprocess them
    #image_dir = os.path.join(args.scene_dir, "images")
    image_dir = args.scene_dir
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # # Load images and original coordinates
    # # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    #print(image_path_list)

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")
    
    # Monitor GPU memory after loading images
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU memory after loading images - Allocated: {allocated:.1f} GB, Cached: {cached:.1f} GB")

    image_dir = args.scene_dir
    path = image_dir.split("/")
    exp_dir = './exps'
    data_dir = os.path.join(
        exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1]
    )


    depth_path = os.path.join(data_dir, 'depth_maps.npy')
    depth_map = np.load(depth_path)
    print(f"Depth maps loaded from {depth_path}")
    depth_conf_path = os.path.join(data_dir, 'depth_confs.npy')
    depth_conf = np.load(depth_conf_path)
    print(f"Depth confidence maps loaded from {depth_conf_path}")

    intrinsics_path = os.path.join(data_dir, 'intrinsic.npy')
    intrinsic = np.load(intrinsics_path)
    print(f"Camera intrinsics loaded from {intrinsics_path}")

    extrinsic_path = os.path.join(data_dir, 'extrinsic.npy')
    extrinsic = np.load(extrinsic_path)
    print(f"Camera extrinsics loaded from {extrinsic_path}")


    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if not args.use_ba:
        raise NotImplementedError("Currently only BA mode is supported")
    image_size = np.array(images.shape[-2:])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = args.shared_camera

    with torch.cuda.amp.autocast(dtype=dtype):
        # Predicting Tracks - Multiple options for memory efficiency
        # Using alternative tracking methods for long sequences (165+ frames)
        
        # # Generate RGB colors for 3D points (needed for all tracking methods)
        # points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        # points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        # points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (S, H, W, 3)
        
        # if args.predict_tracks_type == 'colmap':
                    
        #     # Try COLMAP tracking first (most memory efficient for long sequences)
        #     print("Using COLMAP tracking (production quality, memory efficient)")                               
            
        #     pred_tracks, pred_vis_scores, pred_confs = predict_tracks_with_colmap(
        #         images,
        #         conf=depth_conf,                    
        #         max_query_pts=args.max_query_pts
        #     )
            
        if args.predict_tracks_type == 'vggsfm':
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )

        else:
                raise ValueError(f"Unknown predict_tracks_type: {args.predict_tracks_type}")

        torch.cuda.empty_cache()

    # rescale the intrinsic matrix from 518 to 1024
    intrinsic[:, :2, :] *= scale
    track_mask = pred_vis_scores > args.vis_thresh

    # TODO: radial distortion, iterative BA, masks
    # ADAPTIVE BA: Try progressively more tolerant settings
    reconstruction = None
    attempts = [
        (args.max_reproj_error, args.vis_thresh, "initial"),
        (args.max_reproj_error * 1.5, args.vis_thresh * 0.8, "relaxed"), 
        (args.max_reproj_error * 2.0, args.vis_thresh * 0.5, "very_relaxed"),
        (args.max_reproj_error * 3.0, 0.01, "desperate")
    ]
    
    for max_err, vis_th, attempt_name in attempts:
        print(f"BA attempt '{attempt_name}': max_error={max_err:.1f}, vis_thresh={vis_th:.3f}")
        
        adaptive_mask = pred_vis_scores > vis_th
        print(f"  Using {adaptive_mask.sum()} tracks (was {track_mask.sum()})")
        
        if adaptive_mask.sum() < 50:
            print(f"  Skipping: too few tracks")
            continue
            
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,  # Use W2C format for consistency
            intrinsic,
            pred_tracks,
            image_size,
            masks=adaptive_mask,
            max_reproj_error=max_err,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )
        
        if reconstruction is not None:
            print(f"  SUCCESS with '{attempt_name}' settings!")
            break
        else:
            print(f"  Failed with '{attempt_name}' settings")

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA even with relaxed settings")

    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction_resolution = img_load_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )
    
    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    refined_extrinsic=[]
    refined_intrinsic=[]

    for image in reconstruction.images.values():

        camera = reconstruction.cameras[image.camera_id]
        K = camera.calibration_matrix()
        refined_intrinsic.append(K)

        # Get refined extrinsic (W2C format)        
        quat = image.qvec
        R = R.from_quat(quat[1], quat[2], quat[3], quat[0]).as_matrix()
        t = image.tvec
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = t
        refined_extrinsic.append(W2C[:3, :])
    
    ####################### Save updated extrinsics and intrinsics
    
    # Save refined poses
    refined_extrinsic = np.array(refined_extrinsic)
    refined_intrinsic = np.array(refined_intrinsic)
    
    # Save refined extrinsics (W2C format)
    refined_extrinsic_path = os.path.join(data_dir, 'extrinsic.npy')
    np.save(refined_extrinsic_path, refined_extrinsic)

    # Save refined intrinsics
    refined_intrinsic_path = os.path.join(data_dir, 'intrinsic.npy')
    np.save(refined_intrinsic_path, refined_intrinsic)
       
    print(f"Refined extrinsics (W2C) saved to: {refined_extrinsic_path}")    
    print(f"Refined intrinsics saved to: {refined_intrinsic_path}")
    
        
    # Calculate average pose change
    pose_changes = [np.linalg.norm(np.array(orig[:3, 3]) - np.array(ref[:3, 3])) 
                for orig, ref in zip(extrinsic, refined_extrinsic)]

    original_poses = [np.linalg.norm(np.array(orig[:3, 3])) for orig in extrinsic]

    avg_pose_change = np.mean(pose_changes)/np.max(original_poses)
    max_pose_change = np.max(pose_changes)/np.max(original_poses)

    rot_changes = [np.linalg.norm(Rot_rel) for Rot_rel in [pycolmap.rotmat_to_rotvec(ref[:3,:3] @ orig[:3,:3].T) for orig, ref in zip(extrinsic, refined_extrinsic)]]

    avg_rot_change = np.mean( rot_changes  )
    max_rot_change = np.max( rot_changes  )
    print(f"Bundle adjustment pose refinement:")
    print(f"  Average pose change: {avg_pose_change:.6f}")
    print(f"  Maximum pose change: {max_pose_change:.6f}")
    print(f"  Average rotation change (degrees): {avg_rot_change*180/np.pi:.6f}")
    print(f"  Maximum rotation change (degrees): {max_rot_change*180/np.pi:.6f}")

    print(f"  Number of refined poses: {len(refined_extrinsic)}")

    # attempt reconstruction again 
    print("Validating refined cameras by attempting reconstruction again...")
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            refined_extrinsic,  # Use W2C format for consistency
            refined_intrinsic,
            pred_tracks,
            image_size,
            masks=adaptive_mask,
            max_reproj_error=args.max_err,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

    
    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
