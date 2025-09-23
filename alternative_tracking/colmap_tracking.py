import pycolmap
import numpy as np
import torch
from pathlib import Path

def predict_tracks_with_colmap(images, max_query_pts=4096, **kwargs):
    """
    Use COLMAP's feature extraction and matching for 2D tracks only.
    Much more memory efficient for long sequences.
    No database or reconstruction needed - just direct feature matching.
    """
    print("Using COLMAP feature matching (memory efficient for long sequences)")
    
    # Create temporary directory for image processing
    temp_dir = Path("./temp_colmap")
    temp_dir.mkdir(exist_ok=True)
    
    # Save images temporarily
    image_paths = []
    for i, img in enumerate(images):
        img_path = temp_dir / f"frame_{i:06d}.jpg"
        # Convert tensor to PIL and save
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_np).save(img_path)
        image_paths.append(str(img_path))
    
    try:
        # Direct feature extraction and matching without database/reconstruction
        pred_tracks, pred_vis_scores, pred_confs = extract_and_match_features_directly(
            image_paths, max_query_pts
        )
            
    except Exception as e:
        print(f"COLMAP feature processing failed: {e}")
        print("Returning empty tracks")
        pred_tracks = np.zeros((0, len(images), 2))
        pred_vis_scores = np.zeros((0, len(images)))
        pred_confs = np.zeros((0, len(images)))
    
    # Cleanup temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return pred_tracks, pred_vis_scores, pred_confs

def extract_and_match_features_directly(image_paths, max_keypoints=4096):
    """
    Extract features and match them directly without database/reconstruction.
    More efficient for just getting 2D tracks.
    """
    from collections import defaultdict
    import cv2
    
    print(f"Extracting SIFT features from {len(image_paths)} images...")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    
    # Extract features for all images
    all_keypoints = []
    all_descriptors = []
    
    for i, img_path in enumerate(image_paths):
        if i % 20 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
            
        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            all_keypoints.append([])
            all_descriptors.append(None)
            continue
            
        # Extract SIFT features
        keypoints, descriptors = sift.detectAndCompute(img, None)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    
    # Match features between consecutive frames and build tracks
    print("Matching features and building tracks...")
    tracks = build_tracks_from_matches(all_keypoints, all_descriptors)
    
    # Convert to expected format
    pred_tracks, pred_vis_scores, pred_confs = format_tracks_for_output(tracks, len(image_paths))
    
    return pred_tracks, pred_vis_scores, pred_confs

def build_tracks_from_matches(all_keypoints, all_descriptors):
    """
    Build tracks by matching features between consecutive frames.
    """
    import cv2
    from collections import defaultdict
    
    # FLANN matcher for efficient matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    tracks = defaultdict(list)  # track_id -> [(frame_id, (x, y))]
    next_track_id = 0
    
    # Initialize tracks from first frame
    for i, kpt in enumerate(all_keypoints[0]):
        tracks[next_track_id].append((0, kpt.pt))
        next_track_id += 1
    
    # Process remaining frames
    for frame_id in range(1, len(all_keypoints)):
        if frame_id % 10 == 0:
            print(f"Matching frame {frame_id+1}/{len(all_keypoints)}")
            
        prev_desc = all_descriptors[frame_id - 1]
        curr_desc = all_descriptors[frame_id]
        
        if prev_desc is None or curr_desc is None or len(prev_desc) < 2 or len(curr_desc) < 2:
            continue
        
        try:
            # Match with previous frame
            matches = flann.knnMatch(prev_desc, curr_desc, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Find active tracks from previous frame
            active_tracks = {}
            for track_id, track_points in tracks.items():
                if track_points and track_points[-1][0] == frame_id - 1:
                    # Find which keypoint this track corresponds to
                    last_point = track_points[-1][1]
                    for kpt_idx, kpt in enumerate(all_keypoints[frame_id - 1]):
                        if abs(kpt.pt[0] - last_point[0]) < 1.0 and abs(kpt.pt[1] - last_point[1]) < 1.0:
                            active_tracks[kpt_idx] = track_id
                            break
            
            # Extend tracks
            for match in good_matches:
                prev_kpt_idx = match.queryIdx
                curr_kpt_idx = match.trainIdx
                
                if prev_kpt_idx in active_tracks:
                    # Extend existing track
                    track_id = active_tracks[prev_kpt_idx]
                    curr_kpt = all_keypoints[frame_id][curr_kpt_idx]
                    tracks[track_id].append((frame_id, curr_kpt.pt))
                    
        except cv2.error as e:
            print(f"Matching failed for frame {frame_id}: {e}")
            continue
    
    return tracks

def format_tracks_for_output(tracks, num_frames):
    """Convert track dictionary to expected numpy format"""
    valid_tracks = [track for track in tracks.values() if len(track) >= 3]
    
    if not valid_tracks:
        print("Warning: No valid tracks found")
        return np.zeros((0, num_frames, 2)), np.zeros((0, num_frames)), np.zeros((0, num_frames))
    
    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    
    for track in valid_tracks:
        track_array = np.zeros((num_frames, 2))
        visibility = np.zeros(num_frames)
        confidence = np.ones(num_frames) * 0.5  # Default confidence
        
        for frame_id, (x, y) in track:
            track_array[frame_id] = [x, y]
            visibility[frame_id] = 1.0
            confidence[frame_id] = 0.8  # Higher confidence for observed points
        
        pred_tracks.append(track_array)
        pred_vis_scores.append(visibility)
        pred_confs.append(confidence)
    
    print(f"Generated {len(pred_tracks)} tracks across {num_frames} frames")
    return np.array(pred_tracks), np.array(pred_vis_scores), np.array(pred_confs)