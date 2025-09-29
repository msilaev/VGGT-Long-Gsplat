import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2

import gc
import torch.nn.functional as F

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
# from loop_utils.visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config

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

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []

        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = []

## New addition

        self.combined_depth_maps = []
        self.combined_depth_confs = []

        self.all_camera_depths = []
        self.all_camera_depth_confs = []

class VGGT_Long:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 
        
        ## New addition
        self.all_camera_depths = []
        self.all_camera_depths_confs = []

        self.delete_temp_files = self.config['Model']['delete_temp_files']

        print('Loading model...')

        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        #_URL = self.config['Weights']['VGGT']
        #state_dict = torch.load(_URL, map_location='cuda')
        #self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None

        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # Download skyseg.onnx if it doesn't exist
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        #images = load_and_preprocess_images(chunk_image_paths).to(self.device)

        vggt_fixed_resolution = 518
        img_load_resolution = 1024

        #print(chunk_image_paths)

        images, _ = load_and_preprocess_images_square(chunk_image_paths, img_load_resolution)
        images = images.to(self.device)
        #original_coords = original_coords.to(self.device)

        #extrinsic_1, intrinsic_1, depth_map_1, depth_conf_1 = run_VGGT(self.model, images, self.dtype, vggt_fixed_resolution)

        images = F.interpolate(images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False)

        print(f"Loaded {len(images)} images")
        
        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)
        torch.cuda.empty_cache()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        
        #extrinsic = extrinsic.cpu().numpy().squeeze(0)
        #intrinsic = intrinsic.cpu().numpy().squeeze(0)

        #print(f"Example extrinsics[1]:\n", extrinsic[1])
        #print(f"Example extrinsics[2]:\n", extrinsic[2])
        
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        #predictions["depth"] = depth_map
        #predictions["depth_conf"] = depth_conf

        #print(f"Example extrinsics_1[1]:\n", extrinsic_1[1])
        #print(f"Example extrinsics_1[2]:\n", extrinsic_1[2])



        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
                    
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            
            chunk_range = self.chunk_indices[chunk_idx]

            ##### New addition
            depth_maps = predictions['depth']

            depth_confs = predictions['depth_conf']

            print(f"shape of depth maps: {depth_maps.shape}")
            print(f"shape of depth confidence maps: {depth_confs.shape}")
            print(f"shape of extrinsics: {extrinsics.shape}")
            print(f"shape of intrinsics: {intrinsics.shape}")

            #print(f"Test extrinsics[1]:\n", extrinsics[1] )
            #print(f"Test extrinsics[2]:\n", extrinsics[2] )


            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

            self.all_camera_depths.append((chunk_range, depth_maps))
            self.all_camera_depths_confs.append((chunk_range, depth_confs))

        predictions['depth'] = np.squeeze(predictions['depth'])

        np.save(save_path, predictions)
        
        return predictions if is_loop or range_2 is not None else None
    
    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))
            
            # SMART CHUNKING: Merge small last chunk with previous chunk
            if len(self.chunk_indices) > 1:
                last_chunk_size = self.chunk_indices[-1][1] - self.chunk_indices[-1][0]
                min_chunk_size = max(self.overlap + 5, self.chunk_size * 0.3)  # At least overlap + buffer
                
                if last_chunk_size < min_chunk_size:
                    print(f"WARNING: Last chunk too small ({last_chunk_size}), merging with previous chunk...")
                    
                    # Remove last chunk and extend previous chunk
                    last_start, last_end = self.chunk_indices.pop()
                    prev_start, _ = self.chunk_indices.pop()
                    merged_chunk = (prev_start, last_end)
                    self.chunk_indices.append(merged_chunk)
                    
                    merged_size = merged_chunk[1] - merged_chunk[0]
                    print(f"Merged chunk size: {merged_size} (was {last_chunk_size} + previous)")
        
        # DEBUG: Check for problematic last chunk
        print(f"\n=== CHUNK SIZE ANALYSIS ===")
        print(f"Total images: {len(self.img_list)}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        print(f"Number of chunks: {len(self.chunk_indices)}")
        
        for i, (start, end) in enumerate(self.chunk_indices):
            chunk_size = end - start
            print(f"Chunk {i}: [{start}, {end}) = {chunk_size} images")
            
            if chunk_size < self.overlap and i > 0:
                print(f"  WARNING: Chunk {i} has {chunk_size} images < overlap {self.overlap}!")
                print(f"  This will cause alignment issues with chunk {i-1}")
            
            if chunk_size < self.chunk_size * 0.5 and i == len(self.chunk_indices) - 1:
                print(f"  WARNING: Last chunk {i} is very small ({chunk_size} images)")
                print(f"  Consider merging with previous chunk or adjusting parameters")
        print(f"=== END CHUNK ANALYSIS ===\n")
        
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices, 
                                             self.loop_list, 
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
            

        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        for chunk_idx in range(len(self.chunk_indices)):

            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            # SAFE OVERLAP: Handle variable chunk sizes
            chunk1_size = chunk_data1['world_points'].shape[0]
            chunk2_size = chunk_data2['world_points'].shape[0]
            
            # Use actual overlap size, capped by chunk sizes
            actual_overlap = min(self.overlap, chunk1_size, chunk2_size)
            
            print(f"  Chunk {chunk_idx} size: {chunk1_size}, Chunk {chunk_idx+1} size: {chunk2_size}")
            print(f"  Using overlap: {actual_overlap} (requested: {self.overlap})")
            
            if actual_overlap < 3:
                print(f"  WARNING: Very small overlap ({actual_overlap}) may cause poor alignment!")
            
            point_map1 = chunk_data1['world_points'][-actual_overlap:]
            point_map2 = chunk_data2['world_points'][:actual_overlap]
            conf1 = chunk_data1['world_points_conf'][-actual_overlap:]
            conf2 = chunk_data2['world_points_conf'][:actual_overlap]

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2, 
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))


        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
            
                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
            
                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))


        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f'Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})')
            s, R, t = self.sim3_list[chunk_idx]
            
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)
            
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, chunk_data)
            
            if chunk_idx == 0:
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            
            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,              # shape: (H, W, 3)
                colors=colors,              # shape: (H, W, 3)
                confs=confs,          # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        self.save_camera_poses()
        
        print('Done.')

    
    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                                glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close() # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector # Save GPU Memory
        torch.cuda.empty_cache()

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],    # Dark Red
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")
        
        all_poses = [None] * len(self.img_list)
        all_poses_original = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)
        all_depths = [None] * len(self.img_list)
        all_depth_confs = [None] * len(self.img_list)
        all_poses_original = [None]* len(self.img_list)
        all_poses_w2c=[None]* len(self.img_list)  # Save as 3x4 W2C format
                
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        _, first_chunk_depths = self.all_camera_depths[0]
        _, first_chunk_depths_confs = self.all_camera_depths_confs[0]

        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):

            # Save original poses -added
            all_poses_original[idx] = first_chunk_extrinsics[i]

            #print(f"First chunk original pose for image {idx}:\n", first_chunk_extrinsics[i])

            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i] 
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]
            # added for depth
            all_depths[idx] = first_chunk_depths[i]
            all_depth_confs[idx] = first_chunk_depths_confs[i]        
        
        print("number of chunks:", len(self.all_camera_poses))
        for chunk_idx in range(1, len(self.all_camera_poses)):
            
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]

            # added for depth
            _, chunk_depths = self.all_camera_depths[chunk_idx]
            _, chunk_depths_confs = self.all_camera_depths_confs[chunk_idx]


            s, R, t = self.sim3_list[chunk_idx-1]   # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.
            
            print(f"Applying SIM(3) to chunk {chunk_idx}: Scale {s}, Rotation:\n{R}, Translation: {t}")

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            print(f"Scale matrix S: {s} for chunk {chunk_idx}  ")

            print(S)

            # Test alignment BEFORE overwriting overlap poses
            if chunk_idx == 1:  # Only test for first chunk pair to avoid spam
                prev_chunk_range = self.chunk_indices[chunk_idx-1]
                curr_chunk_range = chunk_range
                
                # Find overlap region
                overlap_start = max(prev_chunk_range[0], curr_chunk_range[0])
                overlap_end = min(prev_chunk_range[1], curr_chunk_range[1])
                
                print(f"\n=== ALIGNMENT TEST for chunks {chunk_idx-1} and {chunk_idx} ===")
                print(f"Previous chunk range: {prev_chunk_range}")
                print(f"Current chunk range: {curr_chunk_range}")
                print(f"Overlap range: [{overlap_start}, {overlap_end})")
                
                if overlap_start < overlap_end:
                    print("Testing pose alignment for overlap images:")
                    
                    # Compare poses for overlap images
                    for test_idx in range(overlap_start, min(overlap_start + 3, overlap_end)):
                        if all_poses[test_idx] is not None:
                            # Pose from previous chunk (already in all_poses)
                            prev_pose = all_poses[test_idx]
                            
                            # Get corresponding pose from current chunk and transform it
                            curr_chunk_local_idx = test_idx - curr_chunk_range[0]
                            if 0 <= curr_chunk_local_idx < len(chunk_extrinsics):
                                w2c_curr = np.eye(4)
                                w2c_curr[:3, :] = chunk_extrinsics[curr_chunk_local_idx]
                                c2w_curr = np.linalg.inv(w2c_curr)
                                transformed_curr = S @ c2w_curr
                                
                                # Compare positions
                                pos_diff = np.linalg.norm(prev_pose[:3, 3] - transformed_curr[:3, 3])

                                # Compare rotation (Frobenius norm of rotation difference)
                                rot_diff = np.linalg.norm(prev_pose[:3, :3] - transformed_curr[:3, :3], 'fro')

                                # Compare depth maps for overlap images
                                prev_chunk_depths = self.all_camera_depths[chunk_idx-1][1]
                                curr_chunk_depths = self.all_camera_depths[chunk_idx][1]
                                
                                prev_depth_local_idx = test_idx - prev_chunk_range[0]
                                if (0 <= prev_depth_local_idx < len(prev_chunk_depths) and 
                                    0 <= curr_chunk_local_idx < len(curr_chunk_depths)):
                                    
                                    prev_depth = prev_chunk_depths[prev_depth_local_idx]
                                    curr_depth = curr_chunk_depths[curr_chunk_local_idx]
                                    
                                    # Compare depth statistics
                                    prev_depth_mean = np.mean(prev_depth)
                                    curr_depth_mean = np.mean(curr_depth)
                                    prev_depth_std = np.std(prev_depth)
                                    curr_depth_std = np.std(curr_depth)
                                    
                                    depth_mean_diff = abs(prev_depth_mean - curr_depth_mean)
                                    depth_std_diff = abs(prev_depth_std - curr_depth_std)
                                    
                                    # Pixel-wise depth difference (sample subset for efficiency)
                                    h, w = prev_depth.shape[:2]
                                    sample_mask = np.random.choice(h*w, min(1000, h*w//4), replace=False)
                                    prev_sample = prev_depth.flatten()[sample_mask]
                                    curr_sample = curr_depth.flatten()[sample_mask]
                                    pixel_diff_mean = np.mean(np.abs(prev_sample - curr_sample))
                                    
                                    print(f"  Image {test_idx}: position diff = {pos_diff:.6f}, rotation diff = {rot_diff:.6f}")
                                    print(f"    Depth mean diff = {depth_mean_diff:.4f}, std diff = {depth_std_diff:.4f}")
                                    print(f"    Pixel-wise depth diff = {pixel_diff_mean:.4f}")
                                else:
                                    print(f"  Image {test_idx}: position diff = {pos_diff:.6f}, rotation diff = {rot_diff:.6f}")
                                    print(f"    Depth comparison: index out of bounds")
                                
                                if pos_diff > 0.1:
                                    print(f"  WARNING: Large position difference for image {test_idx}!")
                                if rot_diff > 0.1:
                                    print(f"  WARNING: Large rotation difference for image {test_idx}!")
                                if 'depth_mean_diff' in locals() and depth_mean_diff > 1.0:
                                    print(f"  WARNING: Large depth mean difference for image {test_idx}!")
                                if 'pixel_diff_mean' in locals() and pixel_diff_mean > 0.5:
                                    print(f"  WARNING: Large pixel-wise depth difference for image {test_idx}!")
                print("=== END ALIGNMENT TEST ===\n")

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):

                # Save original poses - added
                all_poses_original[idx] = chunk_extrinsics[i]

                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i]

                # New addition for depth
                all_depths[idx] = chunk_depths[i]
                all_depth_confs[idx] = chunk_depths_confs[i]
               
        
        
        print(f"Example aligned pose for image 1:\n", all_poses_original[1])
        print(f"Example aligned pose for image 2:\n", all_poses_original[2])

        print(f"Example intrinsic for image 1:\n", all_intrinsics[1])
        print(f"Example intrinsic for image 2:\n", all_intrinsics[2])

        print(f"Example depth map for image 1 shape:\n", all_depths[1][1,1])
        print(f"Example depth map for image 2 shape:\n", all_depths[2][1,1])

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        
        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
        with open(intrinsics_path, 'w') as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f'{fx} {fy} {cx} {cy}\n')

        print(f"Camera intrinsics saved to {intrinsics_path}")
        
        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        
        print(f"Camera poses visualization saved to {ply_path}")

        # New addition
        
        depth_path = os.path.join(self.output_dir, 'depth_maps.npy')
        np.save(depth_path, all_depths)
        print(f"Depth maps saved to {depth_path}")
        depth_conf_path = os.path.join(self.output_dir, 'depth_confs.npy')
        np.save(depth_conf_path, all_depth_confs)
        print(f"Depth confidence maps saved to {depth_conf_path}")

        intrinsics_path = os.path.join(self.output_dir, 'intrinsic.npy')
        np.save(intrinsics_path, all_intrinsics)
        print(f"Camera intrinsics saved to {intrinsics_path}")

        # Convert aligned C2W poses back to W2C format for demo_colmap.py
        # COLMAP needs poses in a consistent global coordinate system (after SIM(3) alignment)
        all_poses_w2c = []
        for pose_c2w in all_poses:
            w2c = np.linalg.inv(pose_c2w)[:3, :]  # Convert to W2C and take the first 3 rows
            all_poses_w2c.append(w2c)  # Save as 3x4 W2C format

        extrinsic_path = os.path.join(self.output_dir, 'extrinsic.npy')
        np.save(extrinsic_path, all_poses_w2c)

        print("Example original W2C pose for image 1:\n", all_poses_original[1])
        print("Example aligned W2C pose for image 1:\n", all_poses_w2c[1])

        print(f"Camera extrinsics (aligned W2C) saved to {extrinsic_path}")
        
        # Also save C2W format for backward compatibility
        c2w_path = os.path.join(self.output_dir, 'extrinsic_c2w.npy')
        np.save(c2w_path, all_poses)
        print(f"Camera extrinsics (C2W aligned) saved to {c2w_path}")

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='Image path')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'


    #save_dir = os.path.join(
    #        exp_dir, image_dir.replace("/", "_"), current_datetime
    #    )

    save_dir = os.path.join(
        exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1]
    )

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_long = VGGT_Long(image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('VGGT Long done.')
    
    # Print helpful command for next step
    print("")
    print("=" * 60)
    print("To run COLMAP demo, use:")
    print(f'python demo_colmap.py --scene_dir "{image_dir}" --output_dir "{save_dir}" --use_ba')
    print("")
    print("Or use the pipeline script:")
    print(f'./pipeline.ps1 -ImageDir "{image_dir}" -UseBa')
    print("=" * 60)
    
    sys.exit()