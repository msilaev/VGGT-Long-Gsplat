import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from loop_closure.retrieval.retrieval_dbow import RetrievalDBOW
from loop_utils.visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
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

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []

class VGGT_Long:
    def __init__(self, image_dir, save_dir):
        self.chunk_size = 60
        self.overlap = 25
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, 'tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, 'tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, 'tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.delet_temp_files = True

        print('Loading model...')

        self.model = VGGT()
        # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        _URL = "/media/deng/Data/VGGT-Long (copy)/model.pt"
        state_dict = torch.load(_URL, map_location='cuda')
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None

        if self.sky_mask:
            print('Loading skyseg.onnx...')
            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

            self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer()

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = True

        if self.loop_enable:
            self.retrieval = RetrievalDBOW(vocab_path = "/media/deng/Data/VGGT-Long (copy)/ORBvoc.txt")

        print('init done.')

    def get_loop_pairs(self):
        
        for frame_id, img_path in tqdm(enumerate(self.img_list)):
            image_ori = np.array(Image.open(img_path))
            if len(image_ori.shape) == 2:
                # gray to rgb
                image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

            frame = image_ori # (height, width, 3)
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.retrieval(frame, frame_id)
            cands = self.retrieval.detect_loop(thresh=0.034, num_repeat=3)

            if cands is not None:
                (i, j) = cands # e.g. cands = (812, 67)
                self.retrieval.confirm_loop(i, j)
                self.retrieval.found.clear()
                self.loop_list.append(cands)

            self.retrieval.save_up_to(frame_id)

        # self.loop_list = remove_duplicates(self.loop_list)

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(images)} images")
        
        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
        
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
                    
        # remove useless item to save disk space
        keys_to_remove = ('pose_enc', 'depth', 'depth_conf', 'intrinsic')
        for key in keys_to_remove:
            predictions.pop(key, None)

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
        
        print('Loop SIM(3) estimating...')
        loop_results = process_loop_list(self.chunk_indices, self.loop_list)
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

        del self.model # Save GPU Memory

        print("Aligning all the chunks...")
        # 暂时还没有考虑最后一组数量不够的情况
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(chunk_idx, chunk_idx+1)

            # Load chunk data from file
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            
            # debug saving 
            points = chunk_data['world_points'].reshape(-1, 3)
            colors = (chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = chunk_data['world_points_conf'].reshape(-1)

            print(f"Aligning {chunk_idx} and {chunk_idx+1}")
            # Load chunk data from files
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.5
            s, R, t = align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold=conf_threshold)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))


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
            # Load chunk data from file
            chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
            
            point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
        
            conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.5
            s_a, R_a, t_a = align_point_maps(point_map_a, conf_a, point_map_loop, conf_loop, conf_threshold=conf_threshold)
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
            # Load chunk data from file
            chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
            
            point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
        
            conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.5
            s_b, R_b, t_b = align_point_maps(point_map_b, conf_b, point_map_loop, conf_loop, conf_threshold=conf_threshold)
            print("Estimated Scale:", s_b)
            print("Estimated Rotation:\n", R_b)
            print("Estimated Translation:", t_b)

            print('a -> b SIM 3')
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        
        
        if True:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visualize trajectory

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

        # apply alignment
        print('apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(chunk_idx, chunk_idx+1)
            s, R, t = self.sim3_list[chunk_idx]
            
            # Load chunk data from file
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            # Apply transformation
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)
            
            # Save aligned result
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, chunk_data)
            
            # For the first chunk, just copy to aligned directory
            if chunk_idx == 0:
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            # Load the aligned data for point cloud generation
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
                conf_threshold=np.median(confs) * 1.2,
                sample_ratio=0.015
            )

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
        self.retrieval.close() # Save CPU Memory

        self.process_long_sequence()

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            Each submap typically occupies ~350 MiB of disk space. For an input stream of 4000 images,
            the total temporary files can consume 60-90 GiB of storage. This cleanup is essential to
            prevent unnecessary disk space usage after processing completes.
        '''
        if not self.delet_temp_files:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    args = parser.parse_args()

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    save_dir = os.path.join(
            exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
        )

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')

    vggt_long = VGGT_Long(image_dir, save_dir)
    vggt_long.run()
    vggt_long.close()
    del vggt_long

    all_ply_path = os.path.join(save_dir, f'pcd/combimed_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('VGGT Long done.')
    sys.exit()