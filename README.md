<<p align="center">
<h1 align="center">Gaussian Splatting from VGGT on Large Number of Frames</h1>
</p>

This repository implements a VGGT-Long → Gaussian Splatting pipeline, combining and modifying code from the following repositories:

- [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)
- [VGGT](https://github.com/facebookresearch/vggt)

Based on the papers:

- [VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences](https://arxiv.org/abs/2507.16443)
- [VGGSfM: Visual Geometry Grounded Deep Structure from Motion](https://arxiv.org/abs/2312.04563)

For Gaussian Splatting, we use code from the gsplat repository:

- [gsplat](https://github.com/nerfstudio-project/gsplat)

##  Setup, Installation & Running

### 🖥️ 1 - Hardware and System Environment 

This project was developed, tested, and run in the following hardware/system environment

```
Hardware Environment are same as for VGGT-Long paper：
    CPU(s): Intel Xeon(R) Gold 6128 CPU @ 3.40GHz × 12
    GPU(s): NVIDIA RTX 4090 (24 GiB VRAM)
    RAM: 67.0 GiB (DDR4, 2666 MT/s)
    Disk: Dell 8TB 7200RPM HDD (SATA, Seq. Read 220 MiB/s)

System Environment：
    Linux System: Ubuntu 22.04.3 LTS
    CUDA Version: 11.8
    cuDNN Version: 9.1.0
    NVIDIA Drivers: 555.42.06
    Conda version: 23.9.0 (Miniconda)
```

### 📦 2 - Environment Setup 

#### Step 1: Dependency Installation

We use three virtual environments:
- `vggsfm_tmp`: Requirements from VGGT-Long repository
- `colmap_env`: Requirements from VGGT repository  
- `py11`: gsplat dependencies

Create these conda environments and install dependencies using the corresponding `requirements.txt` files.

#### Step 2: Weights Download

Download all the pre-trained weights needed:

```cmd
bash ./scripts/download_weights.sh
```

### 🚀 Running Experiments 

The simplest way to run the pipeline is on a remote machine through SSH connection using the scripts:

**Run the pipeline:**
```bash
./src_vggt_colmap/run_pipeline_vggt_long_colmap.sh
```

**Download results:**
```bash
./src_vggt_colmap/download_results_vggt_colmap.sh
```



## 📄 License

This codebase follows VGGT's license. Please refer to `./LICENSE.txt` for applicable terms. 

For commercial use, please follow the link to [VGGT](https://github.com/facebookresearch/vggt) and utilize the commercial version of the pre-trained weights: [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial)

## 📖 References

If you use this code, please cite the original papers:

```bibtex
@article{wang2024vggt-long,
  title={VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences},
  author={Wang, Ziyuan and others},
  journal={arXiv preprint arXiv:2507.16443},
  year={2024}
}

@article{wang2023vggsfm,
  title={VGGSfM: Visual Geometry Grounded Deep Structure from Motion},
  author={Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
  journal={arXiv preprint arXiv:2312.04563},
  year={2023}
}
```

## 🔬 More Experiments

[Content for additional experiments and results would go here]
