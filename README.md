<p align="center">
<p align="center">
<h1 align="center">Gaussian splats from VGGT on large number of frames</h1>
</p>

This repository realizes VGGT-Long-> Gaussian splatting pipeline, combining with some modifications 
codes from the repositories 

 [VGGT-Long](https://github.com/facebookresearch/vggt)
 [VGGT](https://github.com/facebookresearch/vggt)

from the papers 

[VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences](https://arxiv.org/abs/2507.16443)
and

[VGGT-](https://arxiv.org/abs/---)

For the Gaussian splatting with use code from gsplat repository 

[gsplat](https://github.com/nerfstudio-project/gsplat)

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

We use three virtual environmemts - 'vggfm-temp' with requirements from VGGT-Long repository, 
'colmap_env' with requirenets from VGGT repository and 'py11' with gsplat  dependencies.
Create these conda environments and install dependencies using corresponding 'requirements.txt' files.

#### Step 2: Weights Download

Download all the pre-trained weights needed:

```cmd
bash ./scripts/download_weights.sh
```

### Running experiments 

The simpiest way to run pipeline is on the remote machine through ssh connection using scripts
'
./src_vggt_colmap/run_pipeline_vggt_long_colmap.sh
'
and then downloading results
'
./src_vggt_colmap/download_results_vggt_colmap.sh



## License

Theis  codebase follows `VGGT`'s license, please refer to `./LICENSE.txt` for applicable terms. For commercial use, please follow the link [VGGT](https://github.com/facebookresearch/vggt) that should utilize the commercial version of the pre-trained weight. [Link of VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial)

## More Experiments
