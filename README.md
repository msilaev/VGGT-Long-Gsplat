<p align="center">
<p align="center">
<h1 align="center">VGGT-Long: Chunk it, Loop it, Align it – Pushing VGGT’s Limits on Long RGB Sequences</h1>
</p>

### **Change Log**

`[14 Jun 2025]` GitHub code release.

##  Setup, Installation & Running

### 🖥️ 1 - Hardware and System Environment 

This project was developed, tested, and run in the following hardware/system environment

```
Hardware Environment：
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

Creating a virtual environment using conda (or miniconda)

```bash
conda create -n vggt-long python=3.10
conda activate vggt-long
# pip version created by conda: 25.1
```

Next, install `PyTorch`

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# Verified to work with CUDA 11.8 and torch 2.2.0
```

Then install other dependencies exactly same as [VGGT](https://github.com/facebookresearch/vggt)

```bash
pip install -r requirements.txt
```

#### Step 2 (Optional): Compile Loop-Closure Correction Module

We provide a Python-based Sim3 solver, so `VGGT-Long` can run the loop closure correction solving without compiling `C++` code. Therefore, this step is optional and you can skip it. However, we still recommend installing the `C++` solver as it is more stable and faster.

```bash
python setup.py install
```



#### Step 3: Compile Loop-Closure Detection Module

Our goal is to use as much pure `Python` as possible. However, currently `C++` compilation is still necessary to ensure the **CPU memory efficiency** of the code. We will make this `C++` installation step **optional** in upcoming updates.

Install the `OpenCV C++ API`.

```
sudo apt-get install -y libopencv-dev
```

Install `DBoW2`

```
cd DBoW2
mkdir -p build && cd build
cmake ..
make
sudo make install
cd ../..
```

Install the image retrieval

```
pip install ./DPRetrieval
```

#### Step 4: Bag of Words Model Setup

Download the pre-trained Bag of Words vocabulary for `DBoW2`:

```bash
# Download the vocabulary file (about 150 MiB)
wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
# You could download manually from the link via your browser

# Extract the vocabulary file
tar -xzvf ORBvoc.txt.tar.gz

# Verify if the extraction is successful
ls -l ORBvoc.txt
```

### 🚀 3 - Running the code 


```bash
python vggt-long.py --image_dir ./path/to/your/image_sequence
```

## Acknowledgements

Our project is based on [VGGT](https://github.com/facebookresearch/vggt), [DPV-SLAM](https://github.com/princeton-vl/DPVO), [GigaSLAM](https://github.com/DengKaiCQ/GigaSLAM). Our work would not have been possible without these excellent repositories.
