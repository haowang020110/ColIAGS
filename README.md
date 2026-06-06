# ColIAGS
Official implementation of our **ICME 2026** paper (camera-ready information will be updated).

**Update:** ColIAGS is selected as oral presentation (<5%).
<p align="center">
  <img src="assets/methodv3.png" />
</p>

---

## Quick Start

```bash
# 1) Create environment
conda create -n ColIAGS python=3.9 -y
conda activate ColIAGS

# 2) Install dependencies
# See the full environment setup below

# 3) Train one C3VD scene
python train.py \
  -s data/C3VD/undistorted_downsize_270x338/<scene_name> \
  -m output/exp/<scene_name> \
  --eval \
  --port 6009
```

---

## 1. Environment Setup

### 1.1 Requirements

- OS: Linux is recommended
- GPU: NVIDIA GPU with CUDA support
- Python: `3.9` (tested)

### 1.2 Installation

We provide a setup script. It is recommended to run it with `source` so that `conda activate` takes effect in the current shell:

```bash
source create_env.sh
```

If your shell has not initialized Conda yet, run:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
```

Or install the environment manually:

```bash
conda create -n ColIAGS python=3.9 -y
conda activate ColIAGS

# PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# Common dependencies
pip install opencv-python einops tqdm plyfile scipy natsort OpenEXR
pip install numpy==1.26.4

# Required, but may need a wheel matching your PyTorch/CUDA version
pip install torch_scatter

# CUDA extensions
pip install submodules/simple-knn/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization --no-build-isolation
```


---

## 2. Dataset Organization

The `data/` directory is ignored by git in this repository. Please prepare your datasets locally and place them under `data/`.

The code currently supports **three** dataset layouts. The loader is selected automatically based on the `-s/--source_path` string and the presence of key files.

### 2.1 C3VD (standard format)

Expected structure for each scene:

```text
data/C3VD/undistorted_downsize_270x338/<scene_name>/
  camera_pose.txt
  camera.json
  images/
    0_color.png
    1_color.png
    ...
  depths/
    0_depth.png
    1_depth.png
    ...
```

Key files:
- `camera_pose.txt`: comma-separated 4x4 camera-to-world matrices, one per frame.
- `camera.json`: camera intrinsics and image size. It should include `fx`, `fy`, `cx`, `cy`, `h`, and `w`.
- Image/depth naming: the current loader expects filenames in the form `"{idx}_*.png"`, for example `0_color.png` and `0_depth.png`, and checks index consistency.

### 2.2 C3VD with EndoGSLAM initialization
**NOTE: We use the data released by [PR-Endo](https://github.com/SanoScience/PR-ENDO).**

This loader is triggered when the `source_path` contains both `C3VD` and `pr-endo`.

Scene folder:

```text
data/C3VD/pr-endo/C3VD/<scene_name>/
  color/
    *.png
  depth/
    *.tiff
```


### 2.3 ColonRotate
**NOTE: We use the data released by [PR-Endo](https://github.com/SanoScience/PR-ENDO).**
Expected structure:

```text
data/ColonRotate/
  transforms.json
  transforms_test.json
  train_views/
    0000.png
    0001.png
    ...
  depth_train/
    0000.exr
    0001.exr
    ...
  test_views_1/
    0000.png
    0001.png
    ...
  init_point_cloud.ply
```

---

## 3. Training

### 3.1 Train a single scene

```bash
# Example: one standard C3VD scene
python train.py \
  -s data/C3VD/undistorted_downsize_270x338/<scene_name> \
  -m output/exp/<scene_name> \
  --eval \
  --port 6009
```

Arguments:
- `-s` / `--source_path`: input scene path
- `-m` / `--model_path`: output directory
- `--eval`: enables the evaluation split for datasets that support it
- `--port`: viewer port
- `--disable_viewer`: disable the viewer socket during training

---



## 4. Acknowledgements

This codebase is built upon or inspired by the following excellent open-source projects:

- [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
- [PR-Endo](https://github.com/SanoScience/PR-ENDO)

We sincerely thank the authors of these projects for making their code publicly available.

---

## 5. Citation

If you find this repository useful, please cite our ICME 2026 paper.

> The final BibTeX entry will be updated after the camera-ready version is finalized.

```bibtex
@article{wang2025moving,
  title={Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting},
  author={Wang, Hao and Zhou, Ying and Zhao, Haoyu and Wang, Rui and Hu, Qiang and Zhang, Xing and Li, Qiang and Wang, Zhiwei},
  journal={arXiv preprint arXiv:2510.18739},
  year={2025}
}
```
