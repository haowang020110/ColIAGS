conda create -n ColIAGS python=3.9 -y
conda activate ColIAGS
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python einops tqdm plyfile torch_scatter scipy natsort OpenEXR
pip install numpy==1.26.4
pip install submodules/simple-knn/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization --no-build-isolation