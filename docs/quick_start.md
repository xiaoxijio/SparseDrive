# Quick Start

### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

`requirement.txt` uses the pre-built `mmcv-full` and `flash-attn` wheels for
Python 3.8, PyTorch 1.13 and CUDA 11.6. Verify that the active environment
matches before installing:

```bash
python3 -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"
```

### Install a matching CUDA toolkit for compilation

PyTorch is built against CUDA 11.6, so CUDA extensions must also be compiled
with a CUDA 11.6 `nvcc`. If the system toolkit is a different version (for
example CUDA 12.4), install CUDA 11.6 inside the `sparsedrive` Conda
environment. The system CUDA installation does not need to be removed.

```bash
conda activate sparsedrive

# Install the toolkit, including nvcc. Installing cudatoolkit alone is not
# sufficient for compiling CUDA extensions.
conda install cuda -c nvidia/label/cuda-11.6.2

# Use this environment's CUDA toolkit in the current shell.
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Persist CUDA_HOME for future activations of this environment.
conda env config vars set -n sparsedrive CUDA_HOME="${CONDA_PREFIX}"
```

Reactivate the environment after saving `CUDA_HOME`, then verify that both
the compiler and PyTorch use CUDA 11.6:

```bash
conda deactivate
conda activate sparsedrive

which nvcc
nvcc --version
python3 -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print('PyTorch CUDA:', torch.version.cuda); print('CUDA_HOME:', CUDA_HOME)"
```

The CUDA version displayed by `nvidia-smi` is the maximum version supported by
the installed NVIDIA driver and may remain at 12.x. For extension compilation,
use the versions reported by `nvcc --version` and `torch.version.cuda`; both
must be 11.6.

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py clean --all
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```


### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing
```bash
# train
sh scripts/train.sh

# test
sh scripts/test.sh
```

### Visualization
```
sh scripts/visualize.sh
```
