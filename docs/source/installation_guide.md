# Installation Guide

# Contents


---
## Installation with Conda

### GPU

### CPU

---
## Installation with Mamba

### GPU

### CPU

---
## Installation with uv

### GPU

### CPU

---
## Installation with pixi

### GPU
### CPU

(cuda_version)=
## Adjusting for the CUDA version available on your system
The various installation files (`env.yaml`, `pyproject.toml`) were created for systems running CUDA 12.8. 

If your system has a more recent version of CUDA (e.g. 13.0) then these installation files may still work correctly. See the [Common Issues](#common-issues) section below if you are having trouble.

If your system has an older version of CUDA, then there are several dependencies you may need to change: 
- The PyTorch source: 
    For example, if you have CUDA 12.4, 
    ```bash
    --extra-index-url https://download.pytorch.org/whl/cu128
    --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
    ```
    will become
    ```bash
    --extra-index-url https://download.pytorch.org/whl/cu124
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
    ```
    You can find the correct link to use [here](https://pytorch.org/get-started/previous-versions/). 
- The torch version will need to be changed:
    ```bash
    torch==2.8.0+cu128
    ```
    would be come
    ```bash
    torch==2.6.0+cu124
    ```
    (As depicted in the example, you may have to change the torch version number depending on what is available for your CUDA version.)
- You should then be able to remove the version numbers from the following dependencies: 
    - nvidia-cublas-cu12
    - nvidia-cuda-cupti-cu12
    - nvidia-cuda-nvrtc-cu12
    - nvidia-cuda-runtime-cu12
    - nvidia-cudnn-cu12
    - nvidia-cufft-cu12
    - nvidia-cufile-cu12
    - nvidia-curand-cu12
    - nvidia-cusolver-cu12
    - nvidia-cusparse-cu12
    - nvidia-cusparselt-cu12
    - nvidia-nccl-cu12
    - nvidia-nvjitlink-cu12
    - nvidia-nvtx-cu12 
    - sympy
    - torch-cluster
    - torch-scatter
- You may need to find a version of triton that works with your PyTorch/CUDA combination, or you might be able to remove the version requirement from the triton listing. For CUDA 12.4, `triton==3.2.0` works. 
- 

---
## Common Issues

### `ImportError: /lib64/libm.so.6: version `GLIBC_2.27' not found`
If you are seeing this, it is most likely because your CUDA version and the YAML or TOML file you are using to install the dependencies do not match. See [the previous section](#cuda_version) to modify your file for the CUDA version you have access to. 

### `Segmentation Fault (core dumped)`
This can be caused by a variety of different things, including a CUDA version mismatch. If you haven't already modified your TOML or YAML file, see [the previous section](#cuda_version).
If you have updated these files, try adding the `dev` requirements to you installation instructions. 
