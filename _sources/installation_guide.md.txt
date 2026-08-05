# Installation Issues

See the [README](https://github.com/Kuhlman-Lab/HBDesigner/blob/main/README.md) for information about the various ways to install HBDesigner. 

These installation methods assume a system that is running CUDA 12.8 or greater, but have options for CUDA 12.4. If you are running on a system that has a version of CUDA below 12.4 or a CUDA version too recent to be compatible with CUDA 12.8, continue reading for advice for how to adapt the environment files for your system.

(cuda_version)=
## Adjusting for the CUDA version available on your system

If your system has a more recent version of CUDA (e.g. 13.0) then these installation files will likely still work correctly. See the [Common Issues](#common-issues) section below if you are having trouble.

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
    You can find the correct link to use for your torch/CUDA combination [here](https://pytorch.org/get-started/previous-versions/). 
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

---
(common-issues)=
## Common Issues

### `ImportError: /lib64/libm.so.6: version `GLIBC_2.27' not found`
If you are seeing this, it is most likely because your CUDA version and CUDA version the HBDesigner dependencies are expecting do not match. See [the previous section](#cuda_version) to modify your file for the CUDA version you have access to. 

### `Segmentation Fault (core dumped)`
This can be caused by a variety of different things, including a CUDA version mismatch. If you haven't already modified your environment file to customize it for your particular system, see [the previous section](#cuda_version).
If you have updated these files, try adding the `dev` requirements to your installation. 
