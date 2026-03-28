# CUDA-Accelerated-Sobel-Edge-Detection-for-High-Resolution-Imaging
# CUDA-Accelerated Image Processing Pipeline
### Capstone Project: CUDA at Scale for the Enterprise

## Project Description
This project implements a high-performance Sobel Edge Detection filter using native CUDA kernels. The goal is to demonstrate the throughput advantages of GPU acceleration when processing large-scale image datasets (such as the USC SIPI database). By offloading the pixel-wise convolution operations to the GPU, we achieve a significant speedup compared to traditional CPU-based sequential processing.

## Features
- **Custom CUDA Kernel:** Implements 3x3 convolution for horizontal and vertical gradient calculation.
- **Unified Memory:** Utilizes `cudaMallocManaged` for efficient data handling (or standard `cudaMalloc` for performance tuning).
- **Command Line Flexibility:** Accepts input/output file paths and thread block dimensions as arguments.
- **Google C++ Style:** Adheres to industry-standard coding conventions.

## Prerequisites
- NVIDIA GPU (Compute Capability 3.5+)
- CUDA Toolkit 11.0+
- GCC/G++ 5.1.0+
- OpenCV (Optional, for image decoding) or Simple PPM/PGM parser.

## Build and Execution
To build the project, use the provided Makefile:
```bash
make build
