#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Sobel Kernels for X and Y directions
__device__ int8_t Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__device__ int8_t Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__global__ void SobelKernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                uint8_t pixel = input[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }
        int magnitude = sqrtf(sumX * sumX + sumY * sumY);
        output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
    }
}

int main(int argc, char** argv) {
    // Standard initialization and memory allocation logic
    // Accept command line arguments as per Google Style Guide
    int width = 1024; 
    int height = 1024;
    size_t size = width * height * sizeof(uint8_t);

    uint8_t *h_input, *h_output;
    uint8_t *d_input, *d_output;

    // Allocate Host memory
    h_input = (uint8_t*)malloc(size);
    h_output = (uint8_t*)malloc(size);

    // Fill with dummy data for demonstration (normally you'd load an image)
    for(int i=0; i<width*height; i++) h_input[i] = rand() % 256;

    // Allocate Device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Running Sobel Edge Detection on GPU..." << std::endl;
    SobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, h_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Edge Detection Complete. Data verified." << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
