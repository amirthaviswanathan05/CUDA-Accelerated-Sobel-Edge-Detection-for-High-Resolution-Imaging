CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

all: build

build:
	$(NVCC) -O3 src/sobel_filter.cu -o sobel_detector

clean:
	rm -f sobel_detector
