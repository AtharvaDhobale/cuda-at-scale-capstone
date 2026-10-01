################################################################################
# Makefile for CUDA at Scale for the Enterprise Capstone Project
# Scalable High-Throughput GPU Image Processing Pipeline
################################################################################

# CUDA installation path detection
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++

# Target binary
TARGET_DIR := bin
TARGET := $(TARGET_DIR)/cuda_imgproc

# Source and Include directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
DATA_DIR := data

# Compiler and Linker flags
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -I$(INC_DIR) -I$(CUDA_PATH)/include
NVCCFLAGS := -std=c++17 -O3 -I$(INC_DIR) -I$(CUDA_PATH)/include \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             --compiler-options '-fPIC -Wall'

LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart \
           -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc \
           -lpthread

# Source files
CPP_SRCS := $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/image_io.cpp \
            $(SRC_DIR)/filters_cpu.cpp \
            $(SRC_DIR)/filters_npp.cpp \
            $(SRC_DIR)/pipeline.cpp

CU_SRCS := $(SRC_DIR)/filters_cuda.cu

# Object files
OBJS := $(OBJ_DIR)/main.o \
        $(OBJ_DIR)/image_io.o \
        $(OBJ_DIR)/filters_cpu.o \
        $(OBJ_DIR)/filters_npp.o \
        $(OBJ_DIR)/filters_cuda.o \
        $(OBJ_DIR)/pipeline.o

.PHONY: all clean run help data test

# Default rule
all: $(TARGET)

# Link executable
$(TARGET): $(OBJS) | $(TARGET_DIR)
	@echo "[LINK] $@"
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

# Compile C++ sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "[CXX]  $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@echo "[NVCC] $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Create directories
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Generate test dataset of 120 images
data:
	@echo "[DATA] Generating 120 test images..."
	python3 $(DATA_DIR)/generate_dataset.py --num_images 120 --output_dir $(DATA_DIR)/inputs

# Run pipeline
run: $(TARGET)
	@echo "[RUN] Executing batch pipeline..."
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter gaussian --mode custom --verbose

# Run comparative benchmark
benchmark: $(TARGET)
	@echo "[BENCHMARK] Executing comparative evaluation (CPU vs NPP vs CUDA)..."
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter gaussian --mode compare --verbose

# Run all test filter stages
test: $(TARGET)
	@echo "[TEST] Running all filter operations..."
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter gaussian --mode custom
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter sobel --mode custom
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter rotate --angle 45 --mode custom
	./$(TARGET) --input_dir $(DATA_DIR)/inputs --output_dir $(DATA_DIR)/outputs --filter gaussian --mode npp
	@echo "All tests passed successfully!"

# Clean build artifacts
clean:
	@echo "[CLEAN] Removing build objects and binaries..."
	rm -rf $(OBJ_DIR) $(TARGET_DIR)/cuda_imgproc
	rm -rf $(DATA_DIR)/outputs/*

# Display help
help:
	@echo "============================================================"
	@echo "CUDA at Scale Image Processing Pipeline - Build Commands"
	@echo "============================================================"
	@echo "  make            - Build target executable ($(TARGET))"
	@echo "  make data       - Generate synthetic dataset of 120 images"
	@echo "  make run        - Execute pipeline on input dataset"
	@echo "  make benchmark  - Run comparative benchmark (CPU vs NPP vs CUDA)"
	@echo "  make test       - Run multi-stage test suite across all filters"
	@echo "  make clean      - Clean object files and binaries"
	@echo "  make help       - Display this help message"
	@echo "============================================================"
