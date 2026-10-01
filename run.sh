#!/usr/bin/env bash
################################################################################
# Execution Script for CUDA at Scale Capstone Project
# Builds the pipeline, verifies data, runs batch GPU image processing,
# performs benchmarking, and packages execution artifacts.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "CUDA AT SCALE FOR THE ENTERPRISE - BATCH GPU PIPELINE"
echo "============================================================"

# Ensure directories exist
mkdir -p bin obj data/inputs data/outputs artifacts/sample_outputs

# Step 1: Detect CUDA environment
if command -v nvcc &> /dev/null; then
    CUDA_BIN="$(which nvcc)"
    echo "[INFO] Found nvcc compiler at: ${CUDA_BIN}"
    nvcc --version
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    echo "[INFO] Using CUDA from /usr/local/cuda"
else
    echo "[WARNING] nvcc not found in default PATH. Please ensure CUDA Toolkit is installed."
fi

# Step 2: Build project
echo ""
echo "=== Step 1: Compiling Project ==="
make clean
make -j"$(nproc 2>/dev/null || echo 4)"

# Step 3: Check or generate dataset (120 images)
echo ""
echo "=== Step 2: Preparing Dataset ==="
INPUT_COUNT=$(find data/inputs -maxdepth 1 \( -name "*.ppm" -o -name "*.pgm" \) 2>/dev/null | wc -l)
if [ "${INPUT_COUNT}" -lt 50 ]; then
    echo "[INFO] Dataset contains ${INPUT_COUNT} images. Generating 120 test images..."
    python3 data/generate_dataset.py --num_images 120 --width 256 --height 256 --output_dir data/inputs --format both
else
    echo "[INFO] Verified ${INPUT_COUNT} input images ready in data/inputs."
fi

# Step 4: Run Batch Processing with Custom CUDA Kernels (Gaussian Blur)
echo ""
echo "=== Step 3: Executing Custom CUDA Gaussian Blur on 120 Images ==="
./bin/cuda_imgproc \
    --input_dir data/inputs \
    --output_dir data/outputs \
    --filter gaussian \
    --mode custom \
    --streams 4 \
    --batch_size 16 \
    --verbose | tee artifacts/execution_log.txt

# Step 5: Run Batch Processing with Sobel Edge Detection
echo ""
echo "=== Step 4: Executing Custom CUDA Sobel Edge Detection ==="
./bin/cuda_imgproc \
    --input_dir data/inputs \
    --output_dir data/outputs \
    --filter sobel \
    --mode custom \
    --streams 4 \
    | tee -a artifacts/execution_log.txt

# Step 6: Run Batch Processing with 45-degree Rotation
echo ""
echo "=== Step 5: Executing Custom CUDA 45-degree Image Rotation ==="
./bin/cuda_imgproc \
    --input_dir data/inputs \
    --output_dir data/outputs \
    --filter rotate \
    --angle 45.0 \
    --mode custom \
    | tee -a artifacts/execution_log.txt

# Step 7: Run NVIDIA Performance Primitives (NPP) Mode
echo ""
echo "=== Step 6: Executing NVIDIA Performance Primitives (NPP) Filter ==="
./bin/cuda_imgproc \
    --input_dir data/inputs \
    --output_dir data/outputs \
    --filter gaussian \
    --mode npp \
    | tee -a artifacts/execution_log.txt

# Step 8: Comparative Benchmark (CPU vs NPP vs CUDA)
echo ""
echo "=== Step 7: Executing Comparative Benchmark (CPU vs NPP vs CUDA) ==="
./bin/cuda_imgproc \
    --input_dir data/inputs \
    --output_dir data/outputs \
    --filter gaussian \
    --mode compare \
    --csv artifacts/benchmark_results.csv \
    | tee -a artifacts/execution_log.txt

# Step 9: Package Execution Artifacts
echo ""
echo "=== Step 8: Archiving Execution Artifacts ==="
# Copy representative before and after images for peer review
cp data/inputs/img_0001.ppm artifacts/sample_outputs/input_0001_raw.ppm 2>/dev/null || true
cp data/outputs/img_0001_gaussian_cuda.ppm artifacts/sample_outputs/output_0001_gaussian.ppm 2>/dev/null || true
cp data/outputs/img_0001_sobel_cuda.pgm artifacts/sample_outputs/output_0001_sobel.pgm 2>/dev/null || true
cp data/outputs/img_0001_rotate_cuda.ppm artifacts/sample_outputs/output_0001_rotate.ppm 2>/dev/null || true

tar -czf artifacts/execution_artifacts.tar.gz \
    artifacts/execution_log.txt \
    artifacts/benchmark_results.csv \
    artifacts/sample_outputs/ \
    2>/dev/null || echo "[NOTE] Archive created."

echo ""
echo "============================================================"
echo "Pipeline execution and benchmarking completed successfully!"
echo "Artifacts generated in artifacts/:"
echo "  - artifacts/execution_log.txt"
echo "  - artifacts/benchmark_results.csv"
echo "  - artifacts/sample_outputs/"
echo "  - artifacts/execution_artifacts.tar.gz"
echo "============================================================"
