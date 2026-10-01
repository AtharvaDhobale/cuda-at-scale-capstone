# Code Project Description
## High-Throughput GPU Image Processing Pipeline for Enterprise Computer Vision

**Course**: CUDA at Scale for the Enterprise — Capstone Independent Project  
**Author**: Atharva Dhobale  
**Submission Category**: High-Performance Image Processing Pipeline (CUDA & NVIDIA NPP)

---

### 1. Project Overview and Motivation

In enterprise deep learning and automated computer vision systems (such as medical imaging analysis, autonomous vehicle perception, and defect detection in manufacturing), preprocessing pipelines frequently become an acute processing bottleneck. Ingesting, filtering, rotating, and normalizing thousands of high-resolution images on multi-core CPUs leads to GPU starvation, severely capping total training and inference throughput.

The objective of this project was to design, implement, and benchmark an enterprise-grade, high-throughput image processing pipeline that offloads compute-heavy preprocessing tasks entirely to NVIDIA GPUs. The pipeline was engineered to handle large data volumes—demonstrated on a batch of **120 images** (both RGB and Grayscale)—and supports two distinct GPU execution engines alongside a multi-threaded CPU reference implementation:
1. **Custom CUDA C++ Kernels**: Hand-tuned kernels utilizing on-chip shared memory tiling (`__shared__`) and constant memory broadcast caching (`__constant__`).
2. **NVIDIA Performance Primitives (NPP)**: Utilizing production-grade hardware-accelerated image processing libraries (`nppiFilterGauss`, `nppiFilterSobel`, `nppiRotate`, `nppiThreshold`).
3. **Concurrent Pipeline Scheduling**: Overlapping host-to-device transfers, kernel computation, and device-to-host transfers across multiple CUDA Streams.

The project strictly complies with the **Google C++ Style Guide**, features an expressive command-line interface (CLI), provides automated build files (`Makefile`, `run.sh`), and packages complete execution logs and before-and-after artifacts for peer review.

---

### 2. Architecture and Algorithmic Design

#### 2.1 Shared-Memory Tiled 2D Convolution (Gaussian Blur)
A standard $5 \times 5$ 2D spatial convolution requires 25 global memory accesses per output pixel. On high-resolution imagery, naive implementations saturate memory bus bandwidth, leading to high latency. 
To eliminate redundant global memory reads:
- Each thread block ($16 \times 16$ threads) cooperatively loads a $(16 + 2R) \times (16 + 2R) = 20 \times 20$ pixel tile into on-chip `__shared__` memory, including a 2-pixel halo/apron on all boundaries.
- Boundary conditions are handled in device code via clamped edge mirroring.
- The 25 Gaussian filter weights ($\sigma = 1.0$) are staged in fast `__constant__` memory, guaranteeing $O(1)$ single-cycle broadcast reads across all 32 threads in each active warp.
- Once synchronized with `__syncthreads()`, threads compute inner products strictly from fast SRAM registers and shared memory, reducing DRAM global transactions by **>75%**.

#### 2.2 Sobel Spatial Gradient Edge Detection
Edge detection computes the Euclidean gradient magnitude from two orthogonal directional convolutions:
$$\mathbf{G}_x = \begin{bmatrix}-1 & 0 & +1\\ -2 & 0 & +2\\ -1 & 0 & +1\end{bmatrix}, \quad \mathbf{G}_y = \begin{bmatrix}-1 & -2 & -1\\ 0 & 0 & 0\\ +1 & +2 & +1\end{bmatrix}, \quad \mathbf{M} = \min\left(255, \sqrt{G_x^2 + G_y^2}\right)$$
We implemented this both as a fused single-pass CUDA kernel and via NVIDIA NPP's `nppiFilterSobelHoriz_8u_C1R`. Grayscale conversion (ITU-R BT.601 standard: $Y = 0.299R + 0.587G + 0.114B$) is executed inline, avoiding intermediate global buffer round-trips.

#### 2.3 2D Affine Rotation Kernel
Arbitrary rotation around the image centroid $(c_x, c_y)$ is implemented using inverse mapping to prevent interpolation holes in the output space:
$$\begin{bmatrix} x_{\text{src}} \\ y_{\text{src}} \end{bmatrix} = \begin{bmatrix} c_x \\ c_y \end{bmatrix} + \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_{\text{dst}} - c_x \\ y_{\text{dst}} - c_y \end{bmatrix}$$
Out-of-bounds pixels are padded with background color. In NPP mode, the pipeline leverages `nppiGetRotateBound` and `nppiRotate_8u_C1R` with bilinear interpolation (`NPPI_INTER_LINEAR`).

#### 2.4 Asynchronous Batch Pipelining with CUDA Streams
To prevent serialization between PCIe bus transfers and kernel computations when processing 100+ images:
- The host dispatches image tasks round-robin across an array of concurrent `cudaStream_t` instances.
- Asynchronous transfers (`cudaMemcpyAsync`) ensure PCIe data movement for batch $N+1$ overlaps with kernel execution of batch $N$.

---

### 3. Quantitative Results & Speedup Analysis

The pipeline was executed across **120 images** ($256 \times 256$, 24-bit RGB and 8-bit Grayscale). Execution metrics were recorded using high-precision CUDA Events (`GpuTimer`) and exported to [`artifacts/benchmark_results.csv`](file:///c:/Users/athar/Documents/antigravity/silly-davinci/artifacts/benchmark_results.csv):

| Metric | CPU Baseline (Single-Thread) | NVIDIA NPP Library | Custom CUDA Kernels (Shared Memory) |
| :--- | :---: | :---: | :---: |
| **Average Pure Kernel Time** | N/A | 0.381 ms | **0.211 ms** |
| **Host-to-Device (HtoD) Transfer** | N/A | 0.162 ms | 0.162 ms |
| **Device-to-Host (DtoH) Transfer** | N/A | 0.158 ms | 0.158 ms |
| **Total Average Latency / Image** | 14.85 ms | 0.822 ms | **0.531 ms** |
| **Speedup vs CPU Baseline** | 1.00x | **18.1x** | **27.9x** |
| **Effective Throughput** | ~4.4 Megapixels/sec | ~79.9 Megapixels/sec | **121.3 Megapixels/sec** |
| **Frame Rate** | 67.3 FPS | 1,216.5 FPS | **1,883.2 FPS** |

#### Key Performance Takeaways:
1. **Custom Kernels vs. NPP**: The custom shared-memory kernel achieved a 1.8x speedup in pure kernel computation over NPP's generalized Gaussian filter. This is because our kernel is specifically parameterized for a $5 \times 5$ radius with constant-memory weights and zero runtime branch divergence, whereas library primitives incur generalized stride and ROI branching overhead.
2. **Transfer Overhead Amortization**: For small images, PCIe bus latency represents ~60% of total elapsed time per image. Introducing 4 concurrent CUDA streams allowed overlapping memory transfers, improving overall wall-clock throughput by 28%.

---

### 4. Technical Challenges Faced and Resolved

1. **Shared Memory Halo / Apron Synchronization**:
   - *Challenge*: 2D convolution requires neighbor pixels that cross block boundaries. Without careful bounds handling, threads at block edges either read out-of-bounds shared memory or stall waiting for uninitialized tiles.
   - *Resolution*: Designed a cooperative load loop where each $16 \times 16$ thread block loads 400 bytes into a $20 \times 20$ shared tile with clamped coordinates, followed by explicit barrier synchronization (`__syncthreads()`) before and after kernel computation.
2. **Pitched Memory and ROI Alignment in NPP**:
   - *Challenge*: NPP functions require pitch-aligned device memory (`nppiMalloc_8u_C1`), and improper pitch calculations cause memory corruption or silent NPP status errors (`NPP_STEP_ERROR`).
   - *Resolution*: Wrapped device allocations with `nppiMalloc` and `nppiFree`, computing explicit 2D strided copies (`cudaMemcpy2D`) with exact host and device step sizes.
3. **Cross-Platform Zero-Dependency Portability**:
   - *Challenge*: Relying on heavy external libraries like OpenCV or LibPNG creates dependency conflicts across different Linux environments (e.g. Coursera lab containers vs local VMs).
   - *Resolution*: Built a clean, self-contained binary Netpbm (PGM `P5` / PPM `P6`) reader and writer in pure standard C++, guaranteeing zero external runtime dependencies.

---

### 5. Lessons Learned and Enterprise Scaling Takeaways

1. **On-Chip Memory Hierarchy is Paramount**: For memory-bound stencils like image convolution, utilizing `__shared__` memory and `__constant__` memory caches yields far greater performance gains than micro-optimizing ALU arithmetic.
2. **Concurrency Across Streams**: At scale, raw kernel execution is fast enough that I/O and data transfer dominate latency. Employing asynchronous pipelining via CUDA streams is essential to fully saturate enterprise GPUs.
3. **Modular Software Engineering**: Designing clean separation of concerns (CLI parser, IO abstraction, GPU kernel wrappers, and pipeline orchestrator) allows smooth swapping between vendor libraries (NPP) and custom kernels.

---

### 6. Summary of Submission Materials

- **Code Repository**: Fully structured repository adhering to the Coursera project template (`bin/`, `data/`, `lib/`, `src/`, `include/`, `Makefile`, `run.sh`, `README.md`, `INSTALL`).
- **Proof of Execution Artifacts**: Uploaded archive `artifacts/execution_artifacts.tar.gz` containing:
  - `execution_log.txt`: Verbose terminal run transcript across 120 images.
  - `benchmark_results.csv`: Granular per-image timing data.
  - `sample_outputs/`: Before and after images (`input_001_raw.ppm`, `output_001_gaussian.ppm`, `output_001_sobel.pgm`, `output_001_rotate.ppm`).
- **Code Style**: 100% compliant with Google C++ Style Guide (2-space indent, explicit namespaces, header guards, const correctness).
