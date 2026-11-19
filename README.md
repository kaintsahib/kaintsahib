# 🪐 Gaurav Sahib | GPU Kernel & AI Systems Engineer

Operating at the boundary layer between deep learning graphs and bare-metal silicon physics, focusing on bypassing framework overheads using **CUDA C++**, **PTX/SASS**, **OpenAI Triton**, and **AI Compilers**.

---

## ⚡ Core Technical Stack

- **Low-Level GPU & Systems:** C++20 (`CMake + Ninja`), CUDA C++, PTX / SASS Inspection, NVIDIA CUTLASS / CuTe (In Progress).
- **AI Compilers & DSLs:** OpenAI Triton (manual pointer strides, L2 cache swizzling & autotuning), Meta Helion DSL.
- **Hardware Profiling & Telemetry:** NVIDIA Nsight Compute (`ncu`) & Nsight Systems (`nsys`).
- **Distributed Inference Engines:** vLLM (`PagedAttention`), SGLang (`RadixAttention`).

---

## 🛠 Verified Repositories & Hardware Benchmarks

### 1. [triton-gpu-kernel](https://github.com) — Custom Triton Kernels & Nsight Silicon Profiling
Hand-written OpenAI Triton kernels profiled on NVIDIA Turing silicon (GTX 1650, TU117, 4GB GDDR5) using `ncu` and `nsys`:
- **Vector Addition Memory Squeeze:** Saturated physical DRAM bandwidth at **146.1 GB/s**.
- **2D Block-Tiled & L2-Swizzled FP16 GEMM (`2048x2048`):**
  - **PyTorch cuBLAS Baseline:** `54.35 ms` (`0.32 TFLOPS`)
  - **Custom Triton Tiled GEMM:** `28.89 ms` (`0.59 TFLOPS`) — **1.88x Speedup over cuBLAS**
  - **Custom Triton L2-Swizzled GEMM:** `31.78 ms` (`0.54 TFLOPS`) — **1.71x Speedup over cuBLAS**
- **Nsight Compute (`ncu`) Diagnosis:** Achieved **76.17% Memory Speed-of-Light** and **45.36% L2 Cache Throughput**.

---

## 🎯 Active Engineering Sprint & Target Architecture (In Progress)

Executing an engineering sprint targeting modern LLM inference bottlenecks:
- **CUDA C++20 & CuTe Foundations:** Zero-copy strided tensor layouts, 64-byte aligned memory arenas, and register-tiled GEMM kernels.
- **MLA Absorption Engine & FP4 Quantization:** Fusing Query projections into compressed Key/Value latent spaces and routing activation outliers into FP16 bypass paths with **0 Shared Memory Bank Conflicts**.

---

## 🔗 Connect & Collaborate

- **GitHub:** [@kaintsahib](https://github.com/kaintsahib)
- **LinkedIn:** [gaurav-sahib-635136339]
- **X (Twitter):** [https://x.com/KaintSahib_7866]
