---
title: "Work To Note"
date: 2025-12-26
area: "AI Architecture"
summary: "A map of the hardware, software, and co-design topics behind modern AI systems."
---

## Hardware: Architecture & Systems

### 1. Compute Microarchitecture (The Core)

- Matrix Multiplication Units (High): Systolic Arrays (Output vs. Weight Stationary), Tensor Cores, MXU design, Sparsity support (2:4 structured).

- Vector/SIMD Units: Element-wise operations, activation function hardware (GELU, SwiGLU implementations), special function units (SFU).

- Data Types & Precision (High): FP8 (E4M3/E5M2), BF16 vs. FP16, INT8, Block Floating Point (MX formats), Numerics and rounding modes.

- Instruction Set Architecture (ISA): RISC-V extensions for AI, GPU PTX, custom VLIW for accelerators.

### 2. Memory Systems (The Bottleneck)

- On-Chip Memory (High): Scratchpads vs. Caches, Global Buffers, Unified Buffer design, Bank conflicts, Read/Write ports.

- Off-Chip Memory (High): HBM (HBM3/3E/4), GDDR, LPDDR, Memory Controllers, PHY constraints, ECC & Reliability.

- Memory Hierarchy: Cache coherence protocols (if scale-up), Prefetching strategies, Compression (Frame buffer compression).

### 3. Interconnects & Network (The Fabric)

- On-Chip (NoC) (High): Mesh, Torus, Ring, Crossbar topologies, Virtual channels, Deadlock avoidance, Latency vs. Throughput trade-offs.

- Chip-to-Chip (Scale-Up) (High): NVLink, Infinity Fabric, UCIe (Universal Chiplet Interconnect Express), D2D (Die-to-Die) PHYs.

- System Scale-Out: Ethernet vs. InfiniBand, RoCE v2, NIC microarchitecture (SmartNICs), RDMA, Rail-only connectivity.

### 4. VLSI & Physical Design (The Implementation)

- Advanced Packaging (High): CoWoS (Chip-on-Wafer-on-Substrate), 2.5D vs. 3D Stacking (SoIC), Chiplet partitioning, Thermal dissipation/TDP.

- Digital Flow: Clock tree synthesis, Power gating, Dynamic Voltage and Frequency Scaling (DVFS), Critical path analysis.

- Technology Nodes: FinFET vs. GAA (Gate-All-Around), Transistor scaling limits, SRAM scaling issues.

### 5. MTIA specs and features

## II. Software: Workloads & Algorithms

### 1. Transformer Architectures (The Dominant Workload)

- Attention Mechanisms (High): MHA (Multi-Head), MQA (Multi-Query), GQA (Grouped-Query), MLA (Multi-Head Latent Attention - DeepSeek), Linear Attention.

- Positional Embeddings: RoPE (Rotary Positional Embeddings), ALiBi.

- Feed Forward Networks: SwiGLU, MoE (Mixture of Experts) and routing logic.

- Long Context: Ring Attention, Sequence parallelism nuances.

### 2. Efficiency Algorithms (Hardware-Aware)

- Quantization (High): PTQ (Post-Training Quantization), QAT (Quantization-Aware Training), KV Cache quantization, Weight-only vs. Activation quantization.

- Sparsity: Unstructured vs. Semi-structured (N:M), Activation sparsity (ReLU/GELU zeros).

- Speculative Decoding: Draft models, Verification logic (compute bound vs. memory bound shift).

### 3. Emerging & Specialized Models

- Recurrent/State Space Models: Mamba (SSM), RNN derivatives (parallelizability challenges).

- Generative Media: Diffusion Models (UNet, DiT), Multimodal inputs (Image+Text).

- Training Dynamics: Backpropagation compute graphs, Optimizer states (AdamW), Checkpointing.

## III. Interface: Stack, Mapping & Co-Design

### 1. Compilers & Intermediate Representations

- Graph Compilation (High): Operator Fusion (Horizontal/Vertical), XLA (Accelerated Linear Algebra), MLIR (Multi-Level IR), HLO, Graph rewrites.

- Loop Optimization: Tiling, Re-materialization, Double buffering, Polyhedral compilation.

- Auto-Tuning: Search spaces for kernel parameters (Triton, TVM).

### 2. Kernels & Programming Models

- Kernel Optimization (High): FlashAttention (V2/V3 - tiling & recomputation logic), GEMM optimization, Reduction kernels.

- Programming Frameworks: **CUDA**, Triton (Python-based GPU programming), CUTLASS.

### 3. Parallelism & Distribution

- Parallelism Strategies (High): Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (FSDP/DDP), Expert Parallelism (EP).

- Communication Primitives: All-Reduce, All-Gather, Reduce-Scatter, P2P, Barrier synchronization.

- Communication Libraries: NCCL, RCCL, OneCCL.

### 4. Performance Modeling & Analysis

- Analytical Modeling (High): Roofline Model (Arithmetic Intensity), LogP model for network, Memory bandwidth utilization.

- Simulation: Cycle-accurate vs. Trace-driven simulation, Workload characterization (nsight systems, nsight compute).
