# Systems Internals: Pure Java GEMM & SIMD Micro-Kernels

Matrix multiplication (GEMM) is the computational backbone of scientific and deep learning workloads. Achieving peak theoretical CPU floating-point operations per second (FLOPS) requires careful orchestration of CPU instruction pipelines, vector registers, and cache hierarchies.

This document details the systems engineering of JNum's **pure-Java 4-Tier GEMM Engine** ([`MatMul.java`](../../core/src/main/java/jnum/internal/kernel/linalg/MatMul.java)): how work is classified, how the BLIS (Basic Linear Algebra Instruction Set) cache model is implemented, and how vector registers are scheduled to prevent spills.

---

## 1. The 4-Tier Dispatch Matrix

A common flaw in numerical libraries is using a single "one-size-fits-all" matrix multiplication kernel. In practice:
* Tiny matrices (e.g. $3 \times 3$ affine transformations in graphics) suffer severe performance degradation if forced through thread pool synchronization and memory packing.
* Massive matrices (e.g. $4096 \times 4096$) suffer catastrophic cache thrashing if computed with naive direct loops.

JNum's top-level dispatcher ([`MatMul.java`](../../core/src/main/java/jnum/internal/kernel/linalg/MatMul.java)) evaluates both the dimensions $(N, M, P)$ and total arithmetic work volume ($\text{FLOPs} = 2 \cdot N \cdot M \cdot P$) to route calls through 4 specialized execution tiers:

```
                            Matrix Dimensions (N, M, P) & Work Volume
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   maxDim <= 4 ?       │──YES──> Tier 0: Nano Scalar Kernel
                                    └───────────┬───────────┘
                                                │ NO
                                                ▼
                                    ┌───────────────────────┐
                                    │ maxDim <= 128 and     │──YES──> Tier 1: Direct Register-Blocked
                                    │ ops <= 128^3 ?        │         SIMD Kernel
                                    └───────────┬───────────┘
                                                │ NO
                                                ▼
                                    ┌───────────────────────┐
                                    │ Single Core, N < MR,  │──YES──> Tier 2: Single-Threaded BLIS
                                    │ ops < ParallelThresh, │         Cache-Blocked Kernel
                                    │ or maxDim <= 256 ?    │
                                    └───────────┬───────────┘
                                                │ NO
                                                ▼
                                    Tier 3: Parallel BLIS Macro-Kernel (ForkJoin)
```

### Tier Characteristics

| Tier | Class | Typical Matrix Size | Primary Optimization Strategy |
| :--- | :--- | :--- | :--- |
| **Tier 0** | `MatMulTier0Nano` | $N, M, P \le 4$ | Unrolled pure scalar registers. Zero SIMD setup overhead, zero packing. |
| **Tier 1** | `MatMulTier1DirectTiled` | $4 < \text{maxDim} \le 128$ | Direct 2.5D register-tiled SIMD. Bypasses packing because inputs fit natively inside per-core L1D/L2 cache. |
| **Tier 2** | `MatMulTier2BlisSingle` | $128 < \text{maxDim} \le 256$ | Full 5-loop BLIS cache hierarchy with packed micro-panels. Single-threaded to avoid ForkJoin synchronization overhead. |
| **Tier 3** | `MatMulTier3BlisParallel`| $\text{maxDim} > 256$ | Multi-threaded parallel GEBP (General Block-Panel) using `Constants.MATMUL_POOL`. Shared packed $B$ panel in L3 cache. |

---

## 2. The BLIS Cache-Blocking Hierarchy

For large matrices, the limiting factor in GEMM is memory bus bandwidth. If matrix entries must be continuously fetched from main RAM, the CPU ALUs sit idle waiting for cache fills.

JNum adopts the **BLIS (Basic Linear Algebra Instruction Set)** 5-loop cache-blocking framework:

```
Full Matrix A [N x M]                Full Matrix B [M x P]
 ┌─────────────────┐                  ┌─────────────────┐
 │                 │                  │   Panel B_p     │  <-- Packed into Shared L3 (KC x NC)
 │   Block A_i     │                  │   [KC x NC]     │
 │   [MC x KC]     │                  └─────────────────┘
 │                 │                           │
 └─────────────────┘                           │
          │                                    │
          ▼                                    ▼
   Packed into L2                        Streamed into L1D
    Thread-Local                           Micro-Panels
   [MC x KC panel]                         [KC x NR]
          │                                    │
          └─────────────────┬──────────────────┘
                            ▼
                   Register Micro-Kernel
                     (MR x NR = 6 x 2*VL)
                   Held in SIMD Registers
```

The loops are structured to maximize cache residence at every tier:
1. **Loop 5 ($J_C$ Loop)**: Steps through $B$ in blocks of $N_C$ columns.
2. **Loop 4 ($P_C$ Loop)**: Steps through inner dimension $K$ in blocks of $K_C$.
3. **Pack $B$**: Packs a $K_C \times N_C$ slice of $B$ into sequential memory, resident in the **shared L3 cache**.
4. **Loop 3 ($I_C$ Loop)**: Steps through $A$ in blocks of $M_C$ rows.
5. **Pack $A$**: Packs an $M_C \times K_C$ slice of $A$ into thread-local memory, resident in **per-core L2 cache**.
6. **Loop 2 & 1 (Micro-Kernel Loops)**: Streams $M_R \times K_C$ and $K_C \times N_R$ panels directly into vector registers for accumulation.

---

## 3. Micro-Panel Packing (`MatMulPacking.java`)

When matrices are stored in standard row-major layout, iterating along a column of $B$ steps across rows with stride $P$. For large $P$, each element load hits a different virtual memory page, causing massive **Translation Lookaside Buffer (TLB) misses** and cache thrashing.

JNum's packing kernel ([`MatMulPacking.java`](../../core/src/main/java/jnum/internal/kernel/linalg/matmul/MatMulPacking.java)) re-organizes $A$ and $B$ into dense, sequential micro-panels:

* **Packed Panel $A$**: Stored as continuous vertical strips of height $M_R$:
  $$\text{Panel } A = \text{Array of } [M_R \times K_C] \text{ contiguous blocks}$$
* **Packed Panel $B$**: Stored as continuous horizontal strips of width $N_R$:
  $$\text{Panel } B = \text{Array of } [K_C \times N_R] \text{ contiguous blocks}$$

Once packed, memory accesses inside the innermost micro-kernel are strictly **stride-1 sequential vector loads**, enabling CPU hardware prefetchers to stream data at maximum memory bus saturation.

---

## 4. The SIMD Register Micro-Kernel (`MatMulMicroKernels.java`)

The innermost micro-kernel evaluates:

$$C_{\text{micro}}[M_R \times N_R] += A_{\text{micro}}[M_R \times K_C] \times B_{\text{micro}}[K_C \times N_R]$$

To achieve peak CPU floating-point throughput, the micro-kernel must keep accumulators entirely in CPU vector registers without spilling to the stack.

### Vector Register Budget Allocation

On x86-64:
* **AVX2**: Provides 16 256-bit registers (`ymm0` to `ymm15`). Vector length $VL = 8$ floats.
* **AVX-512**: Provides 32 512-bit registers (`zmm0` to `zmm31`). Vector length $VL = 16$ floats.

JNum sizes its micro-kernel registers as:

$$M_R = 6, \quad N_R = 2 \cdot VL$$

$$N_R = 16 \text{ floats on AVX2}, \quad N_R = 32 \text{ floats on AVX-512}$$

This requires an accumulation matrix of $6 \times 2 = \mathbf{12 \text{ SIMD vector registers}}$:

```
Accumulator Grid (12 Vector Registers):
Row 0: [ acc00 (VL floats) ] [ acc01 (VL floats) ]
Row 1: [ acc10 (VL floats) ] [ acc11 (VL floats) ]
Row 2: [ acc20 (VL floats) ] [ acc21 (VL floats) ]
Row 3: [ acc30 (VL floats) ] [ acc31 (VL floats) ]
Row 4: [ acc40 (VL floats) ] [ acc41 (VL floats) ]
Row 5: [ acc50 (VL floats) ] [ acc51 (VL floats) ]
```

### Remaining Register Budget:
* **2 registers** to hold two broadcast segments of $B$: `bVec0`, `bVec1`.
* **1 register** to broadcast scalar elements from $A$: `aBroadcast`.
* **Total registers required**: $12 + 2 + 1 = \mathbf{15 \text{ vector registers}}$.

Because $15 \le 16$, the micro-kernel fits **100% inside hardware physical registers** on AVX2, and leaves 17 free registers on AVX-512. There is **zero stack spilling**.

### Fused Multiply-Add (FMA) Pipeline Saturation

Inside the $K_C$ accumulation loop, JNum executes hardware Fused Multiply-Add instructions:

$$\text{acc}_{i, j} = \text{fma}(a_i, b_j, \text{acc}_{i, j})$$

Modern x86 cores contain 2 parallel FMA execution ports, each capable of retiring one FMA instruction every cycle with a latency of 4 cycles. By unrolling 12 independent accumulator updates across $M_R = 6$, JNum perfectly fills the CPU's out-of-order execution pipeline, preventing execution stalls and achieving near-native computational throughput.
