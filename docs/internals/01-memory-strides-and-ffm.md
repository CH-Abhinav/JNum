# Systems Internals: Off-Heap Memory & Stride Calculus

In traditional JVM programming, numerical performance is severely limited by object layouts and garbage collection overhead. This document details the low-level systems architecture of JNum's memory subsystem: how Java's Foreign Function & Memory (FFM) API is leveraged, the mathematical proofs behind stride transformations, and why 64-byte hardware cache alignment is enforced across all allocations.

---

## 1. Foreign Function & Memory (FFM) vs. `sun.misc.Unsafe`

Historically, high-performance Java libraries had only two options for off-heap native memory:
1. **Direct ByteBuffers (`java.nio.ByteBuffer`)**: Limited to $2 \text{ GB}$ ($2^{31} - 1$ bytes) due to 32-bit integer indexing, carrying significant bounds-checking call overhead.
2. **`sun.misc.Unsafe`**: Allowed raw pointer arithmetic and unlimited address spaces, but bypassed all JVM memory safety. A dangling pointer or double-free resulted in unpredictable memory corruption or an immediate, uncatchable `SIGSEGV` crash. Furthermore, `Unsafe` is being actively deprecated and restricted in modern JDKs.

### The FFM Architecture (JEP 454)

JNum is built entirely upon standard JDK 25 FFM abstractions:

```
┌────────────────────────────────────────────────────────┐
│                   jnum.NDArray                         │
│   - shape: long[]          - strides: long[]           │
│   - dtype: DType           - size: long                │
└───────────────────────────┬────────────────────────────┘
                            │ holds reference
                            ▼
┌────────────────────────────────────────────────────────┐
│             java.lang.foreign.MemorySegment            │
│   - Base Native Address (64-bit raw pointer)           │
│   - Byte Size: long (Supports Exabyte address spaces)  │
│   - Scope / Arena (Governs temporal memory safety)     │
└───────────────────────────┬────────────────────────────┘
                            │ allocated from
                            ▼
┌────────────────────────────────────────────────────────┐
│               java.lang.foreign.Arena                  │
│   - ofAuto(): GC tracked native lifetime               │
│   - ofConfined(): Thread-bound, zero-GC instant drop   │
│   - ofShared(): Multi-threaded safe deterministic drop │
└────────────────────────────────────────────────────────┘
```

FFM provides:
* **64-bit Long Indexing**: Tensors can span tens or hundreds of gigabytes, far exceeding the 2 GB limitation of `ByteBuffer`.
* **Zero JNI Call Overhead**: Panama native calls and memory reads are intrinsified directly by the HotSpot C2 compiler into direct x86/ARM memory move assembly instructions (`movq`, `vmovups`).
* **Temporal Safety Without GC**: An `Arena` guarantees that a segment cannot be accessed after its governing scope is closed, throwing a safe `IllegalStateException` rather than causing a memory corruption vulnerability.

---

## 2. Row-Major Stride Calculus

An `NDArray` maps a continuous linear byte array in physical RAM to an $N$-dimensional coordinate space.

### Canonical Stride Formulation

Let a tensor have dimensions given by the shape tuple:

$$\mathbf{d} = (d_0, d_1, d_2, \dots, d_{D-1})$$

The canonical row-major stride vector $\mathbf{s} = (s_0, s_1, s_2, \dots, s_{D-1})$ specifies how many physical elements must be skipped in memory to advance by $1$ unit along dimension $k$:

$$s_k = \prod_{m=k+1}^{D-1} d_m, \quad \text{for } 0 \le k < D-1$$

$$s_{D-1} = 1$$

### Proof of Contiguity

A tensor is contiguous if and only if every element in the coordinate space maps to a unique, monotonically increasing linear index in $[0, \text{size} - 1]$ without gaps.

For any coordinate $\mathbf{i} = (i_0, i_1, \dots, i_{D-1})$ with $0 \le i_k < d_k$, the linear element offset is given by:

$$\text{offset}(\mathbf{i}) = \sum_{k=0}^{D-1} i_k s_k$$

Because each $s_k = d_{k+1} s_{k+1}$, the summation forms a mixed-radix positional numeral system. The maximum possible offset is:

$$\text{offset}(\mathbf{d} - \mathbf{1}) = \sum_{k=0}^{D-1} (d_k - 1) s_k = \prod_{k=0}^{D-1} d_k - 1 = \text{size} - 1$$

This proves that the mapping is a bijection between $\prod_{k=0}^{D-1} [0, d_k - 1]$ and $[0, \text{size} - 1]$, guaranteeing that contiguous operations can process the memory as a flat 1D stream.

---

## 3. Mathematical Proof of Zero-Copy Transposition & Slicing

### Transposition

Given a 2D matrix with shape $(d_0, d_1)$ and strides $(s_0, s_1) = (d_1, 1)$, a transposed matrix coordinates swap $(i, j) \rightarrow (j, i)$.

The target offset in the original physical buffer is:

$$\text{offset}(j, i) = j \cdot s_0 + i \cdot s_1 = i \cdot s_1 + j \cdot s_0$$

Hence, defining the new transposed shape as $(d_1, d_0)$ and transposed strides as $(s_1, s_0) = (1, d_1)$ perfectly maps the physical memory without moving a single byte. Transposition is an **$O(1)$ constant-time metadata reassignment**.

### Slicing with Strides

When slicing dimension $k$ with range $[ \text{start}_k, \text{stop}_k, \text{step}_k ]$:
1. The new base pointer is shifted by:
   $$\Delta \text{Base} = \text{start}_k \cdot s_k \times \text{elementBytes}$$
2. The new dimension length is:
   $$d'_k = \left\lceil \frac{\text{stop}_k - \text{start}_k}{\text{step}_k} \right\rceil$$
3. The new stride is scaled by the step:
   $$s'_k = s_k \cdot \text{step}_k$$

If $\text{step}_k$ is negative, $s'_k$ becomes negative, and reading sequentially along dimension $k$ traverses backward in memory. This enables **zero-copy array reversal**.

---

## 4. 64-Byte Hardware Cache-Line Alignment

In [`Allocator.java`](../../core/src/main/java/jnum/internal/memory/Allocator.java#L45), every tensor memory segment is allocated with a strict 64-byte alignment requirement:

```java
// Allocates native memory aligned to 64 bytes
MemorySegment segment = arena.allocate(byteSize, 64L);
```

### Why 64-Byte Alignment is Critical

Modern x86 (Intel/AMD) and ARM64 (Apple Silicon, Neoverse) architectures organize their cache hierarchy into **64-byte cache lines**:

```
Cache Line 0 (64 Bytes)          Cache Line 1 (64 Bytes)
[ Byte 0 ................. Byte 63 ] [ Byte 64 ................ Byte 127 ]
```

When memory is unaligned:
1. **Cache-Line Split Penalty**: An unaligned 512-bit vector read (64 bytes for AVX-512) that straddles the boundary between Byte 60 and Byte 124 forces the CPU L1 data cache controller to issue **two separate memory bus transactions** instead of one.
2. **Pipeline Replay Traps**: Modern CPUs maintain load-store execution queues. Splitting a vector load across cache lines can cause a pipeline stall or replay penalty of 10 to 20 clock cycles.
3. **Unaligned Load Elimination**: When native memory is statically guaranteed to be 64-byte aligned, the C2 JIT compiler emits aligned vector load instructions (`vmovaps`) rather than unaligned loads (`vmovups`), enabling the CPU's internal store-forwarding optimizations to run at peak throughput.
