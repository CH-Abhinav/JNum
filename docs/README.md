# JNum Documentation

Welcome to the official documentation for **JNum**, a high-performance numerical computing library for modern Java (JDK 25+) and Kotlin, powered by **Project Panama** (Foreign Function & Memory API and the Vector API).

JNum brings the ergonomics of NumPy and PyTorch to the JVM without native C/JNI build dependencies, off-heap fragmentation, or garbage collection overhead.

---

## Documentation Structure

The documentation is organized into three distinct sections:

### 1. [The User Guide](guide/01-getting-started.md)
A progressive, pedagogical walk-through of the library, structured to teach mental models, idioms, and zero-allocation patterns before scaling into complex numerical computing:

1. [Getting Started](guide/01-getting-started.md) — JDK 25 prerequisites, Panama JVM flags, Maven & Gradle setup, single-file scripts.
2. [Tensors & Fundamentals](guide/02-tensors-and-fundamentals.md) — Off-heap mental model, data types (`DType`), allocation lifecycles (`Arena`), and array factories.
3. [Layouts, Strides, and Slicing](guide/03-layouts-strides-and-slicing.md) — Row-major strides, zero-copy views vs. copies, subviews, Pythonic `Slice` tokens, and string indexing.
4. [SIMD Arithmetic & Broadcasting](guide/04-simd-arithmetic-and-broadcasting.md) — Vector lane operations, NumPy-compatible broadcasting, in-place mutations (`addi`), and pre-allocated buffers.
5. [Comparisons & Boolean Logic](guide/05-comparisons-and-boolean-logic.md) — Clamping (`maximum`, `minimum`), boolean masks, truthiness reductions (`all`, `any`), and bitwise logic.
6. [Transcendental & Unary Operations](guide/06-transcendental-and-unary-ops.md) — Vectorized element-wise math (`exp`, `log`, `sqrt`, `sigmoid`) and circular/hyperbolic trigonometry.
7. [Linear Algebra & Decompositions](guide/07-linear-algebra-and-decompositions.md) — 4-tier GEMM (`matmul`), dot products, matrix properties (`det`, `norm`, `cond`), solvers ($Ax=b$), and factorizations (QR, SVD, Cholesky, Eig).
8. [Descriptive & Inferential Statistics](guide/08-descriptive-and-inferential-stats.md) — The `Stats` facade, moments, 5 quantile algorithms, covariance matrices, cumulative reductions, and discrete differences.
9. [Symbolic Expression Engine](guide/09-symbolic-expression-engine.md) — Eliminating intermediate temporary allocations with `JNum.eval()` via positional `$0` and named variable expressions.
10. [Tabular & NumPy I/O](guide/10-tabular-and-numpy-io.md) — Delimited text (CSV/TSV) parsing with `CsvOptions`, `.npy` files with zero-copy memory mapping (`mmap`), and `.npz` archives.
11. [Safetensors & Asynchronous I/O](guide/11-safetensors-and-async-io.md) — Hugging Face checkpoint inspection, deterministic memory-mapped weights (`MmapTensors`), and non-blocking `CompletableFuture` pipelines.
12. [Neural Network Inference](guide/12-neural-network-inference.md) — The `jnum.nn` framework: `Module`, `Linear` with rank preservation, non-linear activations, and `Sequential` inference containers.
13. [Idiomatic Kotlin DSL](guide/13-idiomatic-kotlin-dsl.md) — First-class Kotlin extensions: operator overloading (`+`, `*`), infix math (`matmul`, `dot`), and multidimensional range slicing (`arr[all, 0..4]`).
14. [Safety & Error Diagnostics](guide/14-safety-and-error-diagnostics.md) — Memory safety guarantees, understanding `IllegalStateException` on closed arenas, broadcasting mismatches, and `NaN`/`Inf` debugging.

---

### 2. [Architecture & Systems Internals](internals/01-memory-strides-and-ffm.md)
An advanced engineering deep-dive for systems programmers, compiler enthusiasts, and JVM performance engineers:

1. [Off-Heap Memory, Strides, and FFM Alignment](internals/01-memory-strides-and-ffm.md) — Native `MemorySegment` mechanics, mathematical derivation of row-major strides, and 64-byte hardware cache-line alignment.
2. [Pure Java GEMM & SIMD Micro-Kernels](internals/02-pure-java-gemm-and-simd.md) — The 4-tier dispatch matrix, Panama Vector API intrinsics, BLIS-style cache hierarchy, and register-blocked micro-panels ($M_R \times N_R = 6 \times 2 \cdot VL$).
3. [Dynamic Hardware & Analytical Cache Probing](internals/03-dynamic-hardware-and-cache-probing.md) — Runtime cache probing across Linux (`sysfs`), macOS (`sysctl`), and Windows (`WMI`/`CIM`), with mathematical tile derivation for $K_C$, $M_C$, and $N_C$.
4. [Zero-Copy Virtual Memory & Custom Byte Parsers](internals/04-zero-copy-io-architecture.md) — OS virtual memory paging, safe deterministic unmapping via `Arena`, zero-heap JSON header parsing, and fast SIMD number parsing.
5. [AST Compilation & Buffer Recycling](internals/05-eval-engine-ast-compilation.md) — Regex-free lexical scanning, Shunting-Yard AST generation, topological evaluation, and temporary buffer recycling.

---

### 3. [Reference](reference/numpy-to-jnum-rosetta-stone.md)
Quick-lookup references and cheatsheets:

* [NumPy to JNum Rosetta Stone](reference/numpy-to-jnum-rosetta-stone.md) — Side-by-side syntax comparison across NumPy (Python), JNum (Java 25), and JNum (Kotlin).
* [JVM & Project Panama Tuning Flags](reference/jvm-panama-tuning-flags.md) — Essential JVM flags, hardware capability validation, disassembly inspection with `-XX:+PrintAssembly`, and overriding tile heuristics via system properties.
