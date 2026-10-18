# JNum

[![Java 25](https://img.shields.io/badge/Java-25%2B-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![Project Panama](https://img.shields.io/badge/Project-Panama-007396?logo=openjdk&logoColor=white)](https://openjdk.org/projects/panama/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Jar Size](https://img.shields.io/badge/Jar%20Size-%3C%20150%20KB-success)](#the-pure-java-advantage)
[![Documentation](https://img.shields.io/badge/Docs-User%20Guide-blueviolet)](docs/)

A high-performance, hardware-accelerated numerical computing library for modern Java (JDK 25+) and Kotlin, powered by **Project Panama** (Foreign Function & Memory API and the Vector API).

JNum brings the ergonomics of **NumPy** and **PyTorch** to the JVM with native BLAS performance—**without native C++ build toolchains, JNI boundaries, or garbage collection overhead.**

---

## Why JNum? (The Pure Java Advantage)

Traditional JVM numerical and tensor libraries (such as ND4J or Deeplearning4j) rely on native C++ backends. They manipulate memory using deprecated `sun.misc.Unsafe` and cross the native boundary via **JNI (Java Native Interface)**.

| Metric | Legacy Java Tensor Libraries | **JNum** |
| :--- | :--- | :--- |
| **Backend Implementation** | C++ (OpenBLAS, OneDNN, LibTorch) bridged via JNI | **100% Pure Java 25 & Kotlin** |
| **Native Library Bloat** | 100+ MB (Bundles `.dll`, `.so`, `.dylib` binaries) | **< 150 KB total JAR footprint** |
| **Memory Architecture** | Deprecated `sun.misc.Unsafe` (Memory corruption risk) | **Foreign Function & Memory (JEP 454)** |
| **SIMD Acceleration** | Opaque native compiler intrinsics | **JVM Vector API (`jdk.incubator.vector`)** |
| **Memory Safety** | Out-of-bounds causes native process crash (`SIGSEGV`) | **100% JVM Memory Safe (Catchable Exceptions)** |
| **Deployment Complexity**| Fragile native dependencies, dynamic linker errors | **Drop-in JAR: Runs anywhere Java runs** |

---

## 30-Second Code Tour

### Modern Java 25

```java
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.Slice;

void main() {
    // 1. Allocate 64-byte aligned off-heap tensors
    NDArray a = JNum.rand(1024, 512);
    NDArray b = JNum.rand(512, 256);

    // 2. Hardware SIMD matrix multiplication (4-tier BLIS engine)
    NDArray c = a.matmul(b); // Shape: [1024, 256]

    // 3. Zero-copy subviews and Pythonic slicing
    NDArray slice = c.slice("0:10, 0:5");

    // 4. In-place zero-allocation mutations
    slice.muli(2.0f);

    IO.println("Top-left block:\n" + slice);
}
```

### Idiomatic Kotlin DSL

```kotlin
import jnum.*

fun main() {
    val x = JNum.rand(128, 64)
    val w = JNum.rand(64, 10)
    val b = JNum.zeros(10)

    // Natural operator overloading and infix linear algebra
    val logits = (x matmul w) + b

    // Python-style multidimensional range slicing
    val topScores = logits[0 until 5, all]

    println("Computed logits shape: ${logits.shapeString()}")
}
```

---

## Feature Highlights

* **Hardware SIMD Vector Lanes**: Automatically probes your CPU to schedule AVX-512 (16 lanes), AVX2 (8 lanes), or ARM NEON (4 lanes) instructions via the Vector API.
* **4-Tier GEMM Engine**: Hardware-adaptive matrix multiplication routing across Nano scalar, Direct register-tiled, Single-threaded BLIS, and Parallel ForkJoin macro-kernels.
* **Zero-Copy Layouts & Slicing**: Transposition, reshaping, subviews, and strided slices are constant-time $O(1)$ operations with zero memory copying.
* **Symbolic Expression Evaluation**: Compile formulas like `JNum.eval("$0 * $1 + $2", a, b, c)` or `JNum.eval("x * w + b", map)` into hardware FMA instructions with automatic buffer recycling.
* **Modern AI Checkpoint I/O**: Direct zero-copy memory mapping for **Hugging Face Safetensors** (`.safetensors`) and **NumPy binaries** (`.npy`, `.npz`), plus high-throughput CSV/TSV parsing.
* **Pure-Java Neural Network Inference**: Built-in `jnum.nn` modules (`Linear` with sequence rank preservation, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `Sequential`).

---

## Quick Installation

Add the JNum core dependency to your project:

### Maven (`pom.xml`)

```xml
<dependency>
    <groupId>io.github.ch-abhinav</groupId>
    <artifactId>jnum</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- Optional Kotlin extensions: -->
<dependency>
    <groupId>io.github.ch-abhinav</groupId>
    <artifactId>jnum-kotlin</artifactId>
    <version>0.2.0</version>
</dependency>
```

### Gradle (`build.gradle.kts`)

```kotlin
dependencies {
    implementation("io.github.ch-abhinav:jnum:0.2.0")
    implementation("io.github.ch-abhinav:jnum-kotlin:0.2.0") // Optional
}
```

> [!IMPORTANT]
> Because JNum leverages Project Panama, builds and runtime invocations require `--add-modules jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED`.
> 
> 👉 **[Read the Full Installation & Setup Guide](docs/guide/01-getting-started.md)** for copy-paste Maven/Gradle compiler plugins and single-file runner configurations.

---

## Documentation

JNum ships with a complete, progressive documentation suite structured like *The Rust Book*, paired with an advanced systems engineering deep dive:

### [User Guide (`docs/guide/`)](docs/guide/01-getting-started.md)
1. **[Getting Started](docs/guide/01-getting-started.md)** — JDK 25 prerequisites, Panama JVM flags, Maven & Gradle setup.
2. **[Tensors & Fundamentals](docs/guide/02-tensors-and-fundamentals.md)** — Off-heap mental model, DTypes, and scoped `Arena` lifecycles.
3. **[Layouts, Strides, and Slicing](docs/guide/03-layouts-strides-and-slicing.md)** — Strides, subviews, Slice tokens, and zero-copy views vs copies.
4. **[SIMD Arithmetic & Broadcasting](docs/guide/04-simd-arithmetic-and-broadcasting.md)** — Vector lanes, broadcasting rules, and zero-GC output buffers.
5. **[Comparisons & Boolean Logic](docs/guide/05-comparisons-and-boolean-logic.md)** — Clamping (`maximum`/`minimum`), boolean masks, and truth reductions.
6. **[Transcendental & Unary Operations](docs/guide/06-transcendental-and-unary-ops.md)** — Vectorized `exp`, `log`, `sigmoid`, and circular/hyperbolic trigonometry.
7. **[Linear Algebra & Decompositions](docs/guide/07-linear-algebra-and-decompositions.md)** — 4-tier GEMM, matrix properties (`det`, `norm`), solvers, and factorizations (QR, SVD, Cholesky, Eig).
8. **[Descriptive & Inferential Statistics](docs/guide/08-descriptive-and-inferential-stats.md)** — The `Stats` facade, moments, 5 quantile algorithms, and covariance.
9. **[Symbolic Expression Engine](docs/guide/09-symbolic-expression-engine.md)** — Eliminating temporary arrays with `JNum.eval()` and AST buffer reuse.
10. **[Tabular & NumPy I/O](docs/guide/10-tabular-and-numpy-io.md)** — CSV/TSV with `CsvOptions`, `.npy` files with memory mapping (`mmap`), and `.npz` archives.
11. **[Safetensors & Asynchronous I/O](docs/guide/11-safetensors-and-async-io.md)** — Hugging Face checkpoints, scoped mmap (`MmapTensors`), and async pipelines.
12. **[Neural Network Inference](docs/guide/12-neural-network-inference.md)** — `Module`, `Linear` with sequence rank preservation, activations, and `Sequential`.
13. **[Idiomatic Kotlin DSL](docs/guide/13-idiomatic-kotlin-dsl.md)** — First-class operator overloads (`+`, `*`), infix math (`matmul`, `dot`), and range slicing.
14. **[Safety & Error Diagnostics](docs/guide/14-safety-and-error-diagnostics.md)** — Memory safety, closed arena states, shape mismatches, and `NaN`/`Inf` diagnostics.

### [Systems Engineering Deep Dive (`docs/internals/`)](docs/internals/01-memory-strides-and-ffm.md)
* **[Off-Heap Memory & Stride Calculus](docs/internals/01-memory-strides-and-ffm.md)** — Native `MemorySegment` mechanics, row-major stride proofs, and 64-byte cache alignment.
* **[Pure Java GEMM & SIMD Micro-Kernels](docs/internals/02-pure-java-gemm-and-simd.md)** — The 4-tier dispatch matrix, BLIS cache hierarchy, and register-blocked micro-panels ($M_R \times N_R = 6 \times 2 \cdot VL$).
* **[Dynamic Hardware & Cache Probing](docs/internals/03-dynamic-hardware-and-cache-probing.md)** — Probing Linux `sysfs`, macOS `sysctl`, and Windows `WMI` to derive analytical $K_C, M_C, N_C$ cache tiles.
* **[Zero-Copy I/O & Custom Parsers](docs/internals/04-zero-copy-io-architecture.md)** — Virtual memory paging, safe deterministic unmapping, and zero-heap byte parsers.
* **[AST Compilation & The Symbolic Engine](docs/internals/05-eval-engine-ast-compilation.md)** — Shunting-Yard AST generation, hardware FMA optimization, and topological buffer recycling.

### [Reference](docs/reference/numpy-to-jnum-rosetta-stone.md)
* **[NumPy to JNum Rosetta Stone](docs/reference/numpy-to-jnum-rosetta-stone.md)** — Side-by-side syntax comparison across NumPy (Python), JNum (Java 25), and JNum (Kotlin).
* **[JVM & Project Panama Tuning Flags](docs/reference/jvm-panama-tuning-flags.md)** — JVM flags, AVX-512/NEON verification, and overriding cache heuristics.

---

## Requirements

* **JDK 25 or later** (Temurin, OpenJDK, or Zulu)
* Architectures supported: **x86_64** (SSE4.2, AVX, AVX2, AVX-512) and **AArch64 / ARM64** (NEON, SVE).

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
