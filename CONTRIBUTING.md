# Contributing to JNum

Thank you for your interest in contributing to **JNum**! 

JNum is a high-performance numerical computing library for Java 25+ and Kotlin, powered by **Project Panama** (Foreign Function & Memory API and the Vector API). We welcome contributions ranging from bug fixes, documentation improvements, and new mathematical kernels to micro-architecture benchmarks.

---

## 1. Core Architectural Invariants

Before writing code, please review the non-negotiable design principles that govern JNum:

1. **100% Pure Java & Kotlin**: 
   * **Zero C/C++ or JNI**: All algorithms, BLAS routines, and parsers must be written in pure Java 25 or Kotlin. No native `.dll`, `.so`, or `.dylib` artifacts are permitted.
2. **Zero Heap Allocation in Hot Paths**:
   * Operations on tensors (`add`, `matmul`, `slice`, reductions) must never allocate temporary Java heap objects (`java.lang.String`, `Double`, `Float`, `Integer`, or intermediate arrays) inside inner computation loops.
   * Primitive values must never be autoboxed.
3. **64-Byte Hardware Alignment**:
   * All native memory allocations must be aligned to 64 bytes (`Allocator.zeros(arena, dtype, shape)`). This prevents cache-line splits and enables CPU unaligned load elimination.
4. **Strict Module Encapsulation (JPMS)**:
   * Public APIs belong exclusively in the exported packages: `jnum`, `jnum.nn`, `jnum.io`, and `jnum.stat`.
   * Low-level kernels, cache probes, and parsers must remain package-private or housed under `jnum.internal.*` and `jnum.io.internal.*`. Do not export internal packages in `module-info.java`.

---

## 2. Development Environment Setup

### Prerequisites
* **Java Development Kit (JDK) 25 or later** (such as Eclipse Temurin, Oracle OpenJDK, or Azul Zulu).
* **Apache Maven 3.9+**
* Hardware with vector extensions (AVX2, AVX-512, or ARM NEON).

Verify your environment:
```bash
java -version
mvn -version
```

### Cloning & Building

```bash
git clone https://github.com/CH-Abhinav/JNum.git
cd JNum

# Compile all modules (core and kotlin-core)
mvn clean compile
```

---

## 3. Running the Test Suite

JNum has extensive unit tests covering numerical precision, edge cases, shape broadcasting, and memory lifecycles:

```bash
# Run all unit tests across modules
mvn test
```

> [!NOTE]
> The Maven Surefire configuration automatically injects `--add-modules jdk.incubator.vector`. Benchmark suites are excluded during regular test runs to keep CI builds fast.

### Running a Specific Test

To run a single test class (e.g., `MatMulTier3BlisParallelTest`):
```bash
mvn test -pl core -Dtest=MatMulTier3BlisParallelTest
```

---

## 4. Running Performance Benchmarks (JMH)

If you modify matrix multiplication, SIMD vector kernels, or memory layouts, you **must run JMH benchmarks** to verify that your changes do not introduce performance regressions.

### Using the Benchmark Helper Scripts (Recommended)

Helper scripts are provided under `scripts/` to run benchmarks without typing long Maven commands:

**On Linux / macOS:**
```bash
chmod +x ./scripts/benchmark.sh

# Run entire Java benchmark suite:
./scripts/benchmark.sh all

# Run specific categories:
./scripts/benchmark.sh matmul 1024
./scripts/benchmark.sh arithmetic 1M
./scripts/benchmark.sh kotlin
```

**On Windows (PowerShell):**
```powershell
.\scripts\benchmark.ps1 matmul 1024
.\scripts\benchmark.ps1 kotlin
```

### Using Maven Directly

```bash
# Run Java benchmarks
mvn test-compile exec:exec -pl core

# Run Kotlin benchmarks
mvn test-compile exec:exec -pl kotlin-core
```

When submitting a performance-oriented PR, include the before-and-after JMH benchmark outputs (throughput or average time in $\text{ops/sec}$ or $\text{ns/op}$) alongside your hardware specs (CPU model and OS).

---

## 5. Coding Standards & Conventions

### Code Formatting & Modern Idioms
* Use modern Java 25 features: `var`, records, pattern matching for `switch`, and compact constructors where appropriate.
* Document public methods with clear Javadoc explaining parameter contracts, shapes, and potential exceptions (`IllegalArgumentException`, `IllegalStateException`).
* Do not remove existing comments or docstrings unrelated to your changes.

### License Headers
The build validates Apache 2.0 license headers on all Java and Kotlin source files. To apply license headers automatically:
```bash
mvn license:format
```
To verify license headers without formatting:
```bash
mvn license:check
```

---

## 6. How to Add a New Feature or Kernel

If you are contributing a new mathematical or tensor operation:

1. **Create the Kernel**: Implement the type-specialized SIMD algorithm under `jnum.internal.kernel.<category>/`.
   * Handle vector lanes with `VectorSpecies.length()`.
   * Handle unaligned tails using masked vector loops (`species.indexInRange(...)`) or scalar loop fallbacks.
2. **Expose in Ops Orchestrator**: Add broadcasting and type-promotion logic under `jnum.internal.ops/`.
3. **Expose Public Facade**:
   * Core tensor operations belong on [`jnum.NDArray`](core/src/main/java/jnum/NDArray.java) or [`jnum.JNum`](core/src/main/java/jnum/JNum.java).
   * Statistical reductions belong in [`jnum.stat.Stats`](core/src/main/java/jnum/stat/Stats.java).
   * Neural network layers belong in [`jnum.nn`](core/src/main/java/jnum/nn/).
4. **Kotlin Extensions**: Add inline operator overloads or extension functions in `kotlin-core/src/main/kotlin/jnum/`.
5. **Unit Tests**: Add tests verifying:
   * Mathematical correctness against known analytical values.
   * Proper behavior across all supported `DType`s (`f32`, `f64`, `i32`).
   * Broadcast compatibility and error handling on mismatched dimensions.
6. **Documentation**: Update the relevant chapter under `docs/guide/` and the Rosetta Stone (`docs/reference/numpy-to-jnum-rosetta-stone.md`).

---

## 7. Submitting a Pull Request (PR)

1. **Fork the Repository**: Create a feature branch off `main` (`git checkout -b feature/my-new-kernel`).
2. **Commit Guidelines**: Write clear, descriptive commit messages:
   * `feat: add vectorized log2 and log1p unary operations`
   * `fix: prevent cache line split during strided matrix transpose`
   * `perf: optimize BLIS KC micro-panel packing on Apple Silicon`
3. **Verify Locally**:
   ```bash
   mvn clean verify
   ```
   Ensure tests pass and `license:check` succeeds.
4. **Open a PR**: Describe what changed, the rationale, and link any related issues.

---

## 8. Reporting Issues

When opening an issue for a bug or numerical discrepancy, please provide:
1. **System Info**: OS version, CPU architecture (`x86_64` or `arm64`), CPU model (e.g. AMD Ryzen 9, Intel Core i9, Apple M3).
2. **JDK Version**: Output of `java -version`.
3. **Minimal Reproducible Example**: A short, self-contained single-file script demonstrating the unexpected behavior or exception stack trace.
