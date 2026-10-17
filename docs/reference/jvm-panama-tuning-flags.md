# Reference: JVM & Project Panama Tuning Flags

Because JNum executes high-throughput SIMD operations and manages off-heap memory, understanding JVM runtime arguments and JNum system properties allows you to extract maximum performance from your target hardware.

---

## 1. Mandatory Runtime & Compiler Flags

These two flags must be passed whenever compiling or executing applications using JNum:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -jar your-application.jar
```

| Flag | Category | Purpose |
| :--- | :--- | :--- |
| `--add-modules jdk.incubator.vector` | Module Resolution | Loads the incubator SIMD vector module into the JVM runtime image. |
| `--enable-native-access=ALL-UNNAMED` | FFM Permissions | Authorizes the default classloader to allocate and bind off-heap `MemorySegment` buffers without terminal security warnings. |

---

## 2. JNum System Properties (Hardware Overrides)

Upon initialization, JNum automatically detects CPU cache hierarchy (L1D, L2, L3) and computes optimal BLIS tiling parameters. You can manually override these values using standard Java system properties (`-Dkey=value`):

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -Djnum.cache.l1=49152 \
     -Djnum.cache.l2=1048576 \
     -Djnum.cache.l3=33554432 \
     -Djnum.matmul.tasksPerCore=2 \
     -jar your-application.jar
```

### Supported Properties

| Property | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`jnum.cache.l1`** | `long` (bytes) | Auto-detected | Overrides detected L1 Data cache capacity (e.g. `32768` for 32 KB). |
| **`jnum.cache.l2`** | `long` (bytes) | Auto-detected | Overrides detected L2 cache capacity (e.g. `524288` for 512 KB). |
| **`jnum.cache.l3`** | `long` (bytes) | Auto-detected | Overrides detected L3 cache capacity (e.g. `16777216` for 16 MB). |
| **`jnum.cache.kc`** | `int` | Derived | Overrides micro-panel inner depth $K_C$ (must be a multiple of 64). |
| **`jnum.cache.mc`** | `int` | Derived | Overrides row block height $M_C$ (must be a multiple of $\text{LCM}(M_R, 8) = 24$). |
| **`jnum.cache.nc`** | `int` | Derived | Overrides column block width $N_C$ (must be a multiple of $\text{LCM}(N_R, 16)$). |
| **`jnum.matmul.tasksPerCore`** | `int` | `1` | Controls parallel ForkJoin sub-task granularity per logical core. Increase to `2` on NUMA or hyperthreaded systems. |

---

## 3. HotSpot C2 Vector JIT Diagnostics

To verify that the JVM HotSpot C2 compiler is generating native hardware vector instructions (and not falling back to scalar emulation):

### A. Inspecting Native Assembly with `hsdis`

Install the HotSpot Disassembler plugin (`hsdis-amd64.so` or `hsdis-aarch64.dylib`) and launch your application with compiler diagnostics:

```bash
java -XX:+UnlockDiagnosticVMOptions \
     -XX:+PrintAssembly \
     -XX:CompileCommand=print,jnum/internal/kernel/linalg/matmul/MatMulMicroKernels.* \
     --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     YourBenchmark
```

**What to look for in disassembly output**:
* On **AVX-512**: Look for `vfmadd231ps zmm...`, `vmovups zmm...` (512-bit vector operations).
* On **AVX2**: Look for `vfmadd231ps ymm...`, `vmovups ymm...` (256-bit vector operations).
* On **Apple Silicon / ARM64**: Look for `fmla v...`, `ldr q...` (128-bit NEON vector operations).

### B. Forcing 512-Bit Vector Registers on Intel/AMD

On some Linux kernels and older JDK builds, the JVM defaults to 256-bit vector widths to avoid AVX-512 thermal throttling penalties. On modern Intel Sapphire Rapids, Emerald Rapids, or AMD Zen 4/5 architectures, you can explicitly enable 512-bit registers:

```bash
java -XX:+UnlockDiagnosticVMOptions \
     -XX:MaxVectorSize=64 \
     --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -jar your-application.jar
```

`-XX:MaxVectorSize=64` sets the maximum vector register size to 64 bytes (512 bits).

---

## 4. Off-Heap Memory & Garbage Collection Tuning

Because `NDArray` data resides in native memory outside the JVM garbage-collected heap:

1. **Native Memory Limit (`MaxDirectMemorySize`)**:
   By default, `Arena.allocate()` native memory is governed by `-XX:MaxDirectMemorySize`. If your application allocates massive multi-gigabyte matrices, ensure this parameter is sized appropriately:
   ```bash
   -XX:MaxDirectMemorySize=64G
   ```
2. **Small JVM Heap Overhead**:
   Because tensor contents live off-heap, an `NDArray` instance itself is an ultra-lightweight 48-byte Java object header holding pointer and stride metadata. A 64 GB dataset can easily run on a JVM configured with just `-Xmx2G` of heap space.
3. **Virtual Threads (Project Loom)**:
   JNum's computation kernels are CPU-bound and run on a dedicated worker pool (`Constants.MATMUL_POOL`). When calling JNum from Virtual Threads (`Thread.ofVirtual()`), compute tasks execute smoothly without pinning the underlying carrier thread.
