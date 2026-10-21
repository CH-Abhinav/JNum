# Systems Internals: Dynamic Hardware & Cache Probing

In high-performance BLAS libraries (such as ATLAS or GotoBLAS), matrix multiplication tile parameters are traditionally tuned via offline compilation scripts tailored to a single micro-architecture. 

In the Java ecosystem, applications must adhere to the *"Write Once, Run Anywhere"* contract: a single `.jar` distribution must run with peak performance across Intel Core, AMD EPYC, Apple Silicon, and AWS Graviton without recompilation.

This document explains how JNum dynamically probes CPU cache topologies at runtime and mathematically derives optimal BLIS tiling parameters ($K_C, M_C, N_C$) and parallel ForkJoin work partitions.

---

## 1. The Perils of Hardcoded Tile Parameters

In the BLIS cache-blocking model:
* **$K_C$** controls the depth of micro-panels fitting inside the L1 Data (L1D) cache.
* **$M_C$** controls the height of panel $A$ fitting inside the per-core L2 cache.
* **$N_C$** controls the width of panel $B$ fitting inside the shared L3 cache.

If these parameters are statically hardcoded:
* On an **Apple M-series CPU** ($128 \text{ KB}$ L1D cache, $4 \text{ MB}$ L2 cache), hardcoded x86 tiles ($32 \text{ KB}$ L1D, $256 \text{ KB}$ L2) severely under-utilize hardware bandwidth.
* On an **AMD Zen or Intel Core CPU** ($32 \text{ KB}$ L1D), oversized tile parameters cause continuous cache thrashing, evicting active micro-panels to main RAM mid-calculation.

JNum resolves this by dynamically probing physical cache sizes upon class loading ([`Constants.java`](../../core/src/main/java/jnum/internal/Constants.java)).

---

## 2. Multi-Platform OS Cache Probing

JNum implements native OS discovery mechanisms that avoid JNI dependencies:

```
                            Operating System Detection
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
        Linux                         macOS                       Windows
  Read /sys filesystem          Invoke sysctl utility       Query CIM / WMI via
  /sys/devices/system/cpu/      - hw.l1dcachesize           PowerShell Get-CimInstance
  cpu0/cache/index*/            - hw.l2cachesize            Win32_CacheMemory
                                - hw.l3cachesize
```

### A. Linux: Direct Virtual Filesystem Parsing

On Linux, JNum inspects `/sys/devices/system/cpu/cpu0/cache/`:
1. Scans directories `index0` through `index5`.
2. Inspects `type` and ignores instruction caches (`"Instruction"`).
3. Reads `level` (1, 2, or 3) and parses `size` (e.g. `"32K"`, `"512K"`, `"16M"`).
4. Extracts clean byte totals without spawning external shell processes.

### B. macOS: Darwin Kernel `sysctl`

On Apple Darwin (macOS x86 and ARM64 Apple Silicon), JNum queries kernel hardware variables:
* Primary data cache: `sysctl -n hw.l1dcachesize` (or `hw.perflevel0.l1dcachesize`).
* Secondary cache: `sysctl -n hw.l2cachesize`.
* Shared tertiary cache: `sysctl -n hw.l3cachesize`.

### C. Windows: CIM / WMI Topology Query

On Windows, JNum executes an instantaneous CIM query:

```powershell
Get-CimInstance Win32_CacheMemory | Select-Object -Property Level,MaxCacheSize
```

* `Level=3`: L1 Data Cache.
* `Level=4`: L2 Cache.
* `Level=5`: L3 Cache.

JNum normalizes per-core cache allocations against physical core counts to prevent over-estimating capacity on multi-socket systems.

### Resilient Fallbacks

If OS permissions restrict hardware discovery, JNum clamps cache boundaries to safe industry baselines:
* $\text{L1D} = 32 \text{ KB}$
* $\text{L2} = 512 \text{ KB}$
* $\text{L3} = 8 \text{ MB}$

---

## 3. Mathematical Derivation of BLIS Parameters

Once physical cache capacities are known, JNum derives optimal tiling parameters using cache-fitting equations:

### 1. Deriving $K_C$ (L1D Cache Residency)

During micro-kernel execution, two micro-panels must reside concurrently inside the L1 Data cache:
* Micro-panel of $A$: size $M_R \times K_C$.
* Micro-panel of $B$: size $K_C \times N_R$.

To prevent eviction caused by cache associativity conflicts, JNum budgets at most **75% of L1D**:

$$(M_R + N_R) \cdot K_C \cdot \text{elementBytes} \le \frac{3}{4} \cdot \text{L1D\_BYTES}$$

$$K_C = \left\lfloor \frac{\frac{3}{4} \cdot \text{L1D\_BYTES}}{\text{elementBytes} \cdot (M_R + N_R)} \right\rfloor_{64}$$

JNum clamps $K_C \in [128, 256]$, aligned to a multiple of 64.

### 2. Deriving $M_C$ (L2 Cache Residency)

A single thread's packed $A$ panel ($M_C \times K_C$) must fit comfortably inside per-core L2 cache, leaving room for operating system and stack buffers. JNum budgets **50% of L2**:

$$M_C \cdot K_C \cdot \text{elementBytes} \le \frac{1}{2} \cdot \text{L2\_BYTES}$$

$$M_C = \left\lfloor \frac{\frac{1}{2} \cdot \text{L2\_BYTES}}{\text{elementBytes} \cdot K_C} \right\rfloor_{\text{LCM}(M_R, 8)}$$

$M_C$ is aligned to a multiple of $\text{LCM}(M_R, 8) = 24$, ensuring every packed panel decomposes into an exact integer number of $M_R = 6$ micro-panels.

### 3. Deriving $N_C$ (Shared L3 Cache Residency)

The shared packed $B$ panel ($K_C \times N_C$) is packed once and shared across all worker threads in the multi-core L3 cache. JNum budgets **50% of L3**:

$$K_C \cdot N_C \cdot \text{elementBytes} \le \frac{1}{2} \cdot \text{L3\_BYTES}$$

$$N_C = \left\lfloor \frac{\frac{1}{2} \cdot \text{L3\_BYTES}}{\text{elementBytes} \cdot K_C} \right\rfloor_{\text{LCM}(N_R, 16)}$$

---

## 4. Balanced Parallel Partitioning (`computeDynamicLeafMc`)

In multi-threaded matrix multiplication, static row chunking ($N / \text{cores}$) causes severe **straggler thread delays** if the matrix height is not an exact multiple of the thread count:

```
Static Partitioning Flaw:
Thread 0: [ 240 rows ] (Heavy load)
Thread 1: [ 240 rows ] (Heavy load)
Thread 2: [ 240 rows ] (Heavy load)
Thread 3: [  30 rows ] (Finishes early; sits idle waiting at barrier)
```

JNum's parallel GEBP engine implements dynamic task sizing ([`computeDynamicLeafMc`](../../core/src/main/java/jnum/internal/Constants.java#L236)):

```java
public static int computeDynamicLeafMc(int totalRows, int mrStep, int maxMc) {
    int totalPanels = (totalRows + mrStep - 1) / mrStep;
    int maxPanelsInL2 = Math.max(1, maxMc / mrStep);
    
    if (AVAILABLE_CORES <= 1 || totalPanels <= 2) {
        return maxPanelsInL2 * mrStep;
    }
    
    // Distribute panels across available workers
    int targetTasks = Math.max(AVAILABLE_CORES, AVAILABLE_CORES * MATMUL_TASKS_PER_CORE);
    int panelsPerTask = Math.max(2, (totalPanels + targetTasks - 1) / targetTasks);
    int leafPanels = Math.min(maxPanelsInL2, panelsPerTask);
    
    return leafPanels * mrStep;
}
```

This guarantees:
1. Every task holds an exact multiple of $M_R = 6$ rows.
2. The work is distributed evenly across all CPU cores with minimal tail remainder.
3. Every task's $A$ sub-panel fits inside the local core's L2 cache while reusing the shared packed $B$ panel from L3 cache.
