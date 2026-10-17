# Chapter 2: Tensors & Fundamentals

To perform fast numerical operations, a program must lay out numbers contiguously in physical RAM so the CPU cache and vector hardware can stream them without pipeline stalls. 

This chapter explains the mental model behind **`NDArray`**, JNum's multidimensional tensor abstraction, how off-heap memory is managed through Java's Foreign Function & Memory (FFM) API, and how to create and inspect tensors.

---

## 1. Motivation: The Problem with Java Heap Arrays

Standard Java multi-dimensional arrays (like `float[][]` or `double[][][]`) are not contiguous matrices. They are **arrays of references pointing to other arrays** scattered across the JVM garbage-collected heap.

```
Standard Java float[3][3]:
Heap Root ──> [ Ref 0 | Ref 1 | Ref 2 ]
                 │       │       │
                 ▼       ▼       ▼
              [f, f, f] [f, f, f] [f, f, f]  <-- Disjoint heap memory locations
```

This layout leads to three critical bottlenecks in high-performance computing:
1. **Pointer Chasing & Cache Misses**: Elements in adjacent rows are not contiguous in physical RAM. Traversing rows causes frequent CPU L1/L2 cache misses.
2. **Object Overhead**: Every row array carries a 16-to-24-byte JVM object header, dramatically increasing memory consumption for small dimensions.
3. **Garbage Collection Pressure**: Allocating millions of intermediate arrays during matrix algebra forces the JVM garbage collector into frequent stop-the-world pauses.

### The JNum Solution

An `NDArray` in JNum is backed by a single, contiguous off-heap **`java.lang.foreign.MemorySegment`**, aligned to 64-byte hardware cache-line boundaries:

```
JNum NDArray (3x3 float32, contiguous off-heap):
Offset (bytes):  0     4     8     12    16    20    24    28    32
Memory Segment: [1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0]
Shape:          [3, 3]
Strides:        [3, 1]  (step 3 elements for row, step 1 for column)
```

Data flows directly between native RAM and CPU SIMD registers without ever touching the JVM garbage-collected heap.

---

## 2. Supported Data Types (`DType`)

Every `NDArray` possesses a concrete data type specified by the [`DType`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/DType.java) enum:

| `DType` | Underlying FFM `ValueLayout` | Size | Description |
| :--- | :--- | :--- | :--- |
| **`f32`** | `ValueLayout.JAVA_FLOAT` | 4 bytes | 32-bit IEEE 754 single-precision float. Default for neural networks and high-throughput vector math. |
| **`f64`** | `ValueLayout.JAVA_DOUBLE` | 8 bytes | 64-bit IEEE 754 double-precision float. Standard for scientific computing, covariance, and solvers. |
| **`i32`** | `ValueLayout.JAVA_INT` | 4 bytes | 32-bit signed two's complement integer. Standard for discrete indexes and masks. |
| **`bool`** | `ValueLayout.JAVA_BYTE` | 1 byte | Boolean truth value stored as an 8-bit byte (`1` for true, `0` for false). |

> [!NOTE]
> `i64` (long) and `f16` (half precision) are not included in JNum 0.2.0. The JVM Vector API currently provides its highest SIMD performance on 32-bit and 64-bit floating-point and integer primitives.

---

## 3. The Memory Lifecycle: Automatic vs Scoped Arenas

In Project Panama, all off-heap allocations are governed by an **`Arena`**. JNum gives you full control over whether native memory is managed automatically by the garbage collector or deterministically by your code.

### Mode A: Automatic Management (`Arena.ofAuto()`)

By default, all `JNum.*` factory methods allocate off-heap memory using `Arena.ofAuto()`. The JVM garbage collector monitors the `MemorySegment` and reclaims native memory once the `NDArray` is no longer reachable.

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // Backed by Arena.ofAuto() - freed automatically when out of scope
    NDArray a = JNum.zeros(1000, 1000);
}
```

### Mode B: Deterministic Scoped Memory (`Arena.ofConfined()`)

For latency-critical applications or long-running servers where you want **zero native memory fragmentation and zero GC overhead**, allocate tensors inside an explicit scoped arena using Java's `try-with-resources`:

```java
import java.lang.foreign.Arena;
import jnum.JNum;
import jnum.DType;
import jnum.NDArray;

void main() {
    // When the try block exits, all native memory allocated inside is 
    // immediately returned to the OS.
    try (Arena arena = Arena.ofConfined()) {
        NDArray weights = JNum.rand(arena, DType.f32, 2048, 2048);
        NDArray inputs  = JNum.ones(arena, DType.f32, 2048, 512);
        
        NDArray result = weights.matmul(inputs);
        IO.println("Result computed: " + result.shapeString());
    } // <-- Immediate 100% deallocation here. Zero GC pauses!
}
```

> [!WARNING]
> Accessing an `NDArray` after its governing `Arena` has closed throws an `IllegalStateException`. See [Chapter 14: Safety & Error Diagnostics](14-safety-and-error-diagnostics.md) for details.

---

## 4. Tensor Factories

The [`JNum`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/JNum.java) class provides static factory methods to construct tensors.

### Uniform Initializers (`zeros`, `ones`, `eye`)

```java
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

void main() {
    // Zeros with default DType.f32
    NDArray z1 = JNum.zeros(3, 4);

    // Zeros with explicit DType.f64
    NDArray z2 = JNum.zeros(DType.f64, 10, 10);

    // Ones (3D tensor of shape [2, 3, 4])
    NDArray o = JNum.ones(2, 3, 4);

    // Identity matrix (4x4, defaults to DType.f64)
    NDArray eye = JNum.eye(4);
}
```

### Numerical Sequences (`arange`)

`JNum.arange` creates evenly spaced 1D sequences:

```java
// Sequence [0.0, 1.0, 2.0, 3.0, 4.0]
NDArray seq1 = JNum.arange(5.0);

// Sequence [2.0, 4.0, 6.0, 8.0]
NDArray seq2 = JNum.arange(2.0, 10.0, 2.0);

// Sequence cast to integer
NDArray seq3 = JNum.arange(0, 10, 1, DType.i32, java.lang.foreign.Arena.ofAuto());
```

### Pseudo-Random Initializers (`rand`)

Populate tensors with uniform random numbers using thread-local fast generators:

```java
// Uniform random floats in [0.0, 1.0)
NDArray r1 = JNum.rand(4, 4);

// Uniform random doubles in [0.0, 1.0)
NDArray r2 = JNum.rand(DType.f64, 4, 4);

// Bounded random floats in [ -1.0f, 1.0f )
NDArray r3 = JNum.rand(-1.0f, 1.0f, DType.f32, 100);

// Bounded random integers in [ 0, 100 )
NDArray r4 = JNum.rand(0, 100, DType.i32, 10, 10);
```

### Bridging Primitive Java Arrays (`JNum.from`)

To load data already held in Java heap primitive arrays, use `JNum.from(...)`. The source primitive array is copied directly into native memory in row-major order:

```java
// Bridge float[] into a 2x3 matrix
float[] rawFloats = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
};
NDArray matrix = JNum.from(rawFloats, 2, 3);

// Bridge double[] into a 1D vector
double[] rawDoubles = {0.1, 0.2, 0.3};
NDArray vector = JNum.from(rawDoubles, 3);

// Bridge boolean flags
boolean[] flags = {true, false, true, true};
NDArray mask = JNum.from(flags, 2, 2);
```

---

## 5. Inspecting Tensors

You can query tensor metadata and examine elements with zero overhead:

```java
NDArray a = JNum.rand(DType.f32, 2, 3, 4);

long[] shape = a.getShape();         // [2, 3, 4]
int rank     = a.dim();              // 3
long count   = a.getSize();          // 24 total elements
DType dtype  = a.getDType();         // DType.f32
String desc  = a.shapeString();      // "(2, 3, 4)"

IO.println("Dimensions: " + desc);
IO.println("Type: " + dtype);
IO.println("Memory Size: " + (count * 4) + " bytes");
```

---

## 6. Practical Recipe: Deterministic Memory Benchmarking

The following complete script demonstrates the memory advantage of JNum. We allocate a large batch of matrices, perform computation, and deterministically release the native memory back to the operating system:

```java
import java.lang.foreign.Arena;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

void main() {
    IO.println("Allocating 1,000 matrices (1000x1000 float32 = ~4 GB RAM) across scoped arenas...");

    long startTime = System.currentTimeMillis();

    for (int batch = 0; batch < 10; batch++) {
        // Confined arena limits native lifetime to this block
        try (Arena batchArena = Arena.ofConfined()) {
            NDArray a = JNum.rand(batchArena, DType.f32, 1000, 1000);
            NDArray b = JNum.ones(batchArena, DType.f32, 1000, 1000);
            
            // Perform in-place addition directly in native memory
            a.addi(b);
            
            // Print progress
            IO.println("Batch " + batch + " completed. Sample element: " + a.getFloat(0, 0));
        } // 400 MB allocated in this batch is unmapped instantly here!
    }

    long elapsed = System.currentTimeMillis() - startTime;
    IO.println("Finished in " + elapsed + " ms with zero heap memory pressure.");
}
```

Now that you understand how tensors live in physical memory, proceed to **[Chapter 3: Layouts, Strides, and Slicing](03-layouts-strides-and-slicing.md)** to master zero-copy views, coordinate subviews, and Pythonic multidimensional slicing.
