# Chapter 14: Safety & Error Diagnostics

One of the greatest hazards of using native C/C++ numerical libraries (like OpenBLAS or LibTorch) via JNI is that an out-of-bounds index or buffer over-read does not throw a catchable Java exception—it triggers a catastrophic **Operating System Segmentation Fault (`SIGSEGV`)**, terminating the entire JVM process immediately.

JNum, built on Java's Foreign Function & Memory (FFM) API, guarantees **JVM Memory Safety**. All native memory bounds and lifetimes are checked by the JVM. Instead of process crashes, errors produce clear, catchable Java exceptions.

This chapter details the common failure modes in numerical computing, how to diagnose them, and how to write robust code.

---

## 1. The Closed Arena: `IllegalStateException`

### The Error

When an `NDArray` is allocated inside a deterministic scoped arena (`Arena.ofConfined()`) or loaded via a scoped memory map (`MmapArray`, `MmapTensors`), its native memory segment is tied to that arena's lifetime.

If you attempt to read or mutate a tensor after its `Arena` has closed:

```java
import java.lang.foreign.Arena;
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray leakedTensor;

    try (Arena arena = Arena.ofConfined()) {
        leakedTensor = JNum.ones(arena, 4, 4);
    } // arena.close() runs here

    // FAILS with IllegalStateException: Already closed
    float val = leakedTensor.getFloat(0, 0);
}
```

```text
Exception in thread "main" java.lang.IllegalStateException: Already closed
    at java.base/jdk.internal.foreign.MemorySessionImpl.alreadyClosed(MemorySessionImpl.java:310)
    at jnum.internal.ops.GetOps.getFloat(GetOps.java:42)
```

### The Fix

Never allow scoped tensors to escape the `try-with-resources` block that created them. If a computed result must outlive the scope, copy it into an auto-managed arena (`Arena.ofAuto()`) using [`a.contiguous(Arena.ofAuto())`](../../core/src/main/java/jnum/NDArray.java#L733):

```java
NDArray persistentResult;

try (Arena arena = Arena.ofConfined()) {
    NDArray a = JNum.rand(arena, 1000, 1000);
    NDArray b = JNum.rand(arena, 1000, 1000);
    NDArray temp = a.matmul(b);

    // Copy result into the persistent GC arena before closing
    persistentResult = temp.contiguous(Arena.ofAuto());
}

// Perfectly safe! Backed by Arena.ofAuto()
float val = persistentResult.getFloat(0, 0);
```

---

## 2. Shape Mismatches: `IllegalArgumentException`

### A. Incompatible Broadcasting Shapes

Broadcasting requires matching dimensions or dimensions of size 1. If two tensors cannot be aligned:

```java
NDArray a = JNum.zeros(3, 4);
NDArray b = JNum.zeros(3, 5);

// FAILS: Dimension 4 and 5 cannot broadcast!
NDArray c = a.add(b);
```

```text
Exception in thread "main" java.lang.IllegalArgumentException: 
    Operands could not be broadcast together with shapes [3, 4] and [3, 5]
```

**Diagnostic**: Always inspect shapes with `a.shapeString()` before combining datasets from different sources.

### B. Matrix Multiplication Inner Dimension Mismatch

GEMM requires the number of columns in $A$ to match the number of rows in $B$ ($A \in [N \times M], B \in [M \times P]$):

```java
NDArray A = JNum.zeros(10, 20);
NDArray B = JNum.zeros(30, 40);

// FAILS: Inner dimensions 20 != 30
NDArray C = A.matmul(B);
```

```text
Exception in thread "main" java.lang.IllegalArgumentException: 
    Matrix dimensions incompatible for matmul: (10, 20) and (30, 40)
```

**Diagnostic**: Transpose the second matrix (`A.matmul(B.transpose())`) if $B$ was loaded in $[P \times M]$ format.

---

## 3. Non-Contiguous Memory & Slicing Pitfalls

Because `transpose()` and `slice()` produce zero-copy views, their strides are non-canonical (e.g. strides of $[1, 4]$ instead of $[4, 1]$).

While JNum's element-wise arithmetic and broadcasting handle non-contiguous strides automatically, algorithms requiring strictly sequential memory (like direct block I/O or specialized external buffers) require contiguous data:

```java
NDArray matrix = JNum.zeros(4, 4);
NDArray transposed = matrix.transpose();

if (!transposed.isContiguous()) {
    // Linearize into contiguous physical layout
    transposed = transposed.contiguous();
}
```

---

## 4. Numerical Diagnostics: Detecting `NaN` and `Infinity`

Floating-point algorithms can produce `NaN` (Not a Number) or `Infinity` through division by zero, logarithms of negative numbers, or exploding gradients.

### Identifying `NaN` Values

In IEEE 754 floating-point arithmetic, `NaN` has the unique property that it is **not equal to itself** ($x \ne x$):

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray data = JNum.from(new float[]{1.0f, Float.NaN, 3.0f}, 3);

    // Summing or computing norm over an array containing NaN produces NaN
    double norm = data.norm();
    if (Double.isNaN(norm)) {
        IO.println("CRITICAL: Tensor contains NaN values!");
    }
}
```

### Preventing Softmax Overflow

When computing exponentials $e^x$ of large logits (e.g. $x = 1000$), standard `exp()` overflows to `Infinity`. Always use JNum's built-in [`jnum.nn.Softmax`](../../core/src/main/java/jnum/nn/Softmax.java), which automatically subtracts the maximum logit before exponentiation:

```java
// DO NOT do this manually without shifting:
// NDArray probs = logits.exp().div(logits.exp().sum()); // Risks overflow!

// DO use jnum.nn.Softmax:
NDArray probs = new jnum.nn.Softmax().forward(logits); // 100% numerically safe
```

---

## 5. Production Diagnostic Checklist

When deploying JNum in production services:

1. **JVM Flags**: Ensure `--add-modules jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED` are set in the production JVM startup script.
2. **Arena Hygiene**: Use `Arena.ofConfined()` for per-request transient computations in web handlers to achieve zero garbage collector pressure.
3. **Pre-Allocated Buffers**: In tight server loops, pre-allocate output arrays (`a.add(b, outBuffer)`) to avoid repeated off-heap allocation syscalls.
4. **Contiguity Checks**: Call `a.isContiguous()` before passing slices to dense I/O or serialization.

This concludes the **JNum User Guide**. For a deep architectural look at JNum's C-level speed, hardware cache probing, and vector micro-kernels, proceed to the **[Systems Internals Guide](../internals/01-memory-strides-and-ffm.md)**.
