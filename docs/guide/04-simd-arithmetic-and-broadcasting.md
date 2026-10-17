# Chapter 4: SIMD Arithmetic & Broadcasting

Modern CPUs contain dedicated SIMD (Single Instruction, Multiple Data) execution units—such as Intel AVX-512, AMD AVX2, and ARM NEON—capable of processing 4, 8, or 16 numbers in a single clock cycle. 

JNum connects directly to these hardware vector units using Java's **Vector API** (`jdk.incubator.vector`). This chapter explains how element-wise arithmetic works, how the zero-copy broadcasting engine aligns mismatched dimensions, and how to write high-performance loops with zero garbage collection allocations.

---

## 1. Motivation: Scalar Loops vs. SIMD Vector Lanes

Consider adding two arrays of 1,000,000 floats. In traditional Java:

```java
// Naive scalar loop: 1 addition per CPU instruction
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

JIT compilers attempt to "auto-vectorize" such loops, but frequently fail due to array aliasing checks, bounds checks, or complex loop steps.

In JNum, arithmetic is vectorized explicitly via the Vector API. When running on an AVX-512 capable CPU, a single `FloatVector.add()` instruction computes **16 float additions simultaneously**:

```
AVX-512 SIMD Vector Lane (512 bits = 16 floats):
Register A: [ a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8 | a9 | a10| a11| a12| a13| a14| a15 ]
               +    +    +    +    +    +    +    +    +    +    +    +    +    +    +    +
Register B: [ b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | b10| b11| b12| b13| b14| b15 ]
               =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =
Output C:   [ c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10| c11| c12| c13| c14| c15 ]
```

---

## 2. Basic Binary Arithmetic

Every `NDArray` supports element-wise addition, subtraction, multiplication, and division:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray a = JNum.from(new float[]{1, 2, 3, 4}, 2, 2);
    NDArray b = JNum.from(new float[]{10, 20, 30, 40}, 2, 2);

    NDArray sum  = a.add(b); // [[11, 22], [33, 44]]
    NDArray diff = b.sub(a); // [[9, 18], [27, 36]]
    NDArray prod = a.mul(b); // [[10, 40], [90, 160]]
    NDArray quot = b.div(a); // [[10, 10], [10, 10]]
}
```

### Scalar Arithmetic

You can combine tensors directly with primitive scalar numbers (`float`, `double`, `int`):

```java
NDArray x = JNum.ones(3, 3);

NDArray scaled = x.mul(10.0f);     // Multiply each element by 10
NDArray offset = scaled.add(5.0);  // Add 5.0 (promotes to DType.f64)
NDArray halved = offset.div(2);    // Divide by integer 2
```

---

## 3. The Broadcasting Engine

When operating on two tensors of different shapes, JNum applies **NumPy-compatible broadcasting rules**. Broadcasting virtually stretches dimensions of size 1 so shapes match without copying data in memory.

### The Broadcasting Rules

1. **Align Dimensions**: Shapes are compared starting from the trailing (rightmost) dimension moving leftward.
2. **Compatibility Check**: Two dimensions are compatible if:
   * They are equal, OR
   * One of them is `1`.
3. **Leading Dimension Padding**: If one tensor has fewer dimensions than the other, its shape is prepended with `1`s until ranks match.

### Example Visualizations

#### Case 1: Matrix + Row Vector
* Shape A: `[3, 4]` (3 rows, 4 columns)
* Shape B: `[4]` $\rightarrow$ aligned to `[1, 4]`
* **Broadcast Result**: `[3, 4]` (Vector B is broadcast across all 3 rows)

```
Matrix A [3, 4]:          Row Vector B [1, 4]:        Result [3, 4]:
[ 1,  2,  3,  4 ]     +   [ 10, 20, 30, 40 ]      =   [ 11, 22, 33, 44 ]
[ 5,  6,  7,  8 ]     +   [ 10, 20, 30, 40 ]      =   [ 15, 26, 37, 48 ]
[ 9, 10, 11, 12 ]     +   [ 10, 20, 30, 40 ]      =   [ 19, 30, 41, 52 ]
```

#### Case 2: Column Vector + Row Vector (Outer Sum)
* Shape A: `[3, 1]`
* Shape B: `[1, 4]`
* **Broadcast Result**: `[3, 4]`

```java
NDArray col = JNum.from(new float[]{1, 2, 3}, 3, 1);
NDArray row = JNum.from(new float[]{10, 20, 30, 40}, 1, 4);

NDArray grid = col.add(row);
IO.println(grid.shapeString()); // "(3, 4)"
IO.println(grid);
```

### Zero-Allocation Stride Magic (Stride = 0)

How does JNum broadcast without duplicating memory? **By setting the stride to 0.**

When a dimension of length 1 is stretched to length $N$, its stride is set to `0`. As the SIMD kernel advances along that dimension in coordinate space, the physical memory pointer adds $i \times 0 = 0$ bytes, continuously reusing the exact same element in the CPU L1 cache!

---

## 4. Zero-Allocation Patterns for Production Loops

In low-latency systems or iterative algorithms (like gradient descent or real-time sensor processing), calling `a.add(b)` repeatedly causes memory churn because every call allocates a new `NDArray` buffer.

JNum provides two zero-allocation patterns to eliminate this overhead:

### Pattern A: In-Place Mutation (`addi`, `subi`, `muli`, `divi`)

Operations ending in `i` mutate the receiver tensor's off-heap memory directly:

```java
NDArray state = JNum.zeros(1000, 1000);
NDArray delta = JNum.ones(1000, 1000);

// In-place: mutates 'state' directly without allocating any new buffer
state.addi(delta);
state.muli(0.99f); // apply decay factor in-place
```

### Pattern B: Pre-Allocated Output Buffers

When you cannot mutate the input tensors, pre-allocate an output buffer once and reuse it across iterations:

```java
NDArray a = JNum.rand(512, 512);
NDArray b = JNum.rand(512, 512);

// Allocate the destination buffer ONCE
NDArray buffer = JNum.zeros(512, 512);

for (int iter = 0; iter < 100_000; iter++) {
    // Writes results directly into 'buffer' - 0 bytes of heap/native allocation!
    a.add(b, buffer);
}
```

---

## 5. Practical Recipe: Zero-GC Min-Max Feature Normalization

The following production recipe standardizes a batch of feature vectors into the range $[0.0, 1.0]$ using min-max normalization:

$$\text{Normalized}(X) = \frac{X - \min(X)}{\max(X) - \min(X)}$$

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // 10,000 samples with 8 numerical features: [10000, 8]
    NDArray features = JNum.rand(10.0f, 100.0f, jnum.DType.f32, 10000, 8);

    // Compute min and max along axis 0 (across samples, retaining dimensions [1, 8])
    NDArray minVals = features.min(0, true); // Shape [1, 8]
    NDArray maxVals = features.max(0, true); // Shape [1, 8]

    // Precompute denominator range: max - min
    NDArray range = maxVals.sub(minVals);

    // Normalize using broadcasting:
    // features [10000, 8] - minVals [1, 8] / range [1, 8]
    NDArray normalized = features.sub(minVals).divi(range);

    IO.println("Original sample 0:   " + features.subview(0));
    IO.println("Normalized sample 0: " + normalized.subview(0));
}
```

Now proceed to **[Chapter 5: Comparisons & Boolean Logic](05-comparisons-and-boolean-logic.md)** to learn about element-wise clamping, boolean masks, and truth reductions.
