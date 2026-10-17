# Chapter 3: Layouts, Strides, and Slicing

In numerical computing, moving and copying bytes is often the slowest part of a pipeline. If a simple matrix transpose or sub-region crop requires copying megabytes of data, algorithm throughput plummets.

JNum solves this by decoupling the **physical native memory** from its **logical multidimensional view**. Operations like transposing, reshaping, sub-setting, and slicing are **$O(1)$ zero-copy views**: they alter only the pointer offset, shape, and strides metadata, leaving the underlying native bytes untouched.

---

## 1. The Mental Model: Row-Major Strides

An `NDArray` stores elements in a flat, 1D off-heap native memory segment. To translate a multidimensional coordinate $(i_0, i_1, \dots, i_{D-1})$ into a physical byte offset in RAM, JNum computes an inner product with the tensor's **strides**:

$$\text{Element Offset} = \sum_{k=0}^{D-1} i_k \cdot s_k$$

$$\text{Byte Address} = \text{Base Address} + (\text{Element Offset} \times \text{elementBytes})$$

For a contiguous row-major tensor of dimensions $[d_0, d_1, \dots, d_{D-1}]$, the canonical stride $s_k$ for dimension $k$ is the product of all subsequent dimensions:

$$s_k = \prod_{m=k+1}^{D-1} d_m, \quad \text{with } s_{D-1} = 1$$

### Example: A 2D Matrix of Shape `[3, 4]`

```
Logical Matrix (Shape: [3, 4], Strides: [4, 1]):
Row 0: [ (0,0),  (0,1),  (0,2),  (0,3) ]
Row 1: [ (1,0),  (1,1),  (1,2),  (1,3) ]
Row 2: [ (2,0),  (2,1),  (2,2),  (2,3) ]

Physical Memory Segment (Flat 1D):
Offset:   0      1      2      3      4      5      6      7      8      9      10     11
Element: (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
```

To access element $(1, 2)$:
$$\text{Element Offset} = 1 \cdot 4 + 2 \cdot 1 = 6$$

---

## 2. Views vs. Copies: Zero-Allocation Transformations

Because the logical coordinates depend entirely on the `shape` and `strides` arrays, transformations that swap or collapse dimensions do not copy any underlying memory.

### Transposition (`transpose()`)

Transposing a 2D matrix simply reverses its strides:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // Shape [2, 3], Strides [3, 1]
    NDArray a = JNum.from(new float[]{
        1, 2, 3,
        4, 5, 6
    }, 2, 3);

    // Transpose: Shape becomes [3, 2], Strides become [1, 3]
    NDArray aT = a.transpose();

    IO.println(aT);
}
```

> [!NOTE]
> `a.transpose()` completes in nanoseconds because it allocates only a new `NDArray` wrapper holding the reversed strides. The underlying data segment is shared. Mutating `aT` will mutate `a`.

### Reshaping (`reshape()`)

Reshaping reorganizes dimensions without altering physical memory order:

```java
NDArray seq = JNum.arange(12.0); // Shape [12]

// Reshape into 3 rows of 4 columns
NDArray matrix = seq.reshape(3, 4);

// Reshape into a 3D batch: 2 batches of 2 rows by 3 columns
NDArray batch = seq.reshape(2, 2, 3);
```

### Squeezing and Unsqueezing Dimensions

To add or remove degenerate dimensions of length 1 (common in batch processing):

```java
NDArray x = JNum.zeros(1, 10, 1);

// Squeeze axis 0 -> Shape [10, 1]
NDArray sq = x.squeeze(0);

// Unsqueeze axis 1 -> Shape [1, 1, 10, 1]
NDArray unsq = x.unsqueeze(1);
```

---

## 3. Contiguity: When Memory Order Matters

A tensor is **contiguous** if its elements follow a strictly sequential, monotonic layout in physical memory with no gaps.

```java
NDArray a = JNum.zeros(4, 4);
IO.println(a.isContiguous()); // true

NDArray aT = a.transpose();
IO.println(aT.isContiguous()); // false (strides are [1, 4] instead of [4, 1])
```

Why does contiguity matter?
* Highly optimized SIMD vector kernels (AVX2/AVX-512) stream contiguous memory sequentially into vector registers.
* When non-contiguous tensors are passed to dense GEMM or reductions, JNum automatically converts them or steps along physical strides.
* To force a non-contiguous view into a freshly allocated, sequential memory segment, call `a.contiguous()` or `a.copy()`:

```java
// Allocates a fresh contiguous segment with stride-1 layout
NDArray contiguousMatrix = aT.contiguous();
IO.println(contiguousMatrix.isContiguous()); // true
```

---

## 4. Extracting Subviews

When you want to slice along leading dimensions by fixing coordinates, use [`subview()`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/NDArray.java#L603).

```java
// A 3D tensor representing [batch=4, channels=3, height=32]
NDArray tensor = JNum.rand(4, 3, 32);

// Extract batch index 0 -> Returns 2D tensor of shape [3, 32]
NDArray firstBatch = tensor.subview(0);

// Fix both batch=0 and channel=1 -> Returns 1D tensor of shape [32]
NDArray channelSlice = tensor.subview(0, 1);
```

`subview(...)` is a zero-copy pointer slice: it adjusts the base memory offset and drops the leading dimensions.

---

## 5. Python-Style Slicing with `Slice`

For arbitrary multidimensional ranges, steps, and strided intervals, JNum provides the [`Slice`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/Slice.java) record.

### Slice Factory Methods

| Method | Equivalent Python Syntax | Meaning |
| :--- | :--- | :--- |
| `Slice.all()` | `:` | Select the entire dimension. |
| `Slice.to(stop)` | `:stop` | From index 0 up to `stop` (exclusive). |
| `Slice.from(start)` | `start:` | From `start` through the end of the dimension. |
| `Slice.range(start, stop)` | `start:stop` | From `start` (inclusive) to `stop` (exclusive). |
| `Slice.range(start, stop, step)`| `start:stop:step` | Strided range with step size. |
| `Slice.step(step)` | `::step` | Entire dimension with custom step (e.g. `-1` to reverse). |

### Example: Multi-Axis Slicing

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.Slice;

void main() {
    // 2D Matrix of shape [6, 8]
    NDArray matrix = JNum.arange(48.0).reshape(6, 8);

    // Select rows 1 to 4 (exclusive), and all columns
    NDArray r1 = matrix.slice(Slice.range(1, 4), Slice.all());
    // Shape: [3, 8]

    // Select every second row and columns 2 to 6
    NDArray r2 = matrix.slice(Slice.step(2), Slice.range(2, 6));
    // Shape: [3, 4]

    // Reverse the entire matrix along both dimensions (zero-copy negative strides!)
    NDArray reversed = matrix.slice(Slice.step(-1), Slice.step(-1));
}
```

---

## 6. String-Based Pythonic Slicing

If you prefer Python syntax without importing `Slice` tokens, pass a string directly to `a.slice(String)`:

```java
NDArray matrix = JNum.arange(100.0).reshape(10, 10);

// Select rows 0 to 5, columns 2 to 8
NDArray s1 = matrix.slice("0:5, 2:8");

// Skip every second column across all rows
NDArray s2 = matrix.slice(":, ::2");

// Reverse rows, keep columns 1 through 4
NDArray s3 = matrix.slice("::-1, 1:4");
```

---

## 7. Direct Element Access & Mutation

To read or update individual scalar values without intermediate object allocations, use the typed getter and setter methods:

```java
NDArray a = JNum.zeros(5, 5);

// Direct 2D mutations
a.setFloat(42.0f, 0, 0);
a.setFloat(3.14f, 1, 2);

// Direct 2D reads
float val1 = a.getFloat(0, 0); // 42.0
float val2 = a.getFloat(1, 2); // 3.14

// General N-D getters (with promotion to double)
double generalVal = a.get(1, 2);

// Flat linear index read (bypasses coordinate calculation)
float firstElement = a.getFlatFloat(0);
```

---

## 8. Practical Recipe: Zero-Copy Image Cropping & Mirroring

The following recipe demonstrates processing an image tensor (`[Height, Width, Channels]`), extracting a region of interest (ROI), and mirroring it horizontally—all with zero heap allocations or buffer copies:

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.Slice;

void main() {
    // Simulate a 1080p RGB image: [1080, 1920, 3]
    NDArray image = JNum.rand(1080, 1920, 3);
    IO.println("Original image: " + image.shapeString());

    // Crop a 400x400 central bounding box:
    // Rows: 340 to 740, Cols: 760 to 1160, All color channels
    NDArray crop = image.slice(
        Slice.range(340, 740),
        Slice.range(760, 1160),
        Slice.all()
    );
    IO.println("Cropped bounding box: " + crop.shapeString());

    // Horizontally flip the cropped image using a negative column step
    NDArray flipped = crop.slice(
        Slice.all(),
        Slice.step(-1), // reverse columns
        Slice.all()
    );
    IO.println("Flipped crop: " + flipped.shapeString());
    IO.println("Memory copied: 0 bytes (zero-copy view)");
}
```

With memory layout and slicing understood, proceed to **[Chapter 4: SIMD Arithmetic & Broadcasting](04-simd-arithmetic-and-broadcasting.md)** to unleash hardware vector processing.
