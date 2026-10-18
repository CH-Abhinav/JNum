# Chapter 5: Comparisons & Boolean Logic

Conditional logic in high-performance computing requires special care. Traditional `if-else` branches inside inner loops cause CPU **branch mispredictions**, flushing hardware instruction pipelines.

JNum solves this by providing SIMD-accelerated **element-wise comparisons and boolean logic**. Comparisons execute branch-free across vector lanes, returning boolean masks or clamped values with deterministic throughput.

---

## 1. Element-Wise Clamping: `maximum` and `minimum`

Rather than branching with `Math.max(x, y)` in scalar code, JNum provides SIMD-vectorized [`maximum()`](../../core/src/main/java/jnum/NDArray.java#L1305) and [`minimum()`](../../core/src/main/java/jnum/NDArray.java#L1337).

### Tensor-Tensor Clamping with Broadcasting

Compute the element-wise upper or lower envelope between two tensors:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray a = JNum.from(new float[]{1.0f, 5.0f, 2.0f, 8.0f}, 4);
    NDArray b = JNum.from(new float[]{4.0f, 3.0f, 6.0f, 7.0f}, 4);

    NDArray maxVals = a.maximum(b); // [4.0, 5.0, 6.0, 8.0]
    NDArray minVals = a.minimum(b); // [1.0, 3.0, 2.0, 7.0]

    IO.println("Max: " + maxVals);
    IO.println("Min: " + minVals);
}
```

### Tensor-Scalar Clamping (The ReLU Pattern)

You can clamp tensors against scalar numbers (`float`, `int`, or `double`). A canonical example is the **Rectified Linear Unit (ReLU)** activation function in deep learning:

$$\text{ReLU}(x) = \max(x, 0)$$

```java
NDArray activations = JNum.from(new float[]{-2.5f, 0.0f, 1.8f, -0.4f, 3.2f}, 5);

// Clamps all negative values to 0.0f without branching
NDArray reluOutput = activations.maximum(0.0f);
// Output: [0.0, 0.0, 1.8, 0.0, 3.2]

// Bounding values between [0.0, 1.0] (Clipping)
NDArray clipped = activations.maximum(0.0f).minimum(1.0f);
// Output: [0.0, 0.0, 1.0, 0.0, 1.0]
```

---

## 2. Boolean Logic: `and`, `or`, `xor`, `not`

Tensors of type `DType.bool` represent arrays of boolean truth values (stored as 8-bit native bytes, where `1` is true and `0` is false).

JNum supports vectorized boolean logic between masks:

```java
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray mask1 = JNum.from(new boolean[]{true,  true,  false, false}, 4);
    NDArray mask2 = JNum.from(new boolean[]{true,  false, true,  false}, 4);

    // Logical AND
    NDArray andResult = mask1.and(mask2); // [true, false, false, false]

    // Logical OR
    NDArray orResult  = mask1.or(mask2);  // [true, true, true, false]

    // Logical XOR
    NDArray xorResult = mask1.xor(mask2); // [false, true, true, false]

    // Logical NOT (Inversion)
    NDArray notResult = mask1.not();      // [false, false, true, true]
}
```

### Zero-Allocation Output Buffers

For performance-critical filtering pipelines, supply a pre-allocated destination buffer:

```java
NDArray resBuffer = JNum.zeros(DType.bool, 4);

mask1.and(mask2, resBuffer); // Writes directly into resBuffer with zero GC
```

---

## 3. Truthiness Reductions: `all` and `any`

To reduce an entire tensor to a single boolean summary flag:

| Method | Behavior |
| :--- | :--- |
| [`a.all()`](../../core/src/main/java/jnum/NDArray.java#L1939) | Returns `true` if **every** element in the tensor is non-zero (truthy). |
| [`a.any()`](../../core/src/main/java/jnum/NDArray.java#L1932) | Returns `true` if **at least one** element in the tensor is non-zero. |

```java
NDArray flags1 = JNum.from(new boolean[]{true, true, true}, 3);
IO.println(flags1.all()); // true
IO.println(flags1.any()); // true

NDArray flags2 = JNum.from(new boolean[]{false, true, false}, 3);
IO.println(flags2.all()); // false
IO.println(flags2.any()); // true

NDArray flags3 = JNum.zeros(DType.bool, 3);
IO.println(flags3.all()); // false
IO.println(flags3.any()); // false
```

---

## 4. Practical Recipe: Anomaly Detection with Bounded Envelopes

The following recipe inspects a stream of sensor readings, applies an acceptable tolerance band using `maximum` and `minimum`, and verifies system safety using `all()`:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // 8 sensor channels with nominal readings
    NDArray readings = JNum.from(new float[]{
        24.5f, 25.1f, 24.8f, 29.8f, 25.0f, 24.9f, 25.3f, 24.7f
    }, 8);

    float lowerThreshold = 24.0f;
    float upperThreshold = 28.0f;

    // Verify all readings are within safe bounds:
    // If maximum(lowerThreshold) == readings, then all readings >= lowerThreshold
    NDArray lowerBounded = readings.maximum(lowerThreshold);
    NDArray upperBounded = readings.minimum(upperThreshold);

    // Compute difference from original readings:
    // If difference is 0 everywhere, value is within limits
    NDArray lowerDiff = lowerBounded.sub(readings).abs();
    NDArray upperDiff = readings.sub(upperBounded).abs();

    boolean satisfiesLower = lowerDiff.sum() == 0.0;
    boolean satisfiesUpper = upperDiff.sum() == 0.0;

    IO.println("All sensors >= 24.0C: " + satisfiesLower);
    IO.println("All sensors <= 28.0C: " + satisfiesUpper);
    
    if (!satisfiesUpper) {
        IO.println("ALERT: Sensor channel exceeded 28.0C threshold!");
    }
}
```

Now proceed to **[Chapter 6: Transcendental & Unary Operations](06-transcendental-and-unary-ops.md)** for vectorized mathematical functions (`exp`, `log`, `sin`, `tanh`).
