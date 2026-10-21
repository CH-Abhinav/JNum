# Chapter 6: Transcendental & Unary Operations

Calculating mathematical functions like exponentials, logarithms, and trigonometry on millions of data points is fundamental to signal processing, statistics, and machine learning.

Executing `Math.sin(x)` or `Math.exp(x)` inside a standard Java loop suffers from substantial scalar function-call overhead. JNum provides SIMD-vectorized **unary and transcendental operations** that evaluate mathematical approximations directly across vector registers.

---

## 1. Unary Mathematical Functions

JNum provides two complementary ways to call unary operations:
1. **NumPy-Style Static Functions via [`JNum`](../../core/src/main/java/jnum/JNum.java#L545)** (e.g. `JNum.sqrt(x)`, `JNum.sin(x)`)
2. **Fluent Instance Methods on [`NDArray`](../../core/src/main/java/jnum/NDArray.java#L1515)** (e.g. `x.sqrt()`, `x.sin()`)

Both styles execute identical SIMD vector kernels. Furthermore, every operation supports three execution variants:
* `JNum.sin(a)` / `a.sin()` — Allocates a new tensor with automatic GC lifecycle (`Arena.ofAuto()`).
* `JNum.sin(a, res)` / `a.sin(res)` — Writes directly into a pre-allocated destination buffer `res` (zero memory allocations).
* `JNum.sin(a, arena)` / `a.sin(arena)` — Allocates the result in an explicit scoped `Arena`.

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray x = JNum.from(new float[]{1.0f, 4.0f, 9.0f, 16.0f}, 4);

    // Fluent method chaining or static JNum call:
    NDArray root1 = x.sqrt();           // [1.0, 2.0, 3.0, 4.0]
    NDArray root2 = JNum.sqrt(x);       // Identical result

    // Absolute value
    NDArray neg = JNum.from(new float[]{-3.5f, 2.0f, -0.1f}, 3);
    NDArray absVal = JNum.abs(neg);     // [3.5, 2.0, 0.1]
}
```

### Exponential and Logarithms

$$\exp(x) = e^x, \quad \ln(x) = \log_e(x), \quad \log_{10}(x)$$

```java
NDArray z = JNum.from(new double[]{0.0, 1.0, 2.0}, 3);

// Natural exponential (e^x)
NDArray e = z.exp(); // [1.0, 2.71828..., 7.38905...]

// Natural logarithm (ln)
NDArray ln = e.log(); // [0.0, 1.0, 2.0]

// Base-10 logarithm
NDArray tens = JNum.from(new double[]{1.0, 10.0, 100.0, 1000.0}, 4);
NDArray log10 = tens.log10(); // [0.0, 1.0, 2.0, 3.0]
```

### Vectorized Sigmoid

The logistic sigmoid function is ubiquitous in binary classification and gating mechanisms:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```java
NDArray logits = JNum.from(new float[]{-6.0f, 0.0f, 6.0f}, 3);
NDArray probabilities = logits.sigmoid();
// Output: [0.00247..., 0.5, 0.99752...]
```

---

## 2. Circular Trigonometry

JNum provides element-wise circular trigonometric functions accepting angles in radians:

| Function | Instance Method | Static Function | Formula |
| :--- | :--- | :--- | :--- |
| **Sine** | [`a.sin()`](../../core/src/main/java/jnum/NDArray.java#L1573) | [`JNum.sin(a)`](../../core/src/main/java/jnum/JNum.java#L769) | $\sin(x)$ |
| **Cosine** | [`a.cos()`](../../core/src/main/java/jnum/NDArray.java#L1582) | [`JNum.cos(a)`](../../core/src/main/java/jnum/JNum.java#L804) | $\cos(x)$ |
| **Tangent** | [`a.tan()`](../../core/src/main/java/jnum/NDArray.java#L1591) | [`JNum.tan(a)`](../../core/src/main/java/jnum/JNum.java#L839) | $\tan(x)$ |

```java
// Sequence of angles from 0 to 2*PI with step 0.1
double pi = Math.PI;
NDArray theta = JNum.arange(0.0, 2.0 * pi, 0.1);

// Static or instance style:
NDArray sinWave = JNum.sin(theta);
NDArray cosWave = theta.cos();
NDArray tanWave = theta.tan();
```

---

## 3. Hyperbolic Trigonometry

Hyperbolic functions are widely used in physics simulations, relativistic mechanics, and neural network recurrent cells:

| Function | Instance Method | Static Function | Formula |
| :--- | :--- | :--- | :--- |
| **Hyperbolic Sine** | [`a.sinh()`](../../core/src/main/java/jnum/NDArray.java#L1600) | [`JNum.sinh(a)`](../../core/src/main/java/jnum/JNum.java#L874) | $\sinh(x) = \frac{e^x - e^{-x}}{2}$ |
| **Hyperbolic Cosine** | [`a.cosh()`](../../core/src/main/java/jnum/NDArray.java#L1609) | [`JNum.cosh(a)`](../../core/src/main/java/jnum/JNum.java#L909) | $\cosh(x) = \frac{e^x + e^{-x}}{2}$ |
| **Hyperbolic Tangent**| [`a.tanh()`](../../core/src/main/java/jnum/NDArray.java#L1618) | [`JNum.tanh(a)`](../../core/src/main/java/jnum/JNum.java#L944) | $\tanh(x) = \frac{\sinh(x)}{\cosh(x)}$ |

```java
NDArray x = JNum.from(new float[]{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, 5);

// Both styles supported:
NDArray tanh1 = x.tanh();
NDArray tanh2 = JNum.tanh(x);
// Output: [-0.9640, -0.7615, 0.0, 0.7615, 0.9640]
```

---

## 4. Practical Recipe: Simulating a Damped Harmonic Wave

The following recipe generates a high-resolution time series modeling a damped physical oscillator:

$$y(t) = A \cdot e^{-\gamma t} \cdot \cos(\omega t + \phi)$$

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    int sampleRate = 1000; // 1000 Hz
    double duration = 2.0; // 2 seconds
    double dt = 1.0 / sampleRate;

    // Time steps: [0.0, 0.001, 0.002, ..., 2.0] -> 2000 samples
    NDArray t = JNum.arange(0.0, duration, dt);

    double amplitude = 5.0;
    double damping = 1.5;     // gamma
    double frequency = 10.0;  // 10 Hz oscillation
    double omega = 2.0 * Math.PI * frequency;

    // 1. Decay envelope: A * exp(-gamma * t)
    NDArray envelope = t.mul(-damping).exp().mul(amplitude);

    // 2. Oscillating carrier: cos(omega * t)
    NDArray carrier = t.mul(omega).cos();

    // 3. Modulated physical wave: envelope * carrier
    NDArray wave = envelope.mul(carrier);

    IO.println("Wave generated with " + wave.getSize() + " samples.");
    IO.println("Sample at t=0.0s:  " + wave.getFloat(0));
    IO.println("Sample at t=0.1s:  " + wave.getFloat(100));
    IO.println("Peak value:        " + wave.abs().max());
}
```

Now proceed to **[Chapter 7: Linear Algebra & Decompositions](07-linear-algebra-and-decompositions.md)** for matrix multiplication, condition numbers, and matrix factorizations (QR, SVD, Cholesky).
