# Neural Networks & Statistical Computing Guide

JNum provides two dedicated modules for high-performance machine learning inference and quantitative data analysis:
1. **`jnum.nn`**: A lightweight, modular neural network inference engine modeled after modern deep learning frameworks (PyTorch), featuring affine transformations, non-linear activation functions, and sequential layer pipelines.
2. **`jnum.stat`**: A comprehensive descriptive and inferential statistics engine with hardware-vectorized moment calculations and 9 continuous quantile estimation methods.

---

# Part 1: Neural Network Modules (`jnum.nn`)

All layers in JNum implement the [`Module`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/nn/Module.java) interface, which establishes a simple forward propagation contract:

```java
package jnum.nn;

import jnum.NDArray;

public interface Module {
    NDArray forward(NDArray input);
}
```

---

## 1. Linear (Fully Connected) Layer

The [`Linear`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/nn/Linear.java) layer applies an affine linear transformation to incoming tensors:

$$y = x W^T + b$$

### Dimension Conventions
* **`weights`**: 2D tensor of shape `[out_features, in_features]`.
* **`bias`**: Optional 1D tensor of shape `[out_features]` (or `null` for un-biased layers).
* **`input`**: Supports 2D matrices `[batch_size, in_features]` or arbitrary N-D sequence tensors `[batch_size, seq_len, in_features]`. All leading dimensions are preserved while transforming the trailing feature dimension.

### Usage Example
```java
import jnum.JNum;
import jnum.NDArray;
import jnum.nn.Linear;

// Weights: 4 output neurons, 3 input features
NDArray weights = JNum.fromArray(new float[]{
    0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f,
    0.7f, 0.8f, 0.9f,
    1.0f, 1.1f, 1.2f
}, 4, 3);

// Bias: 4 values (one per output neuron)
NDArray bias = JNum.fromArray(new float[]{ 0.01f, 0.02f, 0.03f, 0.04f }, 4);

Linear fc = new Linear(weights, bias);

// Forward pass with batch of 2 samples: shape [2, 3] -> output [2, 4]
NDArray x = JNum.fromArray(new float[]{
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
}, 2, 3);

NDArray out = fc.forward(x);
System.out.println("Output shape: " + out.shapeString()); // [2, 4]
```

---

## 2. Activation Functions

All activation functions operate element-wise and preserve the exact shape and data type of the input tensor.

### ReLU (Rectified Linear Unit)
Computes $y = \max(0, x)$. Sets negative values to 0 while keeping positive values unchanged:
```java
import jnum.nn.ReLU;

ReLU relu = new ReLU();
NDArray activated = relu.forward(x);
```

### Sigmoid
Computes the logistic sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$, squashing values into the range $(0, 1)$:
```java
import jnum.nn.Sigmoid;

Sigmoid sigmoid = new Sigmoid();
NDArray probabilities = sigmoid.forward(logits);
```

### Tanh
Computes the hyperbolic tangent $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, mapping values symmetrically into $(-1, 1)$:
```java
import jnum.nn.Tanh;

Tanh tanh = new Tanh();
NDArray zeroCentered = tanh.forward(x);
```

### Softmax (with Numerical Stability)
Computes normalized probability distributions along a specified axis:

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

```java
import jnum.nn.Softmax;

// Applies softmax along trailing dimension (axis = -1) by default
Softmax softmax = new Softmax(); 

// Or specify an explicit dimension (e.g. axis 1)
Softmax colSoftmax = new Softmax(1);

NDArray probs = softmax.forward(logits);
// Every row in probs sums exactly to 1.0
```

> [!TIP]
> **Numerical Stability Trick**: Naive softmax computation ($e^{x_i} / \sum e^{x_j}$) overflows to `+Infinity` for logits $> 88.0$ in float32. JNum automatically subtracts $\max(x)$ along the target axis before exponentiation ($\max \le 0$, so $e^{x_i - \max} \le 1.0$), ensuring zero overflow while producing mathematically identical results.

---

## 3. Sequential Containers & MLP Inference Example

The [`Sequential`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/nn/Sequential.java) module chains multiple layers together, piping the output of layer $i$ into layer $i+1$.

Here is a complete, working Multi-Layer Perceptron (MLP) for classification:

```java
import java.io.IO;
import jnum.JNum;
import jnum.NDArray;
import jnum.nn.*;

void main() {
    IO.println("=== Building 3-Layer MLP Inference Pipeline ===");

    // Layer 1: [in: 4, out: 8]
    var w1 = JNum.rand(8, 4);
    var b1 = JNum.zeros(8);

    // Layer 2: [in: 8, out: 4]
    var w2 = JNum.rand(4, 8);
    var b2 = JNum.zeros(4);

    // Layer 3 (Classifier): [in: 4, out: 3]
    var w3 = JNum.rand(3, 4);
    var b3 = JNum.zeros(3);

    // Assemble sequential pipeline
    var model = new Sequential(
        new Linear(w1, b1),
        new ReLU(),
        new Linear(w2, b2),
        new Tanh(),
        new Linear(w3, b3),
        new Softmax(-1) // Class probabilities over 3 classes
    );

    // Forward inference on a batch of 5 input vectors: [5, 4] -> [5, 3]
    var inputBatch = JNum.rand(5, 4);
    var predictions = model.forward(inputBatch);

    IO.println("Predicted probabilities shape: " + predictions.shapeString());
    IO.println("Sample 0 class probabilities: " + predictions.slice(jnum.Slice.index(0)));
}
```

---

# Part 2: Comprehensive Statistics (`jnum.stat.Stats`)

All descriptive and inferential statistical functions are housed exclusively in [`Stats`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/stat/Stats.java). Operations support both full-tensor scalar reductions and axis-specific reductions.

---

## 1. Central Tendency & Dispersion

### Mean, Variance, and Standard Deviation

```java
import jnum.NDArray;
import jnum.stat.Stats;

NDArray data = JNum.fromArray(new float[]{ 10f, 12f, 23f, 23f, 16f, 23f, 21f, 16f }, 8);

// Full tensor scalar metrics
double mean = Stats.mean(data); // 18.0
double var  = Stats.var(data);  // Population variance (ddof = 0)
double std  = Stats.std(data);  // Population standard deviation (ddof = 0)

// Sample metrics with Bessel's correction (ddof = 1)
double sampleVar = Stats.var(data, 1);
double sampleStd = Stats.std(data, 1);
```

#### Axis Reductions with `keepDims`
```java
NDArray mat = JNum.arange(12).reshape(3, 4);

// Collapse rows (axis 0) -> Shape [4]
NDArray colMeans = Stats.mean(mat, 0);

// Collapse columns (axis 1) and retain shape -> Shape [3, 1]
NDArray rowMeans = Stats.mean(mat, 1, true);
```

### Median
Computes the 50th percentile. For even element counts, interpolates the two central values:
```java
double medianVal = Stats.median(data);
NDArray rowMedians = Stats.median(mat, 1);
```

---

## 2. Quantiles & Percentiles (`QuantileMethod`)

JNum supports all **9 continuous quantile estimation methods** standardized in R and NumPy:

```java
import jnum.stat.QuantileMethod;
import jnum.stat.Stats;

NDArray samples = JNum.fromArray(new float[]{ 15f, 20f, 35f, 40f, 50f }, 5);

// 25th percentile (1st quartile) using standard linear interpolation
double q25 = Stats.quantile(samples, 0.25, QuantileMethod.LINEAR);

// 99th percentile using Weibull estimation
double p99 = Stats.percentile(samples, 99.0, QuantileMethod.WEIBULL);
```

### Supported Quantile Methods

| `QuantileMethod` Enum | Description | Standard Usage |
|:----------------------|:------------|:---------------|
| `INVERTED_CDF` (R-1)  | Step function, discontinuous at observations | Empirical CDF step |
| `AVERAGED_INVERTED_CDF` (R-2) | Similar to R-1 with average at jumps | Empirical CDF |
| `CLOSEST_OBSERVATION` (R-3) | Nearest even integer order statistic | Nearest observation |
| `CALIFORNIA` (R-4)    | Linear interpolation of empirical CDF | California method ($m=0$) |
| `HAZEN` (R-5)         | Piecewise linear with $m = 0.5$ | Hydrology / Hazen formula |
| `WEIBULL` (R-6)       | $p_k = k / (N + 1)$ | NIST standard, Weibull plotting |
| `LINEAR` (R-7)        | Default NumPy/Excel linear interpolation ($m=1-p$) | **Industry Standard (Default)** |
| `MEDIAN_UNBIASED` (R-8)| Median-unbiased estimate ($p_k = (k - 1/3) / (N + 1/3)$) | Robust analysis |
| `NORMAL_UNBIASED` (R-9)| Approximately normal-unbiased ($p_k = (k - 3/8) / (N + 1/4)$) | Normal distribution quantile |

---

## 3. Higher-Order Moments & Shape

Measure the asymmetry and tail heaviness of a distribution:

```java
// Skewness: 0 for symmetric normal, > 0 for right-skewed tail, < 0 for left-skewed tail
double skew = Stats.skewness(data);

// Kurtosis: Fisher's excess kurtosis (0 for normal distribution, > 0 for heavy tails)
double kurt = Stats.kurtosis(data);
```

---

## 4. Robust Dispersion Metrics

In datasets with heavy outliers (e.g. algorithmic trading, sensor noise), classical variance and standard deviation become distorted. JNum provides robust alternatives:

```java
// Interquartile Range (IQR = Q75 - Q25)
double iqr = Stats.iqr(data);

// Median Absolute Deviation: median(|x_i - median(x)|)
double mad = Stats.mad(data);

// Standard Error of the Mean (SEM = std / sqrt(N))
double sem = Stats.sem(data);
```

---

## 5. Cumulative Operations & Differences

```java
NDArray series = JNum.fromArray(new float[]{ 1f, 3f, 6f, 10f, 15f }, 5);

// Cumulative Sum: [1, 4, 10, 20, 35]
NDArray cumsum = Stats.cumsum(series, 0);

// Cumulative Product: [1, 3, 18, 180, 2700]
NDArray cumprod = Stats.cumprod(series, 0);

// Discrete Difference (series[i+1] - series[i]): [2, 3, 4, 5]
NDArray diff1 = Stats.diff(series, 1, 0);

// Peak-to-Peak (Range = max - min)
double range = Stats.ptp(series);
```
