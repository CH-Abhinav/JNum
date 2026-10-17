# Chapter 8: Descriptive & Inferential Statistics

Extracting statistical insights from raw numerical data requires fast, accurate aggregations. From simple summary metrics like mean and variance to non-trivial order statistics like quantiles and mode, statistical calculations must be numerically stable and free of intermediate memory allocations.

JNum houses all statistical algorithms in the dedicated [`jnum.stat.Stats`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/stat/Stats.java) facade.

---

## 1. The `Stats` Facade Architecture

In some libraries, statistical operations are scattered across the core tensor class. JNum keeps `NDArray` lean and focused on linear algebra and memory indexing, while all descriptive and inferential statistics live in `Stats`:

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.stat.Stats;

void main() {
    NDArray data = JNum.from(new float[]{10, 20, 30, 40, 50}, 5);

    // Call through the Stats facade
    double mean = Stats.mean(data);
    double std  = Stats.std(data);
}
```

---

## 2. Measures of Central Tendency

### Arithmetic Mean & Weighted Average

Calculate unweighted means and weighted averages:

$$\mu = \frac{1}{N} \sum_{i=0}^{N-1} x_i, \quad \bar{x}_w = \frac{\sum_{i=0}^{N-1} w_i x_i}{\sum_{i=0}^{N-1} w_i}$$

```java
NDArray scores = JNum.from(new float[]{85, 90, 78, 92}, 4);
NDArray weights = JNum.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 4);

// Unweighted mean
double unweighted = Stats.mean(scores); // 86.25

// Weighted average
double weighted = Stats.average(scores, weights); // 86.7
```

### Multi-Axis Reductions

Calculate means along specific dimensions with optional rank preservation (`keepDims`):

```java
NDArray matrix = JNum.rand(100, 10); // 100 samples, 10 features

// Mean of each feature across all samples (collapses axis 0)
NDArray featureMeans = Stats.mean(matrix, 0); // Shape [10]

// Retain rank for broadcasting: Shape [1, 10]
NDArray broadcastMeans = Stats.mean(matrix, 0, true);
```

### Median and Mode

Order statistics that represent the 50th percentile or the most frequent element:

```java
NDArray values = JNum.from(new float[]{1, 2, 2, 3, 4, 7, 9}, 7);

double median = Stats.median(values); // 3.0
double mode   = Stats.mode(values);   // 2.0
```

---

## 3. Measures of Dispersion

### Variance and Standard Deviation

JNum supports both population ($N$) and sample ($N - 1$) variance using Bessel's correction, as well as arbitrary delta degrees of freedom (`ddof`):

$$\sigma^2 = \frac{1}{N - \text{ddof}} \sum_{i=0}^{N-1} (x_i - \mu)^2$$

```java
NDArray data = JNum.from(new double[]{2, 4, 4, 4, 5, 5, 7, 9}, 8);

// Population variance and standard deviation (ddof = 0)
double popVar = Stats.var(data);
double popStd = Stats.std(data);

// Sample variance and standard deviation (ddof = 1)
double sampleVar = Stats.sampleVar(data);
double sampleStd = Stats.sampleStd(data);

// Custom degrees of freedom (e.g. ddof = 2)
double customVar = Stats.varScalar(data, 2);
```

### Peak-to-Peak (PTP) and Interquartile Range (IQR)

```java
// Range: max(data) - min(data)
double range = Stats.ptp(data); // 9 - 2 = 7.0

// Interquartile Range: 75th percentile - 25th percentile
double iqr = Stats.iqr(data);
```

---

## 4. Quantiles & Percentiles

JNum implements quantile calculations matching NumPy's 5 interpolation methods defined in [`QuantileMethod`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/stat/QuantileMethod.java):

| `QuantileMethod` | Interpolation Strategy |
| :--- | :--- |
| **`LINEAR`** *(Default)* | Linear interpolation between adjacent ranks: $i + (j - i) \times \text{fraction}$. |
| **`LOWER`** | Always takes the lower rank element $i$. |
| **`HIGHER`** | Always takes the higher rank element $j$. |
| **`NEAREST`** | Takes the closest rank index. |
| **`MIDPOINT`** | Computes the arithmetic midpoint $\frac{i + j}{2}$. |

```java
import jnum.stat.QuantileMethod;

NDArray samples = JNum.arange(1.0, 101.0, 1.0); // 1 to 100

// 95th percentile using default LINEAR interpolation
double p95 = Stats.percentile(samples, 95.0, QuantileMethod.LINEAR);

// 25th percentile (Q1) using NEAREST
double q1 = Stats.percentile(samples, 25.0, QuantileMethod.NEAREST);

// Median via quantile (q in [0.0, 1.0])
double q50 = Stats.quantile(samples, 0.50, QuantileMethod.LINEAR);
```

---

## 5. Covariance & Sequential Statistics

### Covariance Matrices

Compute bivariate covariance or full covariance matrices across feature dimensions:

```java
// Bivariate covariance between two vectors
NDArray x = JNum.from(new double[]{1, 2, 3, 4, 5}, 5);
NDArray y = JNum.from(new double[]{2, 4, 5, 4, 5}, 5);
double covXY = Stats.cov(x, y);

// Full Covariance Matrix for an M x N dataset
// rowvar = false indicates columns are variables, rows are observations
NDArray observations = JNum.rand(1000, 4); // 1000 observations of 4 variables
NDArray covMatrix = Stats.cov(observations, false, 1); // 4x4 matrix
```

### Cumulative Reductions & Differences

```java
NDArray seq = JNum.from(new float[]{1, 2, 3, 4}, 4);

// Cumulative sum: [1, 3, 6, 10]
NDArray runningSum = Stats.cumsum(seq, 0);

// Cumulative product: [1, 2, 6, 24]
NDArray runningProd = Stats.cumprod(seq, 0);

// Discrete difference: x[i+1] - x[i] -> [1, 1, 1]
NDArray delta = Stats.diff(seq, 1, 0);
```

---

## 6. Practical Recipe: Z-Score Outlier Filtering

The following recipe computes the Z-score for a stream of observations ($\frac{x - \mu}{\sigma}$) and filters out anomalies that deviate by more than 3 standard deviations:

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.stat.Stats;

void main() {
    // Simulated dataset with an extreme outlier at index 4
    NDArray data = JNum.from(new float[]{
        10.2f, 9.8f, 10.1f, 10.5f, 95.0f, 9.9f, 10.3f, 10.0f
    }, 8);

    double mean = Stats.mean(data);
    double std  = Stats.sampleStd(data);

    IO.println("Dataset Mean: " + mean);
    IO.println("Dataset Std:  " + std);

    // Compute absolute Z-score for every element: |x - mean| / std
    NDArray zScores = data.sub(mean).abs().div(std);

    // Identify indices with Z-score > 3.0
    NDArray isOutlier = zScores.maximum(3.0f).sub(3.0f).abs();

    for (int i = 0; i < data.getSize(); i++) {
        float z = zScores.getFloat(i);
        if (z > 3.0f) {
            IO.println("Outlier detected at index " + i + ": value=" + data.getFloat(i) + " (Z-Score: " + z + ")");
        }
    }
}
```

Now proceed to **[Chapter 9: Symbolic Expression Engine](09-symbolic-expression-engine.md)** to eliminate intermediate memory buffers during complex multi-operator formulas.
