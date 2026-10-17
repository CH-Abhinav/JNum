# Chapter 7: Linear Algebra & Decompositions

Linear algebra is the foundation of machine learning, computer vision, robotics, and scientific computing. Modern AI workloads spend more than 90% of their execution time inside General Matrix Multiplication (GEMM).

JNum includes a **pure Java BLAS and LAPACK engine** written without JNI dependencies, delivering near-C/Fortran performance through high-level Panama Vector SIMD kernels, cache-blocked matrix packing, and matrix decomposition algorithms.

---

## 1. Matrix Multiplication (GEMM)

To compute the matrix product of two 2D matrices $A \in \mathbb{R}^{N \times M}$ and $B \in \mathbb{R}^{M \times P}$:

$$C = A \cdot B, \quad C_{i, j} = \sum_{k=0}^{M-1} A_{i, k} B_{k, j}$$

Call [`a.matmul(b)`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/NDArray.java#L1582):

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // Matrix A: 2 rows, 3 columns
    NDArray a = JNum.from(new float[]{
        1, 2, 3,
        4, 5, 6
    }, 2, 3);

    // Matrix B: 3 rows, 2 columns
    NDArray b = JNum.from(new float[]{
        7,  8,
        9, 10,
        11, 12
    }, 3, 2);

    NDArray c = a.matmul(b);
    IO.println("Result (Shape: " + c.shapeString() + "):");
    IO.println(c);
}
```

Output:
```text
Result (Shape: (2, 2)):
[[58.0, 64.0],
 [139.0, 154.0]]
```

### Zero-Allocation Matmul Buffers

For high-frequency inference loops, avoid allocating a new matrix on every call by providing a pre-allocated output buffer:

```java
NDArray out = JNum.zeros(2, 2);

// Writes directly into 'out' with zero memory allocation
a.matmul(b, out);
```

### The 4-Tier GEMM Dispatch Architecture

JNum inspects the matrix dimensions and total arithmetic work volume ($\mathcal{O}(NMP)$) to route computations through one of four hardware-specialized execution tiers:

1. **Tier 0 (Nano Scalar Kernel)**: For tiny matrices ($N, M, P \le 4$), bypasses SIMD overhead and executes directly in CPU scalar registers.
2. **Tier 1 (Direct Register-Blocked SIMD)**: For matrices up to $128 \times 128$, executes a fast, direct SIMD kernel fitting entirely inside CPU L1/L2 caches.
3. **Tier 2 (Single-Threaded BLIS Kernel)**: For medium matrices ($128 < \text{maxDim} \le 256$), packs memory into sequential micro-panels to prevent TLB misses.
4. **Tier 3 (Parallel BLIS Macro-Kernel)**: For large matrices ($> 256$), distributes packed cache tiles across all CPU cores using an optimized ForkJoin worker pool.

*(See [Systems Internals: Pure Java GEMM & SIMD Micro-Kernels](../internals/02-pure-java-gemm-and-simd.md) for architectural proofs).*

---

## 2. Vector Operations & Metrics

### Dot Product (Inner Product)

For 1D vectors, compute the scalar inner product with [`a.dot(b)`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/NDArray.java#L1399):

```java
NDArray u = JNum.from(new float[]{1.0f, 3.0f, -5.0f}, 3);
NDArray v = JNum.from(new float[]{4.0f, -2.0f, -1.0f}, 3);

double dotProduct = u.dot(v); // 1*4 + 3*(-2) + (-5)*(-1) = 3.0
```

### Vector Norms (Euclidean / Frobenius L2)

Compute the Euclidean L2 length of a vector or Frobenius norm of a matrix:

$$\|A\|_F = \sqrt{\sum_{i} \sum_{j} A_{i, j}^2}$$

```java
NDArray vec = JNum.from(new float[]{3.0f, 4.0f}, 2);
double length = vec.norm(); // 5.0
```

### Cosine Similarity

Measure the angular similarity between two normalized directional vectors:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

```java
NDArray docA = JNum.from(new float[]{1.0f, 2.0f, 0.0f}, 3);
NDArray docB = JNum.from(new float[]{2.0f, 4.0f, 0.0f}, 3);

double similarity = docA.cosineSimilarity(docB); // 1.0 (collinear)
```

---

## 3. Matrix Properties & Solvers

### Matrix Diagnostics

```java
NDArray m = JNum.from(new double[]{
    4, 7,
    2, 6
}, 2, 2);

double trace = m.trace(); // 4 + 6 = 10.0
double det   = m.det();   // (4*6) - (7*2) = 10.0
int rank     = m.matrixRank(); // 2
double cond  = m.cond();       // 2-norm condition number
```

### Matrix Inversion (`inv`) and Pseudoinverse (`pinv`)

Compute exact inverses or Moore-Penrose pseudoinverses:

```java
// Exact inverse for non-singular square matrices: A * A^-1 = I
NDArray mInv = m.inv();

// Moore-Penrose pseudoinverse via SVD (for rectangular or ill-conditioned matrices)
NDArray rect = JNum.rand(jnum.DType.f64, 5, 3);
NDArray rectPinv = rect.pinv(); // Shape [3, 5]
```

### Solving Linear Systems ($Ax = b$)

To solve the system of linear equations $Ax = b$ without explicitly inverting $A$:

```java
// Solve:
// 3x + 1y = 9
// 1x + 2y = 8
NDArray A = JNum.from(new double[]{
    3, 1,
    1, 2
}, 2, 2);

NDArray b = JNum.from(new double[]{9, 8}, 2, 1);

NDArray x = A.solve(b);
// Output: x = [2.0, 3.0]
IO.println("Solution x:");
IO.println(x);
```

---

## 4. Canonical Matrix Decompositions

JNum provides four fundamental factorizations:

### A. Cholesky Decomposition

Factorizes a symmetric positive-definite matrix $A$ into $L \cdot L^T$, where $L$ is a lower triangular matrix with positive diagonal entries.

```java
// Covariance or kernel Gram matrix
NDArray symm = JNum.from(new double[]{
    4, 12, -16,
    12, 37, -43,
    -16, -43, 98
}, 3, 3);

NDArray L = symm.cholesky();
IO.println("Lower triangular factor L:");
IO.println(L);
```

### B. QR Factorization

Decomposes matrix $A$ into an orthogonal matrix $Q$ ($Q^T Q = I$) and an upper triangular matrix $R$:

$$A = Q \cdot R$$

```java
NDArray matrix = JNum.rand(jnum.DType.f64, 5, 3);
var qr = matrix.qr();

NDArray Q = qr.q(); // Shape [5, 5] orthogonal
NDArray R = qr.r(); // Shape [5, 3] upper triangular
```

### C. Singular Value Decomposition (SVD)

Decomposes an arbitrary $M \times N$ matrix $A$ into left singular vectors $U$, singular values $S$, and right singular vectors $V^T$:

$$A = U \cdot \Sigma \cdot V^T$$

```java
NDArray data = JNum.rand(jnum.DType.f64, 4, 3);
var svd = data.svd();

NDArray U  = svd.u();  // [4, 4]
NDArray S  = svd.s();  // [3] singular values in descending order
NDArray Vt = svd.vt(); // [3, 3]
```

### D. Eigensystems (`eig` and `eigh`)

Compute eigenvalues and eigenvectors:

```java
// Symmetric matrix
NDArray cov = JNum.from(new double[]{
    2, 1,
    1, 2
}, 2, 2);

var res = cov.eigh();
NDArray eigenvalues  = res.eigenvalues();  // [1.0, 3.0]
NDArray eigenvectors = res.eigenvectors(); // Orthogonal columns
```

---

## 5. Practical Recipe: Ordinary Least Squares (OLS) Regression

The following recipe solves linear regression $y = X \beta$ for optimal parameters $\hat{\beta} = (X^T X)^{-1} X^T y$ using JNum linear algebra:

```java
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

void main() {
    // 5 observations of 2 predictor variables X
    NDArray X = JNum.from(new double[]{
        1, 2,
        2, 1,
        3, 4,
        4, 3,
        5, 5
    }, 5, 2);

    // Target responses y
    NDArray y = JNum.from(new double[]{
        5, 4, 11, 10, 15
    }, 5, 1);

    // Compute Normal Equation: beta = (X^T * X).solve(X^T * y)
    NDArray Xt = X.transpose();
    NDArray XtX = Xt.matmul(X);
    NDArray Xty = Xt.matmul(y);

    NDArray beta = XtX.solve(Xty);

    IO.println("Fitted coefficients beta:");
    IO.println(beta);

    // Generate predictions: y_pred = X * beta
    NDArray yPred = X.matmul(beta);
    NDArray residuals = y.sub(yPred);
    double rss = residuals.dot(residuals);

    IO.println("Residual Sum of Squares (RSS): " + rss);
}
```

Now proceed to **[Chapter 8: Descriptive & Inferential Statistics](08-descriptive-and-inferential-stats.md)** to learn about the `Stats` facade, moments, and quantile algorithms.
