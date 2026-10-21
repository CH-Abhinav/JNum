# Chapter 7: Linear Algebra & Decompositions

Linear algebra is the foundation of machine learning, computer vision, robotics, and scientific computing. Modern AI workloads spend more than 90% of their execution time inside General Matrix Multiplication (GEMM).

JNum includes a **pure Java BLAS and LAPACK engine** written without JNI dependencies, delivering near-C/Fortran performance through high-level Panama Vector SIMD kernels, cache-blocked matrix packing, and matrix decomposition algorithms.

All linear algebra operations are accessible through two equivalent APIs:
1. **NumPy-Style Static Facade**: [`jnum.linalg.Linalg`](../../core/src/main/java/jnum/linalg/Linalg.java) (e.g., `Linalg.matmul(a, b)`, `Linalg.solve(A, b)`, `Linalg.svd(A)`)
2. **Fluent Instance Methods**: [`NDArray`](../../core/src/main/java/jnum/NDArray.java) (e.g., `a.matmul(b)`, `A.solve(b)`, `A.svd()`)

---

## 1. Matrix Multiplication (GEMM)

To compute the matrix product of two 2D matrices $A \in \mathbb{R}^{N \times M}$ and $B \in \mathbb{R}^{M \times P}$:

$$C = A \cdot B, \quad C_{i, j} = \sum_{k=0}^{M-1} A_{i, k} B_{k, j}$$

Call [`a.matmul(b)`](../../core/src/main/java/jnum/NDArray.java#L1634) or [`Linalg.matmul(a, b)`](../../core/src/main/java/jnum/linalg/Linalg.java#L51):

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.linalg.Linalg;

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

    // Fluent method chaining or static Linalg call:
    NDArray c1 = a.matmul(b);
    NDArray c2 = Linalg.matmul(a, b); // Identical result

    IO.println("Result (Shape: " + c1.shapeString() + "):");
    IO.println(c1);
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

// Writes directly into 'out' with zero memory allocation:
a.matmul(b, out);
// Or via static facade:
Linalg.matmul(a, b, out);
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

For 1D vectors, compute the scalar inner product with [`a.dot(b)`](../../core/src/main/java/jnum/NDArray.java#L1427) or [`Linalg.dot(a, b)`](../../core/src/main/java/jnum/linalg/Linalg.java#L78):

```java
NDArray u = JNum.from(new float[]{1.0f, 3.0f, -5.0f}, 3);
NDArray v = JNum.from(new float[]{4.0f, -2.0f, -1.0f}, 3);

// Fluent instance method or static Linalg call:
double dotProduct1 = u.dot(v);
double dotProduct2 = Linalg.dot(u, v); // 1*4 + 3*(-2) + (-5)*(-1) = 3.0
```

### Vector Norms (Euclidean / Frobenius L2)

Compute the Euclidean L2 length of a vector or Frobenius norm of a matrix via [`a.norm()`](../../core/src/main/java/jnum/NDArray.java#L1670) or [`Linalg.norm(a)`](../../core/src/main/java/jnum/linalg/Linalg.java#L128):

$$\|A\|_F = \sqrt{\sum_{i} \sum_{j} A_{i, j}^2}$$

```java
NDArray vec = JNum.from(new float[]{3.0f, 4.0f}, 2);

double length1 = vec.norm();       // 5.0
double length2 = Linalg.norm(vec); // 5.0
```

### Cosine Similarity

Measure the angular similarity between two normalized directional vectors via [`a.cosineSimilarity(b)`](../../core/src/main/java/jnum/NDArray.java#L1903) or [`Linalg.cosineSimilarity(a, b)`](../../core/src/main/java/jnum/linalg/Linalg.java#L91):

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

```java
NDArray docA = JNum.from(new float[]{1.0f, 2.0f, 0.0f}, 3);
NDArray docB = JNum.from(new float[]{2.0f, 4.0f, 0.0f}, 3);

double similarity1 = docA.cosineSimilarity(docB); // 1.0 (collinear)
double similarity2 = Linalg.cosineSimilarity(docA, docB);
```

---

## 3. Matrix Properties & Solvers

### Matrix Diagnostics

```java
NDArray m = JNum.from(new double[]{
    4, 7,
    2, 6
}, 2, 2);

// Fluent on NDArray or static on Linalg:
double trace = Linalg.trace(m);       // or m.trace() -> 4 + 6 = 10.0
double det   = Linalg.det(m);         // or m.det() -> (4*6) - (7*2) = 10.0
int rank     = Linalg.matrixRank(m);   // or m.matrixRank() -> 2
double cond  = Linalg.cond(m);         // or m.cond() -> 2-norm condition number
var slogdet  = Linalg.slogdet(m);      // sign and log-abs-determinant
```

### Matrix Inversion (`inv`) and Pseudoinverse (`pinv`)

Compute exact inverses or Moore-Penrose pseudoinverses:

```java
// Exact inverse for non-singular square matrices: A * A^-1 = I
NDArray mInv1 = m.inv();
NDArray mInv2 = Linalg.inv(m);

// Moore-Penrose pseudoinverse via SVD (for rectangular or ill-conditioned matrices)
NDArray rect = JNum.rand(jnum.DType.f64, 5, 3);
NDArray rectPinv1 = rect.pinv();         // Shape [3, 5]
NDArray rectPinv2 = Linalg.pinv(rect);   // Identical result
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

// Fluent or static:
NDArray x1 = A.solve(b);
NDArray x2 = Linalg.solve(A, b);
// Output: x = [2.0, 3.0]
IO.println("Solution x:");
IO.println(x1);
```

---

## 4. Canonical Matrix Decompositions

JNum provides four fundamental factorizations, supported both via instance methods and static `Linalg` calls:

### A. Cholesky Decomposition

Factorizes a symmetric positive-definite matrix $A$ into $L \cdot L^T$, where $L$ is a lower triangular matrix with positive diagonal entries.

```java
// Covariance or kernel Gram matrix
NDArray symm = JNum.from(new double[]{
    4, 12, -16,
    12, 37, -43,
    -16, -43, 98
}, 3, 3);

// Fluent or static:
NDArray L1 = symm.cholesky();
NDArray L2 = Linalg.cholesky(symm);
IO.println("Lower triangular factor L:");
IO.println(L1);
```

### B. QR Factorization

Decomposes matrix $A$ into an orthogonal matrix $Q$ ($Q^T Q = I$) and an upper triangular matrix $R$:

$$A = Q \cdot R$$

```java
NDArray matrix = JNum.rand(jnum.DType.f64, 5, 3);

// Fluent or static:
var qr1 = matrix.qr();
var qr2 = Linalg.qr(matrix);

NDArray Q = qr1.q(); // Shape [5, 5] orthogonal
NDArray R = qr1.r(); // Shape [5, 3] upper triangular
```

### C. Singular Value Decomposition (SVD)

Decomposes an arbitrary $M \times N$ matrix $A$ into left singular vectors $U$, singular values $S$, and right singular vectors $V^T$:

$$A = U \cdot \Sigma \cdot V^T$$

```java
NDArray data = JNum.rand(jnum.DType.f64, 4, 3);

// Fluent or static:
var svd1 = data.svd();
var svd2 = Linalg.svd(data);

NDArray U  = svd1.u();  // [4, 4]
NDArray S  = svd1.s();  // [3] singular values in descending order
NDArray Vt = svd1.vt(); // [3, 3]
```

### D. Eigensystems (`eig` and `eigh`)

Compute eigenvalues and eigenvectors:

```java
// Symmetric matrix
NDArray cov = JNum.from(new double[]{
    2, 1,
    1, 2
}, 2, 2);

// Fluent or static:
var res1 = cov.eigh();
var res2 = Linalg.eigh(cov);

NDArray eigenvalues  = res1.eigenvalues();  // [1.0, 3.0]
NDArray eigenvectors = res1.eigenvectors(); // Orthogonal columns
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
