# Reference: NumPy to JNum Rosetta Stone

This reference provides a side-by-side comparison of common numerical operations across **NumPy (Python)**, **JNum (Java 25)**, and **JNum (Kotlin DSL)**.

---

## 1. Array Creation & Factories

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Zeros** | `np.zeros((3, 4))` | `JNum.zeros(3, 4)` | `JNum.zeros(3, 4)` |
| **Ones** | `np.ones((2, 3), dtype=np.float32)` | `JNum.ones(DType.f32, 2, 3)` | `JNum.ones(DType.f32, 2, 3)` |
| **Identity Matrix** | `np.eye(4)` | `JNum.eye(4)` | `JNum.eye(4)` |
| **Range Sequence** | `np.arange(0, 10, 2)` | `JNum.arange(0, 10, 2)` | `JNum.arange(0.0, 10.0, 2.0)` |
| **Uniform Random** | `np.random.rand(3, 3)` | `JNum.rand(3, 3)` | `JNum.rand(3, 3)` |
| **Bounded Random** | `np.random.uniform(-1, 1, (4, 4))` | `JNum.rand(-1.0f, 1.0f, DType.f32, 4, 4)` | `JNum.rand(-1.0f, 1.0f, DType.f32, 4, 4)` |
| **From Array** | `np.array([1.0, 2.0, 3.0])` | `JNum.from(new float[]{1, 2, 3}, 3)` | `JNum.from(floatArrayOf(1f, 2f, 3f), 3)` |

---

## 2. Shape & Layout Manipulations

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Dimensions / Shape**| `a.shape` | `a.getShape()` | `a.shape` |
| **Total Elements** | `a.size` | `a.getSize()` | `a.size` |
| **Reshape** | `a.reshape((2, 6))` | `a.reshape(2, 6)` | `a.reshape(2, 6)` |
| **Transpose** | `a.T` or `a.transpose()` | `a.transpose()` | `a.transpose()` |
| **Remove Axis (1)** | `np.squeeze(a, axis=0)` | `a.squeeze(0)` | `a.squeeze(0)` |
| **Add Axis (1)** | `np.expand_dims(a, axis=1)` | `a.unsqueeze(1)` | `a.unsqueeze(1)` |
| **Flatten to 1D** | `a.flatten()` | `a.reshape(a.getSize())` | `a.reshape(a.size)` |
| **Force Contiguous** | `np.ascontiguousarray(a)` | `a.contiguous()` | `a.contiguous()` |

---

## 3. Indexing & Slicing

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Get Element** | `val = a[1, 2]` | `a.getDouble(1, 2)` or `a.getFloat(1, 2)` | `val = a[1, 2]` |
| **Set Element** | `a[1, 2] = 42.0` | `a.setDouble(42.0, 1, 2)` | `a[1, 2] = 42.0` |
| **Leading Subview** | `row = a[0]` | `a.subview(0)` | `a.subview(0)` |
| **Range Slice** | `a[1:4, :]` | `a.slice(Slice.range(1, 4), Slice.all())` | `a[1..3, all]` |
| **Short-hand Slice** | `a[:, 0:5]` | `a.slice(":, 0:5")` | `a[`_`, 0 until 5]` |
| **Strided Stepping** | `a[::2, :]` | `a.slice(Slice.step(2), Slice.all())` | `a[0..9 step 2, all]` |
| **Reverse Dimension**| `a[::-1, :]` | `a.slice(Slice.step(-1), Slice.all())` | `a.slice(Slice.step(-1), Slice.all())` |

---

## 4. Element-Wise Arithmetic & In-Place Mutation

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Addition** | `a + b` | `a.add(b)` | `a + b` |
| **Subtraction** | `a - b` | `a.sub(b)` | `a - b` |
| **Multiplication** | `a * b` | `a.mul(b)` | `a * b` |
| **Division** | `a / b` | `a.div(b)` | `a / b` |
| **Scalar Add** | `a + 10.0` | `a.add(10.0)` | `a + 10.0` |
| **In-Place Add** | `a += b` | `a.addi(b)` | `a += b` |
| **In-Place Scale** | `a *= 0.5` | `a.muli(0.5f)` | `a *= 0.5f` |
| **Pre-Allocated Dest**| `np.add(a, b, out=c)` | `a.add(b, c)` | `a.add(b, c)` |

---

## 5. Mathematical & Transcendental Functions

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Square Root** | `np.sqrt(a)` | `a.sqrt()` or `JNum.sqrt(a)` | `a.sqrt()` or `JNum.sqrt(a)` |
| **Absolute Value** | `np.abs(a)` | `a.abs()` or `JNum.abs(a)` | `a.abs()` or `JNum.abs(a)` |
| **Exponential** | `np.exp(a)` | `a.exp()` or `JNum.exp(a)` | `a.exp()` or `JNum.exp(a)` |
| **Natural Log** | `np.log(a)` | `a.log()` or `JNum.log(a)` | `a.log()` or `JNum.log(a)` |
| **Base-10 Log** | `np.log10(a)` | `a.log10()` or `JNum.log10(a)` | `a.log10()` or `JNum.log10(a)` |
| **Sine** | `np.sin(a)` | `a.sin()` or `JNum.sin(a)` | `a.sin()` or `JNum.sin(a)` |
| **Cosine** | `np.cos(a)` | `a.cos()` or `JNum.cos(a)` | `a.cos()` or `JNum.cos(a)` |
| **Tangent** | `np.tan(a)` | `a.tan()` or `JNum.tan(a)` | `a.tan()` or `JNum.tan(a)` |
| **Hyperbolic Sine** | `np.sinh(a)` | `a.sinh()` or `JNum.sinh(a)` | `a.sinh()` or `JNum.sinh(a)` |
| **Hyperbolic Cosine** | `np.cosh(a)` | `a.cosh()` or `JNum.cosh(a)` | `a.cosh()` or `JNum.cosh(a)` |
| **Hyperbolic Tanh** | `np.tanh(a)` | `a.tanh()` or `JNum.tanh(a)` | `a.tanh()` or `JNum.tanh(a)` |
| **Sigmoid Activation**| `1 / (1 + np.exp(-a))` | `a.sigmoid()` or `JNum.sigmoid(a)` | `a.sigmoid()` or `JNum.sigmoid(a)` |

---

## 6. Comparisons, Clamping & Boolean Logic

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Element Maximum** | `np.maximum(a, b)` | `a.maximum(b)` | `a.maximum(b)` |
| **Element Minimum** | `np.minimum(a, b)` | `a.minimum(b)` | `a.minimum(b)` |
| **Range Clamping** | `np.clip(a, -1.0, 1.0)` | `a.clamp(-1.0f, 1.0f)` | `a.clamp(-1.0f, 1.0f)` |
| **Scalar Clip / ReLU**| `np.maximum(a, 0)` | `a.maximum(0.0f)` | `a.maximum(0.0f)` |
| **Logical AND** | `a & b` | `a.and(b)` | `a and b` |
| **Logical OR** | `a \| b` | `a.or(b)` | `a or b` |
| **Logical XOR** | `a ^ b` | `a.xor(b)` | `a xor b` |
| **Logical NOT** | `~a` | `a.not()` | `!a` |
| **All Non-Zero** | `np.all(a)` | `a.all()` | `a.all()` |
| **Any Non-Zero** | `np.any(a)` | `a.any()` | `a.any()` |

---

## 7. Linear Algebra & Decompositions

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Matrix Multiply** | `a @ b` or `np.matmul(a, b)` | `a.matmul(b)` or `Linalg.matmul(a, b)` | `a matmul b` or `Linalg.matmul(a, b)` |
| **1D Dot Product** | `np.dot(u, v)` | `u.dot(v)` or `Linalg.dot(u, v)` | `u dot v` or `Linalg.dot(u, v)` |
| **Euclidean Norm** | `np.linalg.norm(a)` | `a.norm()` or `Linalg.norm(a)` | `a.norm()` or `Linalg.norm(a)` |
| **Matrix Trace** | `np.trace(a)` | `a.trace()` or `Linalg.trace(a)` | `a.trace()` or `Linalg.trace(a)` |
| **Determinant** | `np.linalg.det(a)` | `a.det()` or `Linalg.det(a)` | `a.det()` or `Linalg.det(a)` |
| **Matrix Rank** | `np.linalg.matrix_rank(a)`| `a.matrixRank()` or `Linalg.matrixRank(a)` | `a.matrixRank()` or `Linalg.matrixRank(a)` |
| **Condition Number** | `np.linalg.cond(a)` | `a.cond()` or `Linalg.cond(a)` | `a.cond()` or `Linalg.cond(a)` |
| **Matrix Inverse** | `np.linalg.inv(a)` | `a.inv()` or `Linalg.inv(a)` | `a.inv()` or `Linalg.inv(a)` |
| **Solve ($Ax=b$)** | `np.linalg.solve(A, b)` | `A.solve(b)` or `Linalg.solve(A, b)` | `A.solve(b)` or `Linalg.solve(A, b)` |
| **Pseudoinverse** | `np.linalg.pinv(a)` | `a.pinv()` or `Linalg.pinv(a)` | `a.pinv()` or `Linalg.pinv(a)` |
| **Cholesky** | `np.linalg.cholesky(a)` | `a.cholesky()` or `Linalg.cholesky(a)` | `a.cholesky()` or `Linalg.cholesky(a)` |
| **QR Factorization** | `q, r = np.linalg.qr(a)` | `a.qr()` or `Linalg.qr(a)` | `val (q, r) = a.qr()` |
| **SVD** | `u, s, vt = np.linalg.svd(a)` | `a.svd()` or `Linalg.svd(a)` | `val (u, s, vt) = a.svd()` |
| **Eigenvalues** | `w, v = np.linalg.eig(a)` | `a.eig()` or `Linalg.eig(a)` | `val (w, v) = a.eig()` |

---

## 8. Descriptive & Inferential Statistics

| Operation | NumPy (Python) | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Mean** | `np.mean(a)` | `Stats.mean(a)` | `Stats.mean(a)` |
| **Mean along Axis** | `np.mean(a, axis=0)` | `Stats.mean(a, 0)` | `Stats.mean(a, 0)` |
| **Weighted Average** | `np.average(a, weights=w)`| `Stats.average(a, w)` | `Stats.average(a, w)` |
| **Median** | `np.median(a)` | `Stats.median(a)` | `Stats.median(a)` |
| **Variance (Sample)**| `np.var(a, ddof=1)` | `Stats.sampleVar(a)` | `Stats.sampleVar(a)` |
| **Standard Deviation**| `np.std(a, ddof=1)` | `Stats.sampleStd(a)` | `Stats.sampleStd(a)` |
| **Peak-to-Peak** | `np.ptp(a)` | `Stats.ptp(a)` | `Stats.ptp(a)` |
| **Percentile (95%)** | `np.percentile(a, 95)` | `Stats.percentile(a, 95.0, QuantileMethod.LINEAR)` | `Stats.percentile(a, 95.0, QuantileMethod.LINEAR)` |
| **Covariance Matrix**| `np.cov(m)` | `Stats.cov(m, false, 1)` | `Stats.cov(m, false, 1)` |
| **Cumulative Sum** | `np.cumsum(a, axis=0)` | `Stats.cumsum(a, 0)` | `Stats.cumsum(a, 0)` |
| **Discrete Diff** | `np.diff(a, n=1, axis=0)` | `Stats.diff(a, 1, 0)` | `Stats.diff(a, 1, 0)` |

---

## 9. Symbolic Expression Evaluation

| Operation | Python / NumExpr | JNum (Java 25) | JNum (Kotlin) |
| :--- | :--- | :--- | :--- |
| **Positional Expr** | `ne.evaluate("a * b + c")` | `JNum.eval("$0 * $1 + $2", a, b, c)` | `JNum.eval("$0 * $1 + $2", a, b, c)` |
| **Named Variable Expr** | `ne.evaluate("x * w + b")` | `JNum.eval("x * w + b", Map.of("x", x, "w", w, "b", b))` | `JNum.eval("x * w + b", mapOf("x" to x, "w" to w, "b" to b))` |

---

## 10. File I/O & Model Checkpoints

| Operation | NumPy / Hugging Face (Python) | JNum (Java 25 & Kotlin) |
| :--- | :--- | :--- |
| **Save `.npy`** | `np.save("file.npy", arr)` | `JNumIO.writeNpy(arr, Path.of("file.npy"))` |
| **Load `.npy`** | `arr = np.load("file.npy")` | `NDArray arr = JNumIO.readNpy(Path.of("file.npy"))` |
| **Memory-Map `.npy`**| `arr = np.load("file.npy", mmap_mode="r")` | `try (MmapArray m = JNumIO.readNpyMmapScoped(p)) { NDArray a = m.array(); }` |
| **Save `.npz`** | `np.savez("archive.npz", **tensors)` | `JNumIO.writeNpz(tensors, Path.of("archive.npz"))` |
| **Load `.npz`** | `archive = np.load("archive.npz")` | `Map<String, NDArray> tensors = JNumIO.readNpz(Path.of("archive.npz"))` |
| **Read Safetensors**| `from safetensors.numpy import load_file; load_file("model.safetensors")` | `Map<String, NDArray> weights = JNumIO.readSafetensors(Path.of("model.safetensors"))` |
| **Mmap Safetensors**| `from safetensors import safe_open; safe_open("model.safetensors", framework="numpy")` | `try (MmapTensors m = JNumIO.readSafetensorsMmapScoped(p)) { ... }` |
| **Read CSV** | `pd.read_csv("data.csv")` | `JNumIO.readCsv(Path.of("data.csv"), CsvOptions.DEFAULT)` |
