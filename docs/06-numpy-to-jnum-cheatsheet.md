# NumPy to JNum Cheatsheet (Rosetta Stone)

A side-by-side syntax comparison guide for data scientists, ML engineers, and Python/NumPy developers moving to **JNum** in **Java** and **Kotlin**.

---

## 1. Array Creation & Generators

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Zeros** | `np.zeros((2, 3))` | `JNum.zeros(2, 3)` | `JNum.zeros(2, 3)` |
| **Zeros (typed)** | `np.zeros((2, 3), dtype=np.float64)` | `JNum.zeros(new long[]{2, 3}, DType.FLOAT64)` | `JNum.zeros(longArrayOf(2, 3), DType.FLOAT64)` |
| **Ones** | `np.ones((3, 3))` | `JNum.ones(3, 3)` | `JNum.ones(3, 3)` |
| **Range** | `np.arange(0, 10, 2)` | `JNum.arange(0f, 10f, 2f)` | `JNum.arange(0f, 10f, 2f)` |
| **Linspace** | `np.linspace(0, 1, 5)` | `JNum.linspace(0f, 1f, 5)` | `JNum.linspace(0f, 1f, 5)` |
| **Identity Matrix** | `np.eye(4)` | `JNum.eye(4)` | `JNum.eye(4)` |
| **From Array** | `np.array([1, 2, 3, 4]).reshape(2, 2)` | `JNum.fromArray(new float[]{1, 2, 3, 4}, 2, 2)` | `JNum.fromArray(floatArrayOf(1f, 2f, 3f, 4f), 2, 2)` |
| **Random [0, 1)** | `np.random.rand(3, 3)` | `JNum.rand(3, 3)` | `JNum.rand(3, 3)` |
| **Random Bounded** | `np.random.randint(0, 10, (3, 3))` | `JNum.rand(0, 10, DType.INT32, 3, 3)` | `JNum.rand(0, 10, DType.INT32, 3, 3)` |

---

## 2. Shapes, Dimensions & Layouts

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Shape** | `a.shape` | `a.getShape()` | `a.shape` |
| **Number of Dims**| `a.ndim` | `a.dim()` | `a.dim()` |
| **Total Size** | `a.size` | `a.getSize()` | `a.size` |
| **Data Type** | `a.dtype` | `a.getDType()` | `a.dType` |
| **Reshape** | `a.reshape(3, 4)` | `a.reshape(3, 4)` | `a.reshape(3, 4)` |
| **Transpose** | `a.T` or `np.transpose(a)` | `a.transpose()` | `a.transpose()` |
| **Add Axis** | `a[:, np.newaxis]` or `np.expand_dims(a, 1)` | `a.unsqueeze(1)` | `a.unsqueeze(1)` |
| **Remove Axis** | `np.squeeze(a, 0)` | `a.squeeze(0)` | `a.squeeze(0)` |
| **C-Contiguous** | `np.ascontiguousarray(a)` | `a.contiguous()` | `a.contiguous()` |
| **Deep Copy** | `a.copy()` | `a.copy()` | `a.copy()` |

---

## 3. Indexing & Slicing

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Get 1D element** | `val = a[4]` | `float val = a.getFloat(4);` | `val x = a[4]` |
| **Get 2D element** | `val = a[2, 3]` | `float val = a.getFloat(2, 3);` | `val x = a[2, 3]` |
| **Set 2D element** | `a[2, 3] = 42.0` | `a.setFloat(42.0f, 2, 3);` | `a[2, 3] = 42.0` |
| **Slice Range** | `sub = a[0:5]` | `a.slice(Slice.range(0, 5))` | `a[0..4]` |
| **Strided Slice** | `sub = a[0:10:2]` | `a.slice(Slice.range(0, 10, 2))` | `a[0..9 step 2]` |
| **Full Axis Token** | `sub = a[:, 2]` | `a.slice(Slice.all(), Slice.index(2))` | `a[all, 2]` |
| **2D Multi-Slice** | `sub = a[0:2, 1:4]` | `a.slice(Slice.range(0, 2), Slice.range(1, 4))` | `a[0..1, 1..3]` |
| **String Slice** | `sub = a["0:5, :"]` | `a.slice("0:5, :")` | `a.slice("0:5, :")` |
| **Negative Index** | `last = a[-1]` | `a.subview(-1)` | `a.subview(-1)` |

---

## 4. Arithmetic & Math Operations

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Addition** | `a + b` | `a.add(b)` | `a + b` |
| **Subtraction** | `a - b` | `a.sub(b)` | `a - b` |
| **Multiplication**| `a * b` | `a.mul(b)` | `a * b` |
| **Division** | `a / b` | `a.div(b)` | `a / b` |
| **In-Place Add** | `a += b` | `a.addi(b)` | `a += b` |
| **In-Place Mul** | `a *= 2.0` | `a.muli(2.0f)` | `a *= 2.0` |
| **Scalar Add** | `a + 10` | `a.add(10f)` | `a + 10.0` |
| **Negation** | `-a` | `a.mul(-1f)` | `-a` |
| **Square Root** | `np.sqrt(a)` | `a.sqrt()` | `a.sqrt()` |
| **Absolute Value**| `np.abs(a)` | `a.abs()` | `a.abs()` |
| **Exponential** | `np.exp(a)` | `a.exp()` | `a.exp()` |
| **Natural Log** | `np.log(a)` | `a.log()` | `a.log()` |
| **Sine / Cosine** | `np.sin(a)`, `np.cos(a)`| `a.sin()`, `a.cos()` | `a.sin()`, `a.cos()` |
| **Logical AND** | `a & b` | `a.and(b)` | `a and b` |
| **Logical OR** | `a \| b` | `a.or(b)` | `a or b` |
| **Logical NOT** | `~a` | `a.not()` | `!a` |

---

## 5. Linear Algebra & Matrix Multiplication

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Matrix Multiply**| `a @ b` or `np.matmul(a, b)` | `a.matmul(b)` | `a matmul b` |
| **Vector Dot Product** | `np.dot(u, v)` | `u.dot(v)` | `u dot v` |
| **Matrix Inverse**| `np.linalg.inv(a)` | `a.inv()` | `a.inv()` |
| **Solve $Ax = b$**| `np.linalg.solve(a, b)`| `a.solve(b)` | `a.solve(b)` |
| **Determinant** | `np.linalg.det(a)` | `a.det()` | `a.det()` |
| **Log-Determinant**| `np.linalg.slogdet(a)` | `a.slogdet()` | `a.slogdet()` |
| **Matrix Trace** | `np.trace(a)` | `a.trace()` | `a.trace()` |
| **Cholesky ($LL^T$)**| `np.linalg.cholesky(a)`| `a.cholesky()` | `a.cholesky()` |
| **QR ($QR$)** | `q, r = np.linalg.qr(a)`| `var qr = a.qr();` | `val (q, r) = a.qr()` |
| **SVD ($U \Sigma V^T$)**| `u, s, vt = np.linalg.svd(a)`| `var svd = a.svd();` | `val (u, s, vt) = a.svd()` |
| **Eigenvalues** | `w, v = np.linalg.eig(a)`| `var eig = a.eig();` | `val (real, imag) = a.eig()` |
| **Symmetric Eig** | `w, v = np.linalg.eigh(a)`| `var eigh = a.eigh();` | `val (w, v) = a.eigh()` |
| **Matrix Norm** | `np.linalg.norm(a)` | `a.norm(2)` | `a.norm(2)` |
| **Matrix Rank** | `np.linalg.matrix_rank(a)`| `a.matrixRank()` | `a.matrixRank()` |
| **Pseudoinverse**| `np.linalg.pinv(a)` | `a.pinv()` | `a.pinv()` |

---

## 6. Reductions & Statistics

| Operation | Python (NumPy) | Java (JNum) | Kotlin (JNum) |
|:----------|:---------------|:------------|:--------------|
| **Sum** | `a.sum()` | `a.sum()` | `a.sum()` |
| **Sum Axis** | `a.sum(axis=0)` | `a.sum(0)` | `a.sum(0)` |
| **Mean** | `np.mean(a)` | `Stats.mean(a)` or `a.mean()` | `Stats.mean(a)` |
| **Standard Deviation**| `np.std(a)` | `Stats.std(a)` | `Stats.std(a)` |
| **Variance** | `np.var(a)` | `Stats.var(a)` | `Stats.var(a)` |
| **Median** | `np.median(a)` | `Stats.median(a)` | `Stats.median(a)` |
| **Quantile** | `np.quantile(a, 0.75)` | `Stats.quantile(a, 0.75)` | `Stats.quantile(a, 0.75)` |
| **Percentile** | `np.percentile(a, 90)` | `Stats.percentile(a, 90.0)` | `Stats.percentile(a, 90.0)` |
| **Minimum / Maximum**| `a.min()`, `a.max()` | `a.min()`, `a.max()` | `a.min()`, `a.max()` |
| **Argmax / Argmin**| `np.argmax(a)`, `np.argmin(a)` | `a.argmax()`, `a.argmin()` | `a.argmax()`, `a.argmin()` |
| **Cumulative Sum**| `np.cumsum(a, axis=0)` | `Stats.cumsum(a, 0)` | `Stats.cumsum(a, 0)` |
| **Difference** | `np.diff(a)` | `Stats.diff(a, 1, 0)` | `Stats.diff(a, 1, 0)` |

---

## 7. Model I/O & Checkpoints

| Operation | Python (Ecosystem) | Java (JNum) |
|:----------|:-------------------|:------------|
| **Load Safetensors** | `from safetensors import safe_open` | `JNumIO.readSafetensors(path)` |
| **Mmap Safetensors** | `safe_open("m.st", framework="pt")` | `JNumIO.readSafetensorsMmap(path)` |
| **Save Safetensors** | `save_file(tensors, "m.st")` | `JNumIO.writeSafetensors(path, tensors)` |
| **Save NumPy `.npy`**| `np.save("a.npy", a)` | `JNumIO.writeNpy(path, a)` |
| **Load NumPy `.npy`**| `a = np.load("a.npy")` | `JNumIO.readNpy(path)` |
| **Mmap NumPy `.npy`**| `np.load("a.npy", mmap_mode="r")`| `JNumIO.readNpyMmap(path)` |
| **Load NumPy `.npz`**| `archive = np.load("m.npz")` | `JNumIO.readNpz(path)` |
| **Save NumPy `.npz`**| `np.savez_compressed("m.npz", **d)`| `JNumIO.writeNpz(path, map, true)` |
| **Read CSV** | `np.loadtxt("d.csv", delimiter=",")`| `JNumIO.readCsv(path, options)` |
