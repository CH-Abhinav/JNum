# Getting Started with JNum

Welcome to **JNum**, a high-performance, 100% pure Java numerical computing library built on the modern **Foreign Function & Memory (FFM) API** (`java.lang.foreign`) and the **Vector API** (`jdk.incubator.vector`).

JNum delivers the raw speed of native C/C++ libraries (like NumPy and ND4J) with **zero native dependencies (`.so`/`.dll`/`.dylib`)**, an ultra-lightweight JAR footprint (< 100 KB), and complete JVM safety.

---

## Prerequisites & Runtime Requirements

JNum leverages Project Panama hardware SIMD vector instructions and off-heap memory management.

* **JDK Version**: **Java 25** (or later) with Panama Vector API support.
* **Incubator Flags**: Because the Vector API is an incubating module in OpenJDK, you must enable `jdk.incubator.vector` during compilation and runtime.
* **Native Access**: Safe off-heap memory access requires unlocking native access for the application module or unnamed module.

Verify your Java version:
```bash
java --version
```

---

## Installation & Build Configuration

### Maven Setup

Add the JNum dependencies to your `pom.xml`:

```xml
<dependencies>
    <!-- Core JNum Java Library -->
    <dependency>
        <groupId>io.github.ch-abhinav</groupId>
        <artifactId>jnum</artifactId>
        <version>0.2.0</version>
    </dependency>

    <!-- Optional: Kotlin Idiomatic Extensions & Operator Overloads -->
    <dependency>
        <groupId>io.github.ch-abhinav</groupId>
        <artifactId>jnum-kotlin</artifactId>
        <version>0.2.0</version>
    </dependency>
</dependencies>
```

#### Compiler & Test Plugin Configuration

Configure `maven-compiler-plugin` and `maven-surefire-plugin` to enable hardware vector instructions:

```xml
<build>
    <plugins>
        <!-- Compiler Configuration -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.13.0</version>
            <configuration>
                <release>25</release>
                <compilerArgs>
                    <arg>--add-modules</arg>
                    <arg>jdk.incubator.vector</arg>
                </compilerArgs>
            </configuration>
        </plugin>

        <!-- Test Runner Configuration -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-surefire-plugin</artifactId>
            <version>3.5.3</version>
            <configuration>
                <argLine>--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED</argLine>
            </configuration>
        </plugin>
    </plugins>
</build>
```

---

### Gradle Setup

#### Kotlin DSL (`build.gradle.kts`)

```kotlin
plugins {
    java
    // Optional: id("org.jetbrains.kotlin.jvm") version "2.1.10"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.ch-abhinav:jnum:0.2.0")
    // Optional: implementation("io.github.ch-abhinav:jnum-kotlin:0.2.0")
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<Test> {
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}

tasks.withType<JavaExec> {
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}
```

---

### Running Standalone Single-File Scripts

You can execute single-file Java source files directly without a full build system:

**Windows (PowerShell / Command Prompt):**
```powershell
java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -cp ".;jnum-0.2.0.jar" Main.java
```

**Linux / macOS (Bash / Zsh):**
```bash
java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -cp ".:jnum-0.2.0.jar" Main.java
```

---

## Core Concepts & Quickstart (Java)

### 1. Creating Tensors

All tensor creations start from the `JNum` factory facade:

```java
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

// Zeros and Ones
NDArray zeros = JNum.zeros(2, 3);                 // Shape [2, 3], default FLOAT32
NDArray ones  = JNum.ones(new long[]{3, 3}, DType.FLOAT64); // Double precision [3, 3]

// Numerical Ranges
NDArray range = JNum.arange(0.0f, 10.0f, 2.0f);   // [0.0, 2.0, 4.0, 6.0, 8.0]
NDArray space = JNum.linspace(0.0f, 1.0f, 5);      // 5 evenly spaced values from 0.0 to 1.0

// Identity Matrix
NDArray eye = JNum.eye(4);                         // 4x4 Identity matrix

// From Existing Java Arrays
float[] rawData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
NDArray matrix = JNum.fromArray(rawData, 2, 3);    // 2 rows, 3 columns
```

---

### 2. Element-Wise Math & Vectorized SIMD

Arithmetic operations execute across SIMD lanes (AVX-2, AVX-512, or ARM Neon) automatically:

```java
NDArray a = JNum.fromArray(new float[]{ 1f, 2f, 3f, 4f }, 2, 2);
NDArray b = JNum.fromArray(new float[]{ 10f, 20f, 30f, 40f }, 2, 2);

// Element-wise operations (creates new NDArray)
NDArray sum  = a.add(b);  // [[11, 22], [33, 44]]
NDArray diff = b.sub(a);  // [[9, 18], [27, 36]]
NDArray prod = a.mul(b);  // [[10, 40], [90, 160]]
NDArray quot = b.div(a);  // [[10, 10], [10, 10]]

// Scalar operations
NDArray scaled = a.mul(2.5f);
NDArray shifted = a.add(10.0f);
```

#### Automatic Broadcasting
JNum supports NumPy-style broadcasting across dimensions:
```java
NDArray mat = JNum.ones(3, 4); // Shape [3, 4]
NDArray vec = JNum.arange(4);   // Shape [4] -> broadcasts to [3, 4]

NDArray result = mat.add(vec);  // Works seamlessly with zero extra memory allocation!
```

---

### 3. Linear Algebra & Matrix Multiplication (GEMM)

Matrix multiplication is hardware-accelerated via cache-blocked SIMD routines:

```java
NDArray m1 = JNum.ones(2, 3); // 2x3
NDArray m2 = JNum.ones(3, 4); // 3x4

// Standard GEMM: (2x3) x (3x4) -> (2x4)
NDArray out = m1.matmul(m2);

// Matrix Transpose (Instantaneous O(1) zero-copy view)
NDArray transposed = m1.transpose(); // Shape [3, 2], shares memory buffer!

// 1D Vector Dot Product
NDArray v1 = JNum.arange(5);
NDArray v2 = JNum.arange(5);
double dotProd = v1.dot(v2);
```

---

### 4. Slicing & Zero-Copy Views

Slicing creates **zero-copy views** into the original array buffer. No data is duplicated in memory:

```java
import jnum.Slice;

NDArray x = JNum.arange(12).reshape(3, 4);
// [[ 0,  1,  2,  3],
//  [ 4,  5,  6,  7],
//  [ 8,  9, 10, 11]]

// Extract rows 0 to 2, all columns: x[0:2, :]
NDArray sub = x.slice(Slice.range(0, 2), Slice.all());

// Modifying the view modifies the original parent array!
sub.setFloat(999.0f, 0, 0);
System.out.println(x.getFloat(0, 0)); // Prints 999.0!
```

If you need a completely independent copy that does not share memory, explicitly call `copy()`:
```java
NDArray independent = sub.copy();
```

---

### 5. Reductions & Aggregations

Compute statistics along any axis or across the entire tensor:

```java
NDArray data = JNum.fromArray(new float[]{ 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);

// Full tensor reductions
double total = data.sum();   // 21.0
double avg   = data.mean();  // 3.5
double max   = data.max();   // 6.0
double min   = data.min();   // 1.0

// Axis-specific reductions
NDArray colSums = data.sum(0); // Sum along rows (axis 0) -> Shape [3]
NDArray rowSums = data.sum(1); // Sum along cols (axis 1) -> Shape [2]

// Argmax / Argmin indices
long maxFlatIndex = data.argmax();
```

---

## Zero-Allocation & Performance Best Practices

In performance-critical loops (e.g., real-time inference, high-frequency trading), creating new objects triggers Garbage Collector (GC) pressure. JNum provides two zero-allocation patterns:

### 1. In-Place Mutation (`addi`, `subi`, `muli`, `divi`)
Directly mutates the current off-heap memory buffer:
```java
NDArray weights = JNum.zeros(1000, 1000);
NDArray delta   = JNum.ones(1000, 1000);

// In-place addition: modifies `weights` directly without creating any new object
weights.addi(delta);
```

### 2. Pre-Allocated Output Buffers
Pass a reusable destination buffer to standard operations:
```java
NDArray a = JNum.ones(1000, 1000);
NDArray b = JNum.ones(1000, 1000);
NDArray out = JNum.zeros(1000, 1000); // Pre-allocated buffer

// Execute repeatedly inside your tight loop with 0 bytes/sec GC allocation:
for (int step = 0; step < 100_000; step++) {
    a.add(b, out); // Result is written directly into `out`
}
```

---

## Kotlin Idiomatic Quickstart (`jnum-kotlin`)

For Kotlin developers, `jnum-kotlin` provides full operator overloading, infix functions, and Python-style slice notation:

```kotlin
import jnum.*

fun main() {
    val a = JNum.ones(3, 3)
    val b = JNum.eye(3)

    // Arithmetic Operator Overloads
    val c = a + b
    val d = a * 2.5
    val e = -a

    // In-Place Augmented Assignments
    c += b
    c *= 2.0

    // Infix Linear Algebra
    val matProduct = a matmul b
    val dotProduct = a.slice(Slice.index(0)) dot b.slice(Slice.index(0))

    // Python-Style Indexing & Mutation
    val elem = a[0, 1]          // Get element
    a[0, 1] = 42.0              // Set element

    // Python-Style Slicing with Ranges and Steps
    val subMatrix = a[0..1, 1..2]          // a[0:2, 1:3]
    val everySecond = a[0..10 step 2]      // Strided slice
    val fullCol = a[all, 2]                // a[:, 2] using `all`
}
```

---

## Complete Java Example: `Main.java`

Here is a complete, self-contained modern Java script using an implicitly declared class, instance `main()`, `IO.println`, and `var`:

```java
import java.io.IO;
import jnum.JNum;
import jnum.Slice;

void main() {
    IO.println("=== JNum 0.2.0 Quickstart ===");

    // 1. Create a 2x3 matrix
    var a = JNum.fromArray(new float[]{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, 2, 3);
    IO.println("Matrix A:\n" + a);

    // 2. Perform element-wise operations
    var b = a.mul(2.0f);
    IO.println("Matrix B (A * 2):\n" + b);

    // 3. Matrix Multiplication: (2x3) x (3x2) -> (2x2)
    var bTransposed = b.transpose();
    var product = a.matmul(bTransposed);
    IO.println("A @ B.T:\n" + product);

    // 4. Slicing (Zero-Copy View)
    var row0 = a.slice(Slice.index(0), Slice.all());
    IO.println("Row 0:\n" + row0);

    // 5. Statistics & Reductions
    IO.print("Summary: ");
    IO.println("Sum = " + a.sum() + ", Mean = " + a.mean() + ", Max = " + a.max());
}
```

---

## Next Steps

Now that you have JNum up and running, explore the specialized guides for advanced features:

* **[Architecture & Project Panama Deep Dive](02-architecture-and-panama.md)**: How Foreign Function & Memory (FFM), CPU vector species, and 64-byte cache line alignment power JNum.
* **[Memory Strides & Zero-Copy Views](03-memory-strides-and-views.md)**: Deep dive into C-contiguity, multidimensional strides, and memory layouts.
* **[I/O & Model Checkpoints](04-io-and-safetensors.md)**: Loading Hugging Face Safetensors, NumPy `.npy`/`.npz` files, and zero-copy `mmap`.
* **[Neural Networks & Statistics](05-neural-networks-and-stats.md)**: Building forward-pass inference models with `jnum.nn` and statistical metrics with `jnum.stat`.
* **[NumPy to JNum Cheatsheet](06-numpy-to-jnum-cheatsheet.md)**: Quick syntax comparison for Python developers moving to Java/Kotlin.
