# Chapter 13: Idiomatic Kotlin DSL

While JNum is written for modern Java 25, it ships with a first-class companion module—**`jnum-kotlin`**—that transforms JNum into an expressive, Pythonic numerical DSL through Kotlin's inline operator overloads, infix functions, and range indexing.

---

## 1. Adding the Kotlin Dependency

In your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("io.github.ch-abhinav:jnum:0.2.0")
    implementation("io.github.ch-abhinav:jnum-kotlin:0.2.0")
}
```

Import the extensions:

```kotlin
import jnum.*
```

---

## 2. Arithmetic Operator Overloading

Standard mathematical operators map directly to JNum's SIMD kernels. All operators are declared `inline` with primitive overloads to eliminate JVM boxing:

```kotlin
fun main() {
    val a = JNum.from(floatArrayOf(1f, 2f, 3f, 4f), 2, 2)
    val b = JNum.from(floatArrayOf(10f, 20f, 30f, 40f), 2, 2)

    // Element-wise binary operations
    val sum  = a + b
    val diff = b - a
    val prod = a * b
    val quot = b / a

    // Scalar arithmetic
    val scaled = a * 2.5
    val offset = a + 10.0f
    
    // Unary negation and boolean inversion
    val negated = -a
    val inverted = !JNum.from(booleanArrayOf(true, false), 2)
}
```

---

## 3. In-Place Augmented Assignment (`+=`, `-=`, `*=`, `/=`)

In Kotlin, augmented assignment operators automatically route to JNum's in-place native methods (`addi`, `subi`, `muli`, `divi`), mutating native memory directly with **zero heap allocations**:

```kotlin
val weights = JNum.rand(1000, 1000)
val delta   = JNum.ones(1000, 1000)

// In-place mutation directly in native memory
weights += delta
weights *= 0.99f // apply decay rate in-place
weights -= 0.001
```

---

## 4. Infix Mathematical Operators

Kotlin's `infix` notation allows writing clean mathematical expressions without parentheses clutter:

### Matrix Multiplication (`matmul`)

```kotlin
val a = JNum.rand(100, 50)
val b = JNum.rand(50, 20)

// Natural infix matrix multiplication:
val c = a matmul b
```

### 1D Vector Dot Product (`dot`)

```kotlin
val u = JNum.from(floatArrayOf(1f, 2f, 3f), 3)
val v = JNum.from(floatArrayOf(4f, 5f, 6f), 3)

val similarity = u dot v
```

### Boolean Infix Logic (`and`, `or`, `xor`)

```kotlin
val mask1 = JNum.from(booleanArrayOf(true, false), 2)
val mask2 = JNum.from(booleanArrayOf(false, true), 2)

val combined = mask1 and mask2
val union    = mask1 or mask2
val diff     = mask1 xor mask2
```

---

## 5. Multidimensional Indexing & Range Slicing

Kotlin's `get` and `set` operator overloads enable square-bracket indexing `arr[...]` and Pythonic range slicing:

### Coordinate Access & Mutation

```kotlin
val matrix = JNum.zeros(5, 5)

// Read element
val value = matrix[1, 2]

// In-place element mutation (zero boxing)
matrix[1, 2] = 42.0
matrix[0, 0] = 3.1415f
```

### Pythonic Range Slicing with `all` and `_`

Use standard Kotlin ranges (`..`, `until`, `step`) alongside the wildcard tokens `all` or `_` (equivalent to Python's `:`):

```kotlin
val tensor = JNum.arange(100.0).reshape(10, 10)

// Equivalent to Python: tensor[:, 0:5]
val leftHalf = tensor[all, 0..4]

// Using the short-hand '_' wildcard
val rightHalf = tensor[`_`, 5..9]

// Strided stepping: tensor[0:10:2, :]
val evenRows = tensor[0..9 step 2, all]

// Open ranges with 'until': tensor[0:5, 2:8]
val subRegion = tensor[0 until 5, 2 until 8]
```

---

## 6. Practical Recipe: Gradient Descent Step in Kotlin

The following recipe demonstrates how clean and readable numerical algorithms look when written in Kotlin with JNum:

```kotlin
import jnum.*

fun main() {
    val batchSize = 128L
    val features = 64L
    val learningRate = 0.01f

    // Initialize state
    val X = JNum.rand(batchSize, features)
    val W = JNum.rand(features, 1)
    val y = JNum.ones(batchSize, 1)

    // Forward pass: y_pred = X * W
    val yPred = X matmul W

    // Error: residuals = y_pred - y
    val error = yPred - y

    // Backward pass: grad = (X^T * error) / batchSize
    val grad = (X.transpose() matmul error) / batchSize.toDouble()

    // In-place parameter update: W -= grad * learningRate
    W -= grad * learningRate

    println("Gradient step complete. Sample weight: ${W[0, 0]}")
}
```

Now proceed to **[Chapter 14: Safety & Error Diagnostics](14-safety-and-error-diagnostics.md)** to learn how JNum ensures memory safety and diagnoses runtime issues.
