# Chapter 1: Getting Started

Numerical computing on the JVM has traditionally forced developers to make a painful compromise: write slow, garbage-collected code using nested primitive arrays (`float[][]`), or rely on native C/C++ libraries (such as OpenBLAS or Intel MKL) bridged via Java Native Interface (JNI). JNI bridges introduce severe JIT deoptimization boundaries, GC pinning stalls, and complex platform-dependent build toolchains.

**JNum** eliminates this trade-off. Written in pure Java 25 and Kotlin, JNum leverages **Project Panama**—specifically the **Foreign Function & Memory (FFM) API** (JEP 454) and the **Vector API** (JEP 448)—to execute high-throughput SIMD vector operations and off-heap memory management with zero native compilation steps.

---

## 1. Prerequisites

To build and run applications with JNum, you need:
* **Java Development Kit (JDK) 25 or later** (such as Eclipse Temurin, Oracle OpenJDK, or Azul Zulu).
* An architecture supporting hardware SIMD vector extensions:
  * **x86_64 / AMD64**: SSE4.2, AVX, AVX2, or AVX-512.
  * **AArch64 / ARM64**: ARM NEON or SVE (e.g., Apple Silicon M-series, AWS Graviton).
  * **RISC-V**: Vector extension (RVV).

Check your installed JDK version:
```bash
java -version
```

---

## 2. Dependency Coordinates

Add JNum to your build configuration. For Kotlin projects, `jnum-kotlin` bundles the core JNum runtime along with Kotlin operator overloads and DSL extensions.

### Maven (`pom.xml`)

**For Java:**
```xml
<dependency>
    <groupId>io.github.ch-abhinav</groupId>
    <artifactId>jnum</artifactId>
    <version>0.2.0</version>
</dependency>
```

**For Kotlin** *(bundles core + Kotlin extensions)*:
```xml
<dependency>
    <groupId>io.github.ch-abhinav</groupId>
    <artifactId>jnum-kotlin</artifactId>
    <version>0.2.0</version>
</dependency>
```

### Gradle (`build.gradle.kts`)

**For Java:**
```kotlin
dependencies {
    implementation("io.github.ch-abhinav:jnum:0.2.0")
}
```

**For Kotlin** *(bundles core + Kotlin extensions)*:
```kotlin
dependencies {
    implementation("io.github.ch-abhinav:jnum-kotlin:0.2.0")
}
```

---

## 3. Required JVM Runtime Flags

Because JNum leverages Project Panama features, two specific JVM flags must be provided at compile-time and runtime:

| JVM Flag | Purpose |
| :--- | :--- |
| `--add-modules jdk.incubator.vector` | Resolves and enables the hardware SIMD Vector API module. |
| `--enable-native-access=ALL-UNNAMED` | Permits off-heap memory allocations and zero-copy memory mapping via Foreign Function & Memory (FFM) without terminal security warnings. |

> [!IMPORTANT]
> If `--add-modules jdk.incubator.vector` is omitted, the JVM will fail at startup with `java.lang.NoClassDefFoundError: jdk/incubator/vector/FloatVector`.

---

## 4. Build Tool Configuration

### Configuring Maven

Add compiler and surefire arguments to your `pom.xml` so that builds and unit tests execute seamlessly:

```xml
<build>
    <plugins>
        <!-- Java Compiler Configuration -->
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

        <!-- Test Execution Configuration -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-surefire-plugin</artifactId>
            <version>3.2.5</version>
            <configuration>
                <argLine>
                    --add-modules jdk.incubator.vector
                    --enable-native-access=ALL-UNNAMED
                </argLine>
            </configuration>
        </plugin>
    </plugins>
</build>
```

### Configuring Gradle

In your `build.gradle.kts`:

```kotlin
plugins {
    java
    application
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "--add-modules", "jdk.incubator.vector"
    ))
}

tasks.withType<Test> {
    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
}

application {
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
}
```

---

## 5. Single-File Script Execution

Modern Java allows executing single `.java` source files directly without configuring a build tool.

Create a file named `VerifyJNum.java`:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    // Allocate a 3x3 matrix populated with ones
    NDArray matrix = JNum.ones(3, 3);
    
    // Scale matrix by scalar 5.0
    NDArray scaled = matrix.mul(5.0f);
    
    IO.println("Successfully initialized JNum tensor:");
    IO.println(scaled);
}
```

Execute it directly from your terminal:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     VerifyJNum.java
```

Output:
```text
Successfully initialized JNum tensor:
[[5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0]]
```

---

## 6. Verifying Hardware Vector Acceleration

JNum automatically utilizes your CPU's hardware SIMD vector extensions. You can verify your machine's preferred vector species and concurrent lane capacity using the standard Vector API:

```java
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.JNum;
import jnum.NDArray;

void main() {
    VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
    
    IO.println("OS: " + System.getProperty("os.name") + " (" + System.getProperty("os.arch") + ")");
    IO.println("Available CPU Cores: " + Runtime.getRuntime().availableProcessors());
    IO.println("SIMD Preferred Vector Bit-Width: " + species.vectorBitSize() + " bits");
    IO.println("Simultaneous Float32 Lanes: " + species.length());
    
    // Test basic SIMD execution
    NDArray a = JNum.ones(species.length());
    NDArray b = a.add(a);
    IO.println("SIMD Vector Test Result: " + b);
}
```

If your CPU supports **AVX2**, you will observe 256-bit vectors (8 concurrent `float32` lanes). On **AVX-512**, you will observe 512-bit vectors (16 concurrent `float32` lanes). On **Apple Silicon (NEON)**, you will observe 128-bit vectors (4 concurrent lanes).

> [!NOTE]
> JNum's public API surface is strictly encapsulated under `jnum`, `jnum.nn`, `jnum.io`, and `jnum.stat`. All internal implementation details (such as cache probing and low-level kernels under `jnum.internal.*`) are encapsulated by the Java Module System (`module-info.java`) and are not exposed to client applications.

With your environment configured, proceed to **[Chapter 2: Tensors & Fundamentals](02-tensors-and-fundamentals.md)** to learn how off-heap memory, data types, and tensor lifecycles work.

