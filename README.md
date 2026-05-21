# JNum: A Pure Java Numerical Library

JNum is a high-performance numerical computing library built entirely in Java, leveraging the modern **Foreign Function & Memory (FFM) API** and the **Vector API (Project Panama)**.

## Why JNum? (The Pure Java Advantage)

Most existing Java tensor and math libraries (such as ND4J) rely heavily on C++ backends. They execute mathematical operations by calling native code via **JNI (Java Native Interface)** and manipulating memory using `sun.misc.Unsafe`.

This legacy approach introduces several significant problems for modern Java applications:
1. **Deprecation of Unsafe:** The `sun.misc.Unsafe` API is officially scheduled for removal in future Java releases. Java is actively transitioning to the FFM API as the secure, standard alternative.
2. **JNI Overhead:** JNI is notorious for being slow, bloated, and difficult to maintain. Crossing the boundary between Java and native C++ code incurs a measurable performance penalty.
3. **Native Call Latency:** Invoking methods outside the JVM is often slower than keeping the execution within the highly optimized Java environment.

### Our Solution
JNum solves these issues by completely eliminating the C++ backend. By staying 100% pure Java, we avoid the JNI bridge entirely. 

### Memory Footprint and Portability
One of the most massive advantages of using JNum over traditional libraries is the file size and portability.
* **JNum:** Currently under **100 KB** in total size.
* **Traditional Libraries:** Often exceed **100s of MBs** because they are forced to bundle and export native binaries (`.dll` for Windows, `.so` for Linux, `.dylib` for macOS) just to maintain "Write Once, Run Anywhere" compatibility.

## Installation

### System Requirements
* **Java:** Works on Java 25.
* **Build System:** Because JNum utilizes incubating hardware features, you must explicitly configure your build system to enable the Vector API.

### Maven Setup

1. **Install the Binary:** Download the `jnum-0.1.0-PREVIEW.jar` and install it into your local Maven repository (or include it in your project's `lib` folder).
2. **Add the Dependency:** Add the following configuration to your `pom.xml`:

```xml
<dependency>
    <groupId>com.github.ch-abhinav</groupId>
    <artifactId>jnum</artifactId>
    <version>0.1.0-PREVIEW</version>
</dependency>
```

3. **Enable Hardware Acceleration:** You must add the following configuration to the `maven-compiler-plugin` inside your `pom.xml` to allow the JVM to execute the SIMD Vector instructions:

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-compiler-plugin</artifactId>
    <version>3.11.0</version>
    <configuration>
        <compilerArgs>
            <arg>--add-modules</arg>
            <arg>jdk.incubator.vector</arg>
        </compilerArgs>
    </configuration>
</plugin>
```

## Running the Code (Single-File Scripts)

Because JNum relies on incubating hardware APIs, you cannot simply run `java Main.java`. You must explicitly unlock the Vector API and point the Java runtime to the JNum `.jar` file.

To compile and run a single-file program from your terminal, use the following commands:

**On Windows:**
```bash
java --add-modules jdk.incubator.vector -cp ".;jnum-0.1.0-PREVIEW.jar" Main.java
```

**On Linux / macOS:**
```bash
java --add-modules jdk.incubator.vector -cp ".:jnum-0.1.0-PREVIEW.jar" Main.java
```
