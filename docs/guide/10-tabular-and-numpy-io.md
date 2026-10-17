# Chapter 10: Tabular & NumPy I/O

Numerical models rarely exist in isolation. They ingest data exported from Python pipelines (Pandas, NumPy, PyTorch) or database exports, and export processed tensors back for downstream serving.

JNum provides high-speed I/O primitives housed in [`jnum.io.JNumIO`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/JNumIO.java), offering 100% binary interoperability with **CSV/TSV**, **NumPy `.npy` binaries**, and **NumPy `.npz` archives**.

---

## 1. Delimited Text: CSV and TSV

Reading CSV files in traditional Java is notorious for creating millions of intermediate `String` objects, triggering severe GC pauses. JNum uses a high-throughput, off-heap byte parser ([`FastNumberParser`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/internal/common/FastNumberParser.java)) that parses ASCII bytes directly into native memory.

### Configuring with `CsvOptions`

The [`CsvOptions`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/CsvOptions.java) builder provides granular control over parsing:

```java
import java.nio.file.Path;
import jnum.DType;
import jnum.NDArray;
import jnum.io.CsvOptions;
import jnum.io.JNumIO;

void main() throws Exception {
    CsvOptions options = CsvOptions.builder()
        .delimiter(',')          // Field separator (e.g. '\t' for TSV)
        .hasHeader(true)         // Skips the first header row
        .commentMarker("#")      // Ignores comment lines starting with '#'
        .dtype(DType.f32)        // Destination tensor type
        .skipRows(0)             // Extra rows to skip
        .maxRows(100_000)        // Maximum rows to ingest (-1 for all)
        .build();

    // Read directly into an NDArray
    NDArray dataset = JNumIO.readCsv(Path.of("dataset.csv"), options);
    IO.println("Loaded CSV matrix: " + dataset.shapeString());
}
```

### Writing CSV Files

Export tensors to delimited text using custom format strings:

```java
CsvOptions writeOptions = CsvOptions.builder()
    .delimiter(',')
    .formatString("%.4f") // Decimal precision format
    .build();

JNumIO.writeCsv(dataset, writeOptions, Path.of("output.csv"));
```

---

## 2. NumPy `.npy` Binary Format

The `.npy` format is NumPy's native disk format. It contains a lightweight header describing the array's data type, endianness, and shape, followed by raw binary array bytes.

### Synchronous Reading & Writing

```java
import java.nio.file.Path;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.JNumIO;

void main() throws Exception {
    NDArray matrix = JNum.rand(100, 50);

    // Save tensor to .npy file
    Path path = Path.of("weights.npy");
    JNumIO.writeNpy(matrix, path);

    // Read .npy file back into a newly allocated NDArray
    NDArray loaded = JNumIO.readNpy(path);
    IO.println("Loaded shape: " + loaded.shapeString());
}
```

### Zero-Copy Memory Mapping (`mmap`)

For datasets that are gigabytes in size, loading the entire file into RAM causes excessive memory usage. With **memory mapping**, the operating system maps the file directly into virtual address space, paging bytes into RAM on-demand with **zero buffer copying**:

```java
import java.nio.file.Path;
import jnum.io.JNumIO;
import jnum.io.MmapArray;

void main() throws Exception {
    Path hugeFile = Path.of("10gb_embeddings.npy");

    // Scoped memory mapping with try-with-resources
    try (MmapArray mmap = JNumIO.readNpyMmapScoped(hugeFile)) {
        NDArray embeddings = mmap.array();
        
        IO.println("Dataset mapped: " + embeddings.shapeString());
        
        // Read sample vectors instantly via virtual memory page faults
        float val = embeddings.getFloat(0, 0);
        IO.println("Sample element: " + val);
    } // <-- File is deterministically unmapped immediately on close!
}
```

---

## 3. NumPy `.npz` Multi-Tensor Archives

An `.npz` file is a ZIP archive containing multiple `.npy` files keyed by name.

```java
import java.nio.file.Path;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.JNumIO;

void main() throws Exception {
    // 1. Pack multiple tensors into a dictionary
    Map<String, NDArray> modelWeights = Map.of(
        "layer1.weight", JNum.rand(64, 128),
        "layer1.bias",   JNum.zeros(64),
        "layer2.weight", JNum.rand(10, 64)
    );

    // 2. Write archive
    Path archivePath = Path.of("model.npz");
    JNumIO.writeNpz(modelWeights, archivePath);

    // 3. Read archive back into a Map
    Map<String, NDArray> restored = JNumIO.readNpz(archivePath);
    NDArray layer1W = restored.get("layer1.weight");
    IO.println("Restored layer1.weight: " + layer1W.shapeString());
}
```

---

## 4. Asynchronous Non-Blocking I/O

For web servers and asynchronous pipelines, JNum supports non-blocking I/O using Java's `CompletableFuture`:

```java
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import jnum.NDArray;
import jnum.io.JNumIO;

void main() {
    IO.println("Launching async read...");

    CompletableFuture<NDArray> future = JNumIO.readNpyAsync(Path.of("large_matrix.npy"));

    future.thenAccept(array -> {
        IO.println("Async load complete! Shape: " + array.shapeString());
    }).exceptionally(ex -> {
        IO.println("Failed to read: " + ex.getMessage());
        return null;
    });

    // Main thread continues execution without blocking
    future.join();
}
```

---

## 5. Practical Recipe: Python-to-Java Pipeline Verification

The following recipe saves a synthetic dataset in JNum, reads it back via zero-copy memory mapping, and computes numerical mean and variance to verify binary integrity:

```java
import java.nio.file.Path;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.JNumIO;
import jnum.io.MmapArray;
import jnum.stat.Stats;

void main() throws Exception {
    Path tempFile = Path.of("verify_interop.npy");

    // 1. Generate normal numbers in JNum
    NDArray original = JNum.rand(DType.f32, 500, 500);
    double originalMean = Stats.mean(original);

    // 2. Write to standard .npy binary
    JNumIO.writeNpy(original, tempFile);

    // 3. Re-open using zero-copy scoped memory mapping
    try (MmapArray mmap = JNumIO.readNpyMmapScoped(tempFile)) {
        NDArray mapped = mmap.array();
        double mappedMean = Stats.mean(mapped);

        IO.println("Original Mean: " + originalMean);
        IO.println("Mapped Mean:   " + mappedMean);
        IO.println("Difference:    " + Math.abs(originalMean - mappedMean));
    }
}
```

Now proceed to **[Chapter 11: Safetensors & Asynchronous I/O](11-safetensors-and-async-io.md)** to load state-of-the-art Hugging Face neural network checkpoints.
