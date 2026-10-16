# Model I/O & Safetensors Guide

JNum provides high-performance, 100% pure Java serialization and deserialization routines built directly on the **Foreign Function & Memory (FFM) API**.

JNum natively inter-operates with the modern deep learning and Python scientific ecosystem:
* **Hugging Face Safetensors (`.safetensors`)**: Zero-overhead loading of modern transformer models (Llama, Mistral, BERT, Whisper) with metadata inspection and zero-copy memory mapping (`mmap`).
* **NumPy (`.npy` & `.npz`)**: Binary compatibility with NumPy arrays and multi-array archives.
* **Delimited Text (CSV / TSV)**: High-throughput, zero-garbage tabular text parsing and serialization.
* **Asynchronous I/O**: Non-blocking file operations returning `CompletableFuture`.

All file operations are accessed through the [`JNumIO`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/JNumIO.java) facade in the `jnum.io` package.

---

## 1. Hugging Face Safetensors

The [Safetensors](https://github.com/huggingface/safetensors) format is the open standard for storing neural network weights safely. Unlike Python's `pickle` format, Safetensors is completely immune to arbitrary code execution vulnerabilities and allows instant zero-copy loading.

### Safetensors Binary Architecture
A `.safetensors` file is structured into three continuous segments:
1. **Header Size (8 bytes)**: An unsigned 64-bit little-endian integer specifying the length $N$ of the JSON header.
2. **JSON Header ($N$ bytes)**: UTF-8 JSON text containing tensor names, data types, shapes, and byte offsets `[start, end]`.
3. **Raw Binary Data**: Contiguous, 64-byte aligned tensor memory buffers.

```
+------------------+------------------------------------+---------------------------------------+
| Header Size (8B) | JSON Metadata Header (N bytes)     | Raw Tensor Buffers (Aligned)          |
| 0x00000000000004A0 | {"weight": {"dtype": "F32", ...}}  | [ 1.0f, 2.0f, 3.0f, ... ]             |
+------------------+------------------------------------+---------------------------------------+
```

---

### Reading Safetensors into Memory

Load all tensors from a checkpoint file into a `Map<String, NDArray>`:

```java
import java.nio.file.Path;
import java.util.Map;
import jnum.NDArray;
import jnum.io.JNumIO;

// Reads all tensors using an automatic garbage-collected Arena
Map<String, NDArray> weights = JNumIO.readSafetensors(Path.of("model.safetensors"));

NDArray embeddings = weights.get("model.embed_tokens.weight");
System.out.println("Embeddings shape: " + embeddings.shapeString());
```

#### Selective Loading
If your model checkpoint is 20 GB and you only need specific layers, read the tensors selectively to conserve RAM:

```java
import java.util.Set;

Set<String> layersToLoad = Set.of(
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight"
);

Map<String, NDArray> selected = JNumIO.readSafetensors(
    Path.of("model.safetensors"), 
    layersToLoad
);
```

---

### Zero-Copy Memory Mapping (`mmap`)

For massive checkpoints (10 GB to 70 GB+), loading weights directly into physical RAM can trigger `OutOfMemoryError` or heavy GC pressure.

JNum's [`readSafetensorsMmap`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/JNumIO.java) maps the file directly into the operating system's virtual memory address space (`FileChannel.MapMode.READ_ONLY`):

```java
import java.nio.file.Path;
import jnum.io.JNumIO;
import jnum.io.MmapTensors;

// Opens memory-mapped tensors with deterministic lifecycle management
try (MmapTensors mmap = JNumIO.readSafetensorsMmap(Path.of("llama-3-8b.safetensors"))) {
    // Access tensor header metadata without reading any data bytes
    System.out.println("Tensors in checkpoint: " + mmap.tensorNames());

    // Access specific tensor: memory pages are lazily paged in by the OS kernel
    NDArray qProj = mmap.get("model.layers.0.self_attn.q_proj.weight");
    System.out.println("Loaded tensor shape: " + qProj.shapeString());

    // Perform inference or forward GEMM
    // ...
} // Closing unmaps all virtual memory pages deterministically!
```

> [!NOTE]
> When `mmap.close()` is invoked, JNum cleanly closes the underlying `Arena`, releasing the OS virtual memory mapping immediately without waiting for Java GC finalization and without any risk of JVM segmentation faults.

---

### Header & Metadata Inspection

Inspect model metadata and tensor descriptions without reading any raw weight buffers:

```java
import java.nio.file.Path;
import jnum.io.JNumIO;
import jnum.io.SafetensorsInfo;

SafetensorsInfo info = JNumIO.readSafetensorsInfo(Path.of("model.safetensors"));

// Access custom user metadata (e.g. Hugging Face training metadata)
System.out.println("Format: " + info.metadata().get("format"));

// Inspect tensor shapes and data types
for (var entry : info.tensors().entrySet()) {
    SafetensorsInfo.TensorDesc desc = entry.getValue();
    System.out.printf("Tensor: %s | DType: %s | Shape: %s\n", 
        entry.getKey(), desc.dtype(), desc.shapeString());
}
```

---

### Writing Safetensors

Save trained weights or intermediate activations to a `.safetensors` file:

```java
import java.nio.file.Path;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.JNumIO;

Map<String, NDArray> tensors = Map.of(
    "linear.weight", JNum.fromArray(new float[]{ 1f, 2f, 3f, 4f }, 2, 2),
    "linear.bias",   JNum.fromArray(new float[]{ 0.1f, 0.2f }, 2)
);

Map<String, String> userMetadata = Map.of(
    "author", "JNum Team",
    "framework", "JNum 0.2.0"
);

JNumIO.writeSafetensors(Path.of("output.safetensors"), tensors, userMetadata);
```

---

## 2. NumPy Interoperability (`.npy` & `.npz`)

JNum provides 100% binary-level compatibility with NumPy format versions 1.0 and 2.0.

### Single Array Files (`.npy`)

#### Reading `.npy`
```java
import java.nio.file.Path;
import jnum.NDArray;
import jnum.io.JNumIO;

NDArray array = JNumIO.readNpy(Path.of("dataset.npy"));
System.out.println("Loaded array: " + array.shapeString() + " " + array.getDType());
```

#### Writing `.npy`
```java
NDArray matrix = JNum.linspace(0f, 10f, 100).reshape(10, 10);
JNumIO.writeNpy(Path.of("matrix.npy"), matrix);
```

#### Memory-Mapped `.npy` Arrays (`MmapArray`)
For large datasets exceeding available heap RAM:
```java
import jnum.io.MmapArray;

try (MmapArray mmap = JNumIO.readNpyMmap(Path.of("large_dataset.npy"))) {
    NDArray arr = mmap.array();
    
    // Read only a slice: OS only loads the requested 4KB memory pages
    NDArray row0 = arr.slice(Slice.index(0), Slice.all());
    System.out.println("First row sum: " + row0.sum());
}
```

---

### Archive Files (`.npz`)

NumPy `.npz` files are ZIP containers holding multiple named `.npy` arrays. JNum supports both uncompressed and compressed (`zip-deflate`) `.npz` archives.

#### Reading `.npz`
```java
import java.util.Map;

Map<String, NDArray> archive = JNumIO.readNpz(Path.of("model_checkpoint.npz"));

NDArray w1 = archive.get("weights1");
NDArray b1 = archive.get("biases1");
```

#### Writing `.npz`
```java
Map<String, NDArray> data = Map.of(
    "features", JNum.ones(100, 10),
    "labels",   JNum.zeros(100, 1)
);

// Save compressed
boolean compressed = true;
JNumIO.writeNpz(Path.of("data.npz"), data, compressed);
```

---

## 3. Delimited Text (CSV / TSV)

JNum includes an ultra-fast streaming CSV/TSV parser that reads tabular text directly into off-heap `NDArray` memory buffers with zero intermediate object allocation.

### Customizing CSV Parsing (`CsvOptions`)

Use [`CsvOptions`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/CsvOptions.java) to configure delimiters, headers, missing value tokens, and row limits:

```java
import jnum.DType;
import jnum.io.CsvOptions;

CsvOptions options = CsvOptions.builder()
    .delimiter(',')               // Delimiter: ',' for CSV, '\t' for TSV, ' ' for space
    .hasHeader(true)              // Automatically skip column header row
    .commentChar('#')             // Ignore comment lines starting with '#'
    .naValue("NaN")               // Treat "NaN", "NA", or "null" as missing
    .naValue("null")
    .skipRows(2)                  // Skip first 2 metadata rows
    .maxRows(10_000)              // Read at most 10,000 data rows
    .dtype(DType.FLOAT32)         // Target data type (FLOAT32, FLOAT64, INT32)
    .build();
```

### Reading and Writing CSV Files

```java
import java.nio.file.Path;
import jnum.NDArray;
import jnum.io.JNumIO;

// Read dataset
NDArray table = JNumIO.readCsv(Path.of("train.csv"), options);
System.out.println("Dataset shape: " + table.shapeString()); // e.g. [10000, 8]

// Write dataset back to CSV
JNumIO.writeCsv(Path.of("output.csv"), table, options);
```

---

## 4. Asynchronous I/O

For high-throughput inference servers and data ingestion pipelines, all read/write operations provide non-blocking asynchronous variants that return standard `CompletableFuture`:

```java
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import jnum.NDArray;
import jnum.io.JNumIO;

CompletableFuture<NDArray> future = JNumIO.readNpyAsync(Path.of("data.npy"));

future.thenAccept(arr -> {
    System.out.println("Async loaded array sum: " + arr.sum());
}).exceptionally(ex -> {
    System.err.println("Failed to read file: " + ex.getMessage());
    return null;
});
```

---

## Supported Data Types in I/O

| JNum DType | Safetensors Format | NumPy Typecode | CSV / TSV Parsing |
|:-----------|:-------------------|:---------------|:------------------|
| `FLOAT32`  | `F32`              | `<f4`          | `float`           |
| `FLOAT64`  | `F64`              | `<f8`          | `double`          |
| `INT32`    | `I32`              | `<i4`          | `int`             |
| `INT64`    | `I64`              | `<i8`          | `long`            |
| `INT8`     | `I8`               | `|i1`          | `byte`            |
| `BOOLEAN`  | `BOOL`             | `|b1`          | `true` / `false`  |
| `BFLOAT16` | `BF16`             | `<b2`          | Promoted to `f32` |
| `FLOAT16`  | `F16`              | `<f2`          | Promoted to `f32` |
