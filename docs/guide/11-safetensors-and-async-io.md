# Chapter 11: Safetensors & Asynchronous I/O

For years, deep learning checkpoints were distributed as Python `pickle` files (`.pt` or `.bin`). Pickles present a severe security vulnerability: deserializing an untrusted checkpoint can execute arbitrary Python shell code on the host machine. Furthermore, loading pickles in Java requires heavy external dependencies or Python interop bridges.

To solve this, Hugging Face introduced **`safetensors`**—a simple, secure, header-indexed binary format designed for instant zero-copy memory mapping. JNum provides native, pure-Java support for reading, writing, inspecting, and memory-mapping `.safetensors` files.

---

## 1. The Architecture of a `.safetensors` File

A `.safetensors` file has an elegant binary structure:

```
┌────────────────────────┬───────────────────────────────────┬────────────────────────────────┐
│  Header Length (8 B)   │        JSON Metadata Header       │       Raw Tensor Buffers       │
│  64-bit Little-Endian  │   UTF-8 JSON string describing    │   Contiguous native bytes      │
│     Unsigned Int       │    offsets, shapes, and types     │    of all model parameters     │
└────────────────────────┴───────────────────────────────────┴────────────────────────────────┘
 0                      8                                   N                                EOF
```

Because the JSON header specifies the exact byte start and end offset of every tensor, JNum can map a 100 GB LLM model into address space in milliseconds, without scanning or parsing the tensor data.

---

## 2. Inspecting Checkpoints Without Loading Data

Before allocating memory, you can inspect the metadata, tensor names, shapes, and data types of a checkpoint using [`JNumIO.inspectSafetensors()`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/JNumIO.java#L224):

```java
import java.nio.file.Path;
import jnum.io.JNumIO;
import jnum.io.SafetensorsInfo;

void main() throws Exception {
    Path modelPath = Path.of("models/llama-3.safetensors");

    // Parses only the JSON header - zero tensor memory allocated
    SafetensorsInfo info = JNumIO.inspectSafetensors(modelPath);

    IO.println("Total Tensors: " + info.tensorCount());
    IO.println("User Metadata: " + info.userMetadata());

    // Inspect individual tensor descriptors
    for (String tensorName : info.tensorNames()) {
        long[] shape = info.getShape(tensorName);
        String dtype = info.getDType(tensorName);
        IO.println(tensorName + " -> " + java.util.Arrays.toString(shape) + " [" + dtype + "]");
    }
}
```

---

## 3. Loading Checkpoints: Bulk vs. Selective

### Bulk Loading (`readSafetensors`)

To load all weights into native memory:

```java
import java.nio.file.Path;
import java.util.Map;
import jnum.NDArray;
import jnum.io.JNumIO;

void main() throws Exception {
    Map<String, NDArray> weights = JNumIO.readSafetensors(Path.of("model.safetensors"));

    NDArray embed = weights.get("model.embed_tokens.weight");
    IO.println("Embedding weights: " + embed.shapeString());
}
```

### Selective Single-Tensor Loading (`readSafetensor`)

If you only need a single layer or specific projection matrix from a massive checkpoint, loading the entire file wastes RAM. JNum allows you to extract a single named tensor:

```java
// Loads ONLY the specified tensor from disk; skips all other layers
NDArray gateProj = JNumIO.readSafetensor(
    Path.of("model.safetensors"),
    "model.layers.0.mlp.gate_proj.weight"
);

IO.println("Loaded gate projection: " + gateProj.shapeString());
```

---

## 4. Zero-Copy Deterministic Memory Mapping (`MmapTensors`)

For multi-gigabyte models, copying weights from disk to RAM is inefficient. JNum allows memory-mapping an entire `.safetensors` archive using [`readSafetensorsMmapScoped`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/JNumIO.java#L278). 

Tensors are returned inside an `AutoCloseable` container ([`MmapTensors`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/MmapTensors.java)), guaranteeing safe and deterministic unmapping when execution completes:

```java
import java.nio.file.Path;
import jnum.NDArray;
import jnum.io.JNumIO;
import jnum.io.MmapTensors;

void main() throws Exception {
    Path weightsPath = Path.of("large_transformer.safetensors");

    // Map entire archive into virtual address space
    try (MmapTensors model = JNumIO.readSafetensorsMmapScoped(weightsPath)) {
        NDArray layer1 = model.get("transformer.h.0.attn.c_attn.weight");
        
        IO.println("Memory-mapped tensor: " + layer1.shapeString());
        IO.println("First weight element: " + layer1.getFloat(0, 0));
        
        // Execute inference...
    } // <-- Unmapped from virtual memory immediately on close. Zero leaks!
}
```

---

## 5. Writing Safetensors Checkpoints

You can serialize models trained or processed in JNum back to `.safetensors`, including arbitrary string metadata (such as model architecture, tokenizer vocab size, or training hyperparameters):

```java
import java.nio.file.Path;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.JNumIO;

void main() throws Exception {
    // 1. Prepare weight dictionary
    Map<String, NDArray> tensors = Map.of(
        "encoder.weight", JNum.rand(256, 512),
        "encoder.bias",   JNum.zeros(256)
    );

    // 2. Prepare user metadata dictionary
    Map<String, String> userMetadata = Map.of(
        "format", "jnum-v0.2.0",
        "author", "DeepLearningTeam",
        "precision", "float32"
    );

    // 3. Write safetensors checkpoint
    Path exportPath = Path.of("custom_model.safetensors");
    JNumIO.writeSafetensors(tensors, userMetadata, exportPath);
    IO.println("Successfully exported checkpoint to " + exportPath);
}
```

---

## 6. Practical Recipe: Checkpoint Validation Pipeline

The following recipe verifies that a checkpoint contains the expected layers and ensures no weight matrices contain NaN values:

```java
import java.nio.file.Path;
import jnum.NDArray;
import jnum.io.JNumIO;
import jnum.io.MmapTensors;

void main() throws Exception {
    Path checkpoint = Path.of("model.safetensors");

    try (MmapTensors weights = JNumIO.readSafetensorsMmapScoped(checkpoint)) {
        IO.println("Validating checkpoint layers...");

        for (var entry : weights.tensors().entrySet()) {
            String name = entry.getKey();
            NDArray tensor = entry.getValue();

            // Compute summary statistics directly over memory-mapped pages
            double norm = tensor.norm();
            
            if (Double.isNaN(norm)) {
                IO.println("CRITICAL: " + name + " contains NaN values!");
            } else {
                IO.println("OK: " + name + " (Shape: " + tensor.shapeString() + ", L2-Norm: " + norm + ")");
            }
        }
    }
}
```

Now proceed to **[Chapter 12: Neural Network Inference](12-neural-network-inference.md)** to compose modules, layers, and forward passes with `jnum.nn`.
