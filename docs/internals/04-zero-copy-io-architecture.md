# Systems Internals: Zero-Copy I/O & Custom Parsers

High-throughput numerical ingestion presents two major bottlenecks in standard Java:
1. **Garbage Collection Churn from String Tokens**: Standard CSV and JSON parsers (such as Jackson or `Scanner`) allocate millions of temporary `String`, `Double`, and `Token` objects on the heap, triggering severe GC pauses.
2. **Buffer Duplication in Virtual Memory**: Reading binary files via `InputStream.read()` copies bytes from the OS page cache into an intermediate JVM heap byte array, and then copies them again into off-heap memory.

This document details how JNum eliminates these overheads through **zero-copy memory mapping**, **scoped deterministic unmapping**, and **off-heap byte-level parsers**.

---

## 1. Virtual Memory Paging & The OS Page Cache

When loading multi-gigabyte models or datasets (such as a 20 GB `.safetensors` file or a 5 GB `.npy` embeddings matrix), JNum bypasses traditional stream reads in favor of OS virtual memory paging via [`FileChannel.map(MapMode.READ_ONLY)`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/internal/npy/NpyReader.java):

```
Disk File (e.g., 20 GB weights.safetensors)
   │
   ▼ OS mmap() Syscall
Process Virtual Address Space (20 GB mapped range)
   │
   ▼ On-Demand 4 KB / 2 MB Virtual Memory Page Faults
Physical RAM (Only accessed pages are loaded into memory)
   │
   ▼ CPU SIMD Vector Registers (vmovups)
```

Advantages:
* **Zero Initial Load Time**: The `mmap` syscall completes in microseconds. The file is mapped into address space without reading the underlying bytes.
* **Kernel-Level Page Management**: The operating system's page cache automatically fetches 4 KB or 2 MB (huge) pages into RAM when an `NDArray` accesses that region, and unmaps cold pages under memory pressure.
* **Zero Double-Buffering**: Bytes flow directly from the OS page cache into CPU SIMD registers without ever being duplicated in Java heap memory.

---

## 2. Safe Deterministic Unmapping via Panama `Arena`

For 25 years, Java suffered from a notorious flaw in `java.nio.MappedByteBuffer`: **it had no public `unmap()` method**. 

To release a mapped file, applications had to rely on internal reflection hacks (`sun.misc.Cleaner`), wait for non-deterministic garbage collection, or restart the JVM. On Windows, this led to files remaining locked on disk, preventing deletion or re-writing. More dangerously, unmapping a buffer while another thread was reading it could crash the entire JVM with a native `SIGSEGV`.

JNum solves this using Panama's [`Arena`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/MmapArray.java):

```java
public final class MmapArray implements AutoCloseable {
    private final NDArray array;
    private final Arena arena;

    @Override
    public void close() {
        arena.close(); // Triggers atomic, safe OS unmap
    }
}
```

How Panama solves the unmapping problem:
1. **Deterministic Release**: Calling `arena.close()` instantly issues the native `munmap` syscall, freeing OS virtual memory and releasing Windows file locks immediately.
2. **Temporal Safety**: Every read through an `NDArray` is guarded by an internal Panama memory session check. If a thread attempts to read a mapped tensor after the arena has closed, HotSpot throws a safe `java.lang.IllegalStateException: Already closed` instead of terminating the process with a fatal segmentation fault.

---

## 3. Off-Heap Byte-Level Parsing (`FastNumberParser.java`)

When reading tabular text files (CSV/TSV), standard Java creates millions of `String` objects (e.g. `Double.parseDouble(line.split(",")[i])`), triggering catastrophic GC overhead.

JNum's [`FastNumberParser`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/internal/common/FastNumberParser.java) parses numbers directly from off-heap `MemorySegment` byte slices without allocating a single `java.lang.String`:

```
Off-Heap MemorySegment ASCII Bytes:
['-', '1', '2', '3', '.', '4', '5', '6', 'e', '+', '0', '2', ',']
  │
  ▼ FastNumberParser.parseDouble(segment, start, end)
Primitive double: -12345.6  (0 Heap Objects Allocated!)
```

### Parsing Pipeline:
1. **Direct Byte Scanning**: Reads raw ASCII bytes using `segment.get(ValueLayout.JAVA_BYTE, offset)`.
2. **Whitespace & Missing Sentinel Trimming**: Strips spaces, quotes, and detects custom NaN markers (`"NA"`, `"null"`, `"-999"`).
3. **Integer & Fraction Accumulation**: Accumulates digit characters directly into primitive `long` accumulators:
   $$\text{value} = \text{intPart} + \frac{\text{fracPart}}{10^{\text{fracDigits}}}$$
4. **Exponent Processing**: Applies powers of 10 scaling using a precomputed lookup table, avoiding slow `Math.pow()` calls.

---

## 4. Zero-Heap Safetensors JSON Parser (`SafetensorsJsonParser.java`)

To inspect and load `.safetensors` model weights without external dependencies like Jackson or Gson, JNum includes a specialized off-heap JSON scanner ([`SafetensorsJsonParser.java`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/internal/safetensors/SafetensorsJsonParser.java)).

Because the Safetensors JSON header is guaranteed to describe only tensor metadata (data types, shapes, and byte offsets), the parser:
* Scans string boundaries by indexing matching quotes `"` and braces `{}`.
* Decodes byte offsets (`data_offsets: [start, end]`) directly into primitive `long` fields.
* Allocates only the resulting [`SafetensorsTensorDesc`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/io/internal/safetensors/SafetensorsTensorDesc.java) records, consuming less than 1 MB of heap memory even when inspecting a 70-billion-parameter LLM checkpoint containing thousands of weight tensors.
