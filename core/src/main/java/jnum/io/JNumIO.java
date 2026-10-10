package jnum.io;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import jnum.NDArray;
import jnum.io.internal.async.AsyncIO;
import jnum.io.internal.npy.NpyReader;
import jnum.io.internal.npy.NpyWriter;
import jnum.io.internal.npy.NpzArchive;
import jnum.io.internal.safetensors.SafetensorsReader;
import jnum.io.internal.safetensors.SafetensorsWriter;

/**
 * High-performance I/O operations for JNum.
 * Provides 100% native FFM binary compatibility with NumPy (.npy, .npz),
 * Hugging Face (.safetensors), and zero-copy memory mapping.
 */
public final class JNumIO {

    private JNumIO() {
        throw new AssertionError("JNumIO cannot be instantiated.");
    }

    // =========================================================================
    // NumPy .npy Operations
    // =========================================================================

    public static NDArray readNpy(Path path) throws IOException {
        return NpyReader.read(path, Arena.ofAuto());
    }

    public static NDArray readNpy(Path path, Arena arena) throws IOException {
        return NpyReader.read(path, arena);
    }

    /**
     * Zero-copy memory-mapped load (lifetime managed by GC).
     */
    public static NDArray readNpyMmap(Path path) throws IOException {
        return NpyReader.readMmap(path, Arena.ofAuto());
    }

    /**
     * Zero-copy memory-mapped load with deterministic lifecycle control.
     */
    public static MmapArray readNpyMmapScoped(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        NDArray arr = NpyReader.readMmap(path, arena);
        return new MmapArray(arr, arena);
    }

    public static void writeNpy(NDArray array, Path path) throws IOException {
        NpyWriter.write(array, path);
    }

    public static CompletableFuture<NDArray> readNpyAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readNpy(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    public static CompletableFuture<Void> writeNpyAsync(NDArray array, Path path) {
        return AsyncIO.runAsync(() -> {
            try {
                writeNpy(array, path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    // =========================================================================
    // NumPy .npz Operations
    // =========================================================================

    public static Map<String, NDArray> readNpz(Path path) throws IOException {
        return NpzArchive.read(path, Arena.ofAuto());
    }

    public static Map<String, NDArray> readNpz(Path path, Arena arena) throws IOException {
        return NpzArchive.read(path, arena);
    }

    public static void writeNpz(Map<String, NDArray> arrays, Path path) throws IOException {
        NpzArchive.write(arrays, path);
    }

    public static CompletableFuture<Map<String, NDArray>> readNpzAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readNpz(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    public static CompletableFuture<Void> writeNpzAsync(Map<String, NDArray> arrays, Path path) {
        return AsyncIO.runAsync(() -> {
            try {
                writeNpz(arrays, path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    // =========================================================================
    // Hugging Face .safetensors Operations
    // =========================================================================

    /**
     * Inspects a .safetensors file's tensor definitions and metadata without loading tensor weights.
     */
    public static SafetensorsInfo inspectSafetensors(Path path) throws IOException {
        var meta = SafetensorsReader.readMetadata(path);
        Map<String, SafetensorsInfo.TensorInfo> map = new HashMap<>();
        for (var desc : meta.tensors().values()) {
            map.put(desc.name(), new SafetensorsInfo.TensorInfo(desc.name(), desc.dtype(), desc.shape(), desc.byteSize()));
        }
        return new SafetensorsInfo(map, meta.userMetadata(), meta.headerLength());
    }

    /**
     * Reads all tensors from a .safetensors file into off-heap memory managed by GC.
     */
    public static Map<String, NDArray> readSafetensors(Path path) throws IOException {
        return SafetensorsReader.read(path, Arena.ofAuto());
    }

    public static Map<String, NDArray> readSafetensors(Path path, Arena arena) throws IOException {
        return SafetensorsReader.read(path, arena);
    }

    /**
     * Instant zero-copy memory-mapped load of all tensors (managed by GC).
     */
    public static Map<String, NDArray> readSafetensorsMmap(Path path) throws IOException {
        return SafetensorsReader.readMmap(path, Arena.ofAuto());
    }

    /**
     * Instant zero-copy memory-mapped load of all tensors with deterministic lifecycle control.
     */
    public static MmapTensors readSafetensorsMmapScoped(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        Map<String, NDArray> tensors = SafetensorsReader.readMmap(path, arena);
        return new MmapTensors(tensors, arena);
    }

    /**
     * Reads a single tensor by name from a .safetensors file.
     */
    public static NDArray readSafetensor(Path path, String tensorName) throws IOException {
        return SafetensorsReader.readSingle(path, tensorName, Arena.ofAuto());
    }

    public static NDArray readSafetensor(Path path, String tensorName, Arena arena) throws IOException {
        return SafetensorsReader.readSingle(path, tensorName, arena);
    }

    /**
     * Writes tensors to a .safetensors file.
     */
    public static void writeSafetensors(Map<String, NDArray> tensors, Path path) throws IOException {
        SafetensorsWriter.write(tensors, null, path);
    }

    public static void writeSafetensors(Map<String, NDArray> tensors, Map<String, String> userMetadata, Path path) throws IOException {
        SafetensorsWriter.write(tensors, userMetadata, path);
    }

    public static CompletableFuture<Map<String, NDArray>> readSafetensorsAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readSafetensors(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    public static CompletableFuture<Void> writeSafetensorsAsync(Map<String, NDArray> tensors, Path path) {
        return AsyncIO.runAsync(() -> {
            try {
                writeSafetensors(tensors, path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }
}