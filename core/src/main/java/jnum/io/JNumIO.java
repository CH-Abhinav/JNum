package jnum.io;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import jnum.NDArray;
import jnum.io.internal.async.AsyncIO;
import jnum.io.internal.npy.NpyReader;
import jnum.io.internal.npy.NpyWriter;
import jnum.io.internal.npy.NpzArchive;

/**
 * High-performance I/O operations for JNum.
 * Provides 100% native FFM binary compatibility with NumPy (.npy, .npz) and zero-copy memory mapping.
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
}