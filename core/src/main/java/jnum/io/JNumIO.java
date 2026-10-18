/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.io;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import jnum.DType;
import jnum.NDArray;
import jnum.io.internal.async.AsyncIO;
import jnum.io.internal.csv.CsvReader;
import jnum.io.internal.csv.CsvWriter;
import jnum.io.internal.npy.NpyReader;
import jnum.io.internal.npy.NpyWriter;
import jnum.io.internal.npy.NpzArchive;
import jnum.io.internal.safetensors.SafetensorsReader;
import jnum.io.internal.safetensors.SafetensorsWriter;

/**
 * High-performance I/O operations for JNum.
 *
 * <p>Provides 100% native FFM binary compatibility with:
 * <ul>
 *   <li><b>NumPy {@code .npy} &amp; {@code .npz}:</b> Read and write single or multi-array archives with full
 *       header parsing and zero-copy memory mapping.</li>
 *   <li><b>Hugging Face {@code .safetensors}:</b> Read and write modern neural network checkpoints, supporting
 *       selective tensor loading, header metadata inspection, and instant memory-mapped access.</li>
 *   <li><b>Delimited Text (CSV, TSV):</b> High-throughput SIMD-accelerated tabular text parsing and serialization.</li>
 *   <li><b>Asynchronous I/O:</b> Non-blocking async execution returning {@link CompletableFuture}.</li>
 * </ul>
 * </p>
 */
public final class JNumIO {

    private JNumIO() {
        throw new AssertionError("JNumIO cannot be instantiated.");
    }

    // =========================================================================
    // NumPy .npy Operations
    // =========================================================================

    /**
     * Reads a NumPy {@code .npy} binary file into a newly allocated {@link NDArray} managed by automatic GC arena.
     *
     * @param path filesystem path to the {@code .npy} file
     * @return deserialized {@link NDArray}
     * @throws IOException if an I/O error occurs or the file contains an invalid NumPy magic header
     */
    public static NDArray readNpy(Path path) throws IOException {
        return NpyReader.read(path, Arena.ofAuto());
    }

    /**
     * Reads a NumPy {@code .npy} binary file into a newly allocated {@link NDArray} using the specified {@link Arena}.
     *
     * @param path filesystem path to the {@code .npy} file
     * @param arena memory arena governing lifecycle of the native tensor buffer
     * @return deserialized {@link NDArray}
     * @throws IOException if an I/O error occurs or the file contains an invalid NumPy magic header
     */
    public static NDArray readNpy(Path path, Arena arena) throws IOException {
        return NpyReader.read(path, arena);
    }

    /**
     * Loads a NumPy {@code .npy} file using zero-copy memory mapping governed by an automatic GC arena.
     *
     * <p>Pages are brought into memory on-demand by the OS kernel without loading entire multi-gigabyte files into RAM.</p>
     *
     * @param path filesystem path to the {@code .npy} file
     * @return memory-mapped {@link NDArray}
     * @throws IOException if an I/O or memory mapping error occurs
     */
    public static NDArray readNpyMmap(Path path) throws IOException {
        return NpyReader.readMmap(path, Arena.ofAuto());
    }

    /**
     * Loads a NumPy {@code .npy} file using zero-copy memory mapping with deterministic lifecycle control.
     *
     * <p>Closing the returned {@link MmapArray} unmaps memory immediately from the operating system.</p>
     *
     * @param path filesystem path to the {@code .npy} file
     * @return an {@link AutoCloseable} {@link MmapArray} container
     * @throws IOException if an I/O or memory mapping error occurs
     */
    public static MmapArray readNpyMmapScoped(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        NDArray arr = NpyReader.readMmap(path, arena);
        return new MmapArray(arr, arena);
    }

    /**
     * Serializes an {@link NDArray} into standard NumPy {@code .npy} binary format.
     *
     * @param array tensor to write
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs while writing
     */
    public static void writeNpy(NDArray array, Path path) throws IOException {
        NpyWriter.write(array, path);
    }

    /**
     * Asynchronously reads a NumPy {@code .npy} file on a background worker thread.
     *
     * @param path filesystem path to the {@code .npy} file
     * @return a {@link CompletableFuture} completing with the deserialized {@link NDArray}
     */
    public static CompletableFuture<NDArray> readNpyAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readNpy(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Asynchronously serializes an {@link NDArray} to a {@code .npy} file on a background worker thread.
     *
     * @param array tensor to write
     * @param path destination filesystem path
     * @return a {@link CompletableFuture} completing when writing finishes
     */
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

    /**
     * Reads all arrays from a compressed or uncompressed NumPy {@code .npz} ZIP archive.
     *
     * @param path filesystem path to the {@code .npz} file
     * @return map of array names to deserialized {@link NDArray} instances
     * @throws IOException if an I/O error occurs reading the ZIP archive
     */
    public static Map<String, NDArray> readNpz(Path path) throws IOException {
        return NpzArchive.read(path, Arena.ofAuto());
    }

    /**
     * Reads all arrays from a NumPy {@code .npz} ZIP archive using the specified {@link Arena}.
     *
     * @param path filesystem path to the {@code .npz} file
     * @param arena memory arena governing lifecycle of all loaded tensor buffers
     * @return map of array names to deserialized {@link NDArray} instances
     * @throws IOException if an I/O error occurs reading the ZIP archive
     */
    public static Map<String, NDArray> readNpz(Path path, Arena arena) throws IOException {
        return NpzArchive.read(path, arena);
    }

    /**
     * Serializes a dictionary of named tensors into a NumPy {@code .npz} archive.
     *
     * @param arrays map of tensor names to {@link NDArray} instances
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs writing the ZIP archive
     */
    public static void writeNpz(Map<String, NDArray> arrays, Path path) throws IOException {
        NpzArchive.write(arrays, path);
    }

    /**
     * Asynchronously reads a NumPy {@code .npz} archive on a background worker thread.
     *
     * @param path filesystem path to the {@code .npz} file
     * @return a {@link CompletableFuture} completing with the map of named tensors
     */
    public static CompletableFuture<Map<String, NDArray>> readNpzAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readNpz(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Asynchronously writes a map of tensors to a NumPy {@code .npz} archive on a background worker thread.
     *
     * @param arrays map of tensor names to {@link NDArray} instances
     * @param path destination filesystem path
     * @return a {@link CompletableFuture} completing when writing finishes
     */
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
     * Inspects a {@code .safetensors} file's tensor definitions and metadata without reading weight payloads.
     *
     * <p>Executes in microseconds by reading only the leading JSON header.</p>
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @return {@link SafetensorsInfo} describing all tensors and user metadata in the file
     * @throws IOException if an I/O error occurs or the file is not a valid safetensors archive
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
     * Reads all tensors from a {@code .safetensors} file into native memory managed by GC.
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @return map of tensor names to {@link NDArray} instances
     * @throws IOException if an I/O error occurs reading the file
     */
    public static Map<String, NDArray> readSafetensors(Path path) throws IOException {
        return SafetensorsReader.read(path, Arena.ofAuto());
    }

    /**
     * Reads all tensors from a {@code .safetensors} file into native memory managed by the specified {@link Arena}.
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @param arena memory arena governing lifecycle of all loaded tensor buffers
     * @return map of tensor names to {@link NDArray} instances
     * @throws IOException if an I/O error occurs reading the file
     */
    public static Map<String, NDArray> readSafetensors(Path path, Arena arena) throws IOException {
        return SafetensorsReader.read(path, arena);
    }

    /**
     * Instant zero-copy memory-mapped load of all tensors from a {@code .safetensors} file (managed by GC).
     *
     * <p>Enables instantaneous access to 100GB+ LLM weights with zero initial load latency.</p>
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @return map of tensor names to memory-mapped {@link NDArray} instances
     * @throws IOException if an I/O or memory mapping error occurs
     */
    public static Map<String, NDArray> readSafetensorsMmap(Path path) throws IOException {
        return SafetensorsReader.readMmap(path, Arena.ofAuto());
    }

    /**
     * Instant zero-copy memory-mapped load of all tensors with deterministic lifecycle control.
     *
     * <p>Closing the returned {@link MmapTensors} unmaps the entire file from the OS immediately.</p>
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @return an {@link AutoCloseable} {@link MmapTensors} container
     * @throws IOException if an I/O or memory mapping error occurs
     */
    public static MmapTensors readSafetensorsMmapScoped(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        Map<String, NDArray> tensors = SafetensorsReader.readMmap(path, arena);
        return new MmapTensors(tensors, arena);
    }

    /**
     * Reads a single tensor by name from a {@code .safetensors} file without parsing unneeded tensors.
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @param tensorName name of the tensor weight to load
     * @return deserialized {@link NDArray}
     * @throws IOException if an I/O error occurs or the tensor is not found
     */
    public static NDArray readSafetensor(Path path, String tensorName) throws IOException {
        return SafetensorsReader.readSingle(path, tensorName, Arena.ofAuto());
    }

    /**
     * Reads a single tensor by name from a {@code .safetensors} file using the specified {@link Arena}.
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @param tensorName name of the tensor weight to load
     * @param arena memory arena governing lifecycle of the loaded tensor buffer
     * @return deserialized {@link NDArray}
     * @throws IOException if an I/O error occurs or the tensor is not found
     */
    public static NDArray readSafetensor(Path path, String tensorName, Arena arena) throws IOException {
        return SafetensorsReader.readSingle(path, tensorName, arena);
    }

    /**
     * Writes a collection of tensors to a {@code .safetensors} file.
     *
     * @param tensors map of tensor names to {@link NDArray} instances
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs during writing
     */
    public static void writeSafetensors(Map<String, NDArray> tensors, Path path) throws IOException {
        SafetensorsWriter.write(tensors, null, path);
    }

    /**
     * Writes a collection of tensors and custom user metadata strings to a {@code .safetensors} file.
     *
     * @param tensors map of tensor names to {@link NDArray} instances
     * @param userMetadata key-value string metadata (e.g. format version, model hyperparameters)
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs during writing
     */
    public static void writeSafetensors(Map<String, NDArray> tensors, Map<String, String> userMetadata, Path path) throws IOException {
        SafetensorsWriter.write(tensors, userMetadata, path);
    }

    /**
     * Asynchronously reads all tensors from a {@code .safetensors} file on a background worker thread.
     *
     * @param path filesystem path to the {@code .safetensors} file
     * @return a {@link CompletableFuture} completing with the map of loaded tensors
     */
    public static CompletableFuture<Map<String, NDArray>> readSafetensorsAsync(Path path) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readSafetensors(path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Asynchronously serializes a map of tensors to a {@code .safetensors} file on a background worker thread.
     *
     * @param tensors map of tensor names to {@link NDArray} instances
     * @param path destination filesystem path
     * @return a {@link CompletableFuture} completing when writing finishes
     */
    public static CompletableFuture<Void> writeSafetensorsAsync(Map<String, NDArray> tensors, Path path) {
        return AsyncIO.runAsync(() -> {
            try {
                writeSafetensors(tensors, path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    // =========================================================================
    // Delimited Text (.csv, .tsv) Operations
    // =========================================================================

    /**
     * Reads a comma-delimited CSV file using default configuration ({@link CsvOptions#DEFAULT}).
     *
     * @param path filesystem path to the CSV file
     * @return deserialized 2D {@link NDArray}
     * @throws IOException if an I/O error occurs reading the text file
     */
    public static NDArray readCsv(Path path) throws IOException {
        return readCsv(path, CsvOptions.DEFAULT, Arena.ofAuto());
    }

    /**
     * Reads a comma-delimited CSV file parsed into the specified {@link DType}.
     *
     * @param path filesystem path to the CSV file
     * @param dtype target data type (e.g. {@link DType#f32})
     * @return deserialized 2D {@link NDArray}
     * @throws IOException if an I/O error occurs reading the text file
     */
    public static NDArray readCsv(Path path, DType dtype) throws IOException {
        return readCsv(path, CsvOptions.builder().dtype(dtype).build(), Arena.ofAuto());
    }

    /**
     * Reads a delimited text file using the provided {@link CsvOptions}.
     *
     * @param path filesystem path to the text file
     * @param options configuration options governing delimiters, headers, and parsing
     * @return deserialized 2D {@link NDArray}
     * @throws IOException if an I/O error occurs reading the text file
     */
    public static NDArray readCsv(Path path, CsvOptions options) throws IOException {
        return readCsv(path, options, Arena.ofAuto());
    }

    /**
     * Reads a delimited text file using the provided {@link CsvOptions} and {@link Arena}.
     *
     * @param path filesystem path to the text file
     * @param options configuration options governing delimiters, headers, and parsing
     * @param arena memory arena governing lifecycle of the parsed tensor buffer
     * @return deserialized 2D {@link NDArray}
     * @throws IOException if an I/O error occurs reading the text file
     */
    public static NDArray readCsv(Path path, CsvOptions options, Arena arena) throws IOException {
        return CsvReader.read(path, options, arena);
    }

    /**
     * Writes an {@link NDArray} to a text file in comma-delimited CSV format using default options.
     *
     * @param array 2D tensor to serialize
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs during writing
     */
    public static void writeCsv(NDArray array, Path path) throws IOException {
        writeCsv(array, CsvOptions.DEFAULT, path);
    }

    /**
     * Writes an {@link NDArray} to a text file using custom {@link CsvOptions}.
     *
     * @param array 2D tensor to serialize
     * @param options formatting configuration options
     * @param path destination filesystem path
     * @throws IOException if an I/O error occurs during writing
     */
    public static void writeCsv(NDArray array, CsvOptions options, Path path) throws IOException {
        CsvWriter.write(array, options, path);
    }

    /**
     * Asynchronously reads a CSV file using default options on a background worker thread.
     *
     * @param path filesystem path to the CSV file
     * @return a {@link CompletableFuture} completing with the deserialized 2D {@link NDArray}
     */
    public static CompletableFuture<NDArray> readCsvAsync(Path path) {
        return readCsvAsync(path, CsvOptions.DEFAULT);
    }

    /**
     * Asynchronously reads a delimited text file using custom options on a background worker thread.
     *
     * @param path filesystem path to the text file
     * @param options parsing configuration options
     * @return a {@link CompletableFuture} completing with the deserialized 2D {@link NDArray}
     */
    public static CompletableFuture<NDArray> readCsvAsync(Path path, CsvOptions options) {
        return AsyncIO.supplyAsync(() -> {
            try {
                return readCsv(path, options);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Asynchronously writes an {@link NDArray} to a CSV file using default options on a background worker thread.
     *
     * @param array 2D tensor to serialize
     * @param path destination filesystem path
     * @return a {@link CompletableFuture} completing when writing finishes
     */
    public static CompletableFuture<Void> writeCsvAsync(NDArray array, Path path) {
        return writeCsvAsync(array, CsvOptions.DEFAULT, path);
    }

    /**
     * Asynchronously writes an {@link NDArray} to a delimited text file using custom options on a background worker thread.
     *
     * @param array 2D tensor to serialize
     * @param options formatting configuration options
     * @param path destination filesystem path
     * @return a {@link CompletableFuture} completing when writing finishes
     */
    public static CompletableFuture<Void> writeCsvAsync(NDArray array, CsvOptions options, Path path) {
        return AsyncIO.runAsync(() -> {
            try {
                writeCsv(array, options, path);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }
}