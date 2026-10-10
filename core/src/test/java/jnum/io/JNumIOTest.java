package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class JNumIOTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<JNumIO> constructor = JNumIO.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    // =========================================================================
    // NPY Tests
    // =========================================================================

    @Test
    void testNpyRoundTripFloat32() throws IOException {
        Path file = tempDir.resolve("arr_f32.npy");
        NDArray original = JNum.from(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);

        JNumIO.writeNpy(original, file);
        assertTrue(Files.exists(file));

        NDArray loaded = JNumIO.readNpy(file);
        assertEquals(DType.f32, loaded.getDType());
        assertArrayEquals(new long[]{2, 3}, loaded.getShape());
        assertEquals(1.0f, loaded.getFloat(0, 0), 1e-6f);
        assertEquals(6.0f, loaded.getFloat(1, 2), 1e-6f);
    }

    @Test
    void testNpyRoundTripWithCustomArena() throws IOException {
        Path file = tempDir.resolve("arr_arena.npy");
        NDArray original = JNum.from(new double[]{10.5, 20.5}, 2);

        JNumIO.writeNpy(original, file);
        try (Arena arena = Arena.ofConfined()) {
            NDArray loaded = JNumIO.readNpy(file, arena);
            assertEquals(10.5, loaded.getDouble(0), 1e-9);
            assertEquals(20.5, loaded.getDouble(1), 1e-9);
        }
    }

    @Test
    void testNpyMmapRead() throws IOException {
        Path file = tempDir.resolve("arr_mmap.npy");
        NDArray original = JNum.from(new int[]{7, 8, 9}, 3);

        JNumIO.writeNpy(original, file);
        NDArray loaded = JNumIO.readNpyMmap(file);
        assertEquals(DType.i32, loaded.getDType());
        assertEquals(7, loaded.getInt(0));
        assertEquals(9, loaded.getInt(2));
    }

    @Test
    void testNpyMmapScopedRead() throws IOException {
        Path file = tempDir.resolve("arr_mmap_scoped.npy");
        NDArray original = JNum.from(new float[]{4.0f, 5.0f}, 2);

        JNumIO.writeNpy(original, file);
        try (MmapArray mmap = JNumIO.readNpyMmapScoped(file)) {
            NDArray loaded = mmap.array();
            assertEquals(4.0f, loaded.getFloat(0), 1e-6f);
            assertEquals(5.0f, loaded.getFloat(1), 1e-6f);
        }
    }

    @Test
    void testNpyAsyncReadWrite() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("arr_async.npy");
        NDArray original = JNum.from(new float[]{11.0f, 12.0f}, 2);

        JNumIO.writeNpyAsync(original, file).get();
        assertTrue(Files.exists(file));

        NDArray loaded = JNumIO.readNpyAsync(file).get();
        assertEquals(11.0f, loaded.getFloat(0), 1e-6f);
        assertEquals(12.0f, loaded.getFloat(1), 1e-6f);
    }

    // =========================================================================
    // NPZ Tests
    // =========================================================================

    @Test
    void testNpzRoundTrip() throws IOException {
        Path file = tempDir.resolve("archive.npz");
        NDArray a = JNum.from(new float[]{1f, 2f}, 2);
        NDArray b = JNum.from(new int[]{10, 20, 30}, 3);
        Map<String, NDArray> map = Map.of("arr_a", a, "arr_b", b);

        JNumIO.writeNpz(map, file);
        assertTrue(Files.exists(file));

        Map<String, NDArray> loaded = JNumIO.readNpz(file);
        assertEquals(2, loaded.size());
        assertEquals(1f, loaded.get("arr_a").getFloat(0), 1e-6f);
        assertEquals(30, loaded.get("arr_b").getInt(2));
    }

    @Test
    void testNpzWithCustomArena() throws IOException {
        Path file = tempDir.resolve("archive_arena.npz");
        NDArray a = JNum.from(new double[]{3.14}, 1);
        JNumIO.writeNpz(Map.of("pi", a), file);

        try (Arena arena = Arena.ofConfined()) {
            Map<String, NDArray> loaded = JNumIO.readNpz(file, arena);
            assertEquals(3.14, loaded.get("pi").getDouble(0), 1e-6);
        }
    }

    @Test
    void testNpzAsyncReadWrite() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("archive_async.npz");
        NDArray a = JNum.from(new float[]{99f}, 1);
        Map<String, NDArray> map = Map.of("x", a);

        JNumIO.writeNpzAsync(map, file).get();
        Map<String, NDArray> loaded = JNumIO.readNpzAsync(file).get();
        assertEquals(99f, loaded.get("x").getFloat(0), 1e-6f);
    }

    // =========================================================================
    // Safetensors Tests
    // =========================================================================

    @Test
    void testSafetensorsRoundTripAndInspect() throws IOException {
        Path file = tempDir.resolve("model.safetensors");
        NDArray w = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray b = JNum.from(new float[]{0.5f, 0.5f}, 2);
        Map<String, NDArray> tensors = Map.of("weight", w, "bias", b);
        Map<String, String> meta = Map.of("arch", "dense");

        JNumIO.writeSafetensors(tensors, meta, file);

        SafetensorsInfo info = JNumIO.inspectSafetensors(file);
        assertEquals(2, info.tensors().size());
        assertEquals("dense", info.userMetadata().get("arch"));
        assertEquals(DType.f32, info.tensors().get("weight").dtype());

        Map<String, NDArray> loaded = JNumIO.readSafetensors(file);
        assertEquals(2, loaded.size());
        assertEquals(4f, loaded.get("weight").getFloat(1, 1), 1e-6f);
    }

    @Test
    void testSafetensorsWithoutUserMetadata() throws IOException {
        Path file = tempDir.resolve("model_nometa.safetensors");
        NDArray w = JNum.from(new double[]{1.1, 2.2}, 2);
        JNumIO.writeSafetensors(Map.of("w", w), file);

        Map<String, NDArray> loaded = JNumIO.readSafetensors(file);
        assertEquals(1.1, loaded.get("w").getDouble(0), 1e-6);
    }

    @Test
    void testSafetensorsMmapAndScoped() throws IOException {
        Path file = tempDir.resolve("model_mmap.safetensors");
        NDArray w = JNum.from(new float[]{10f, 20f}, 2);
        JNumIO.writeSafetensors(Map.of("w", w), file);

        Map<String, NDArray> loaded = JNumIO.readSafetensorsMmap(file);
        assertEquals(10f, loaded.get("w").getFloat(0), 1e-6f);

        try (MmapTensors scoped = JNumIO.readSafetensorsMmapScoped(file)) {
            assertEquals(20f, scoped.get("w").getFloat(1), 1e-6f);
        }
    }

    @Test
    void testSafetensorsReadSingle() throws IOException {
        Path file = tempDir.resolve("model_single.safetensors");
        NDArray w1 = JNum.from(new float[]{1f, 2f}, 2);
        NDArray w2 = JNum.from(new float[]{3f, 4f}, 2);
        JNumIO.writeSafetensors(Map.of("w1", w1, "w2", w2), file);

        NDArray loadedW2 = JNumIO.readSafetensor(file, "w2");
        assertEquals(3f, loadedW2.getFloat(0), 1e-6f);

        try (Arena arena = Arena.ofConfined()) {
            NDArray loadedW1 = JNumIO.readSafetensor(file, "w1", arena);
            assertEquals(1f, loadedW1.getFloat(0), 1e-6f);
        }

        assertThrows(IllegalArgumentException.class, () -> JNumIO.readSafetensor(file, "nonexistent"));
    }

    @Test
    void testSafetensorsAsync() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("model_async.safetensors");
        NDArray w = JNum.from(new float[]{5f, 6f}, 2);
        Map<String, NDArray> map = Map.of("w", w);

        JNumIO.writeSafetensorsAsync(map, file).get();
        Map<String, NDArray> loaded = JNumIO.readSafetensorsAsync(file).get();
        assertEquals(5f, loaded.get("w").getFloat(0), 1e-6f);
    }

    // =========================================================================
    // CSV Tests
    // =========================================================================

    @Test
    void testCsvRoundTripDefault() throws IOException {
        Path file = tempDir.resolve("data.csv");
        NDArray orig = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2);

        JNumIO.writeCsv(orig, file);
        NDArray loaded = JNumIO.readCsv(file);

        assertEquals(DType.f64, loaded.getDType());
        assertArrayEquals(new long[]{2, 2}, loaded.getShape());
        assertEquals(1.0, loaded.getDouble(0, 0), 1e-6);
        assertEquals(4.0, loaded.getDouble(1, 1), 1e-6);
    }

    @Test
    void testCsvRoundTripWithDType() throws IOException {
        Path file = tempDir.resolve("data_f32.csv");
        NDArray orig = JNum.from(new float[]{1.5f, 2.5f}, 1, 2);

        JNumIO.writeCsv(orig, file);
        NDArray loaded = JNumIO.readCsv(file, DType.f32);

        assertEquals(DType.f32, loaded.getDType());
        assertEquals(1.5f, loaded.getFloat(0, 0), 1e-6f);
        assertEquals(2.5f, loaded.getFloat(0, 1), 1e-6f);
    }

    @Test
    void testCsvWithOptionsAndCustomArena() throws IOException {
        Path file = tempDir.resolve("data_custom.tsv");
        NDArray orig = JNum.from(new float[]{10f, 20f, 30f, 40f}, 2, 2);

        CsvOptions options = CsvOptions.builder().delimiter('\t').dtype(DType.f32).build();
        JNumIO.writeCsv(orig, options, file);

        try (Arena arena = Arena.ofConfined()) {
            NDArray loaded = JNumIO.readCsv(file, options, arena);
            assertEquals(10f, loaded.getFloat(0, 0), 1e-6f);
            assertEquals(40f, loaded.getFloat(1, 1), 1e-6f);
        }
    }

    @Test
    void testCsvAsyncReadWrite() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("data_async.csv");
        NDArray orig = JNum.from(new double[]{7.0, 8.0}, 1, 2);

        JNumIO.writeCsvAsync(orig, file).get();
        NDArray loaded = JNumIO.readCsvAsync(file).get();
        assertEquals(7.0, loaded.getDouble(0, 0), 1e-6);

        Path file2 = tempDir.resolve("data_async2.csv");
        CsvOptions options = CsvOptions.DEFAULT;
        JNumIO.writeCsvAsync(orig, options, file2).get();
        NDArray loaded2 = JNumIO.readCsvAsync(file2, options).get();
        assertEquals(8.0, loaded2.getDouble(0, 1), 1e-6);
    }
}
