package jnum.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NpyIOTest {

    @TempDir
    Path tempDir;

    @Test
    void testNpyRoundTripFloat32() throws IOException {
        Path file = tempDir.resolve("test_f32.npy");
        NDArray original = JNum.from(new float[]{1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f}, 2, 3);

        JNumIO.writeNpy(original, file);
        assertTrue(Files.exists(file));

        NDArray loaded = JNumIO.readNpy(file);

        assertEquals(DType.f32, loaded.getDType());
        assertArrayEquals(new long[]{2, 3}, loaded.getShape());
        assertEquals(1.5f, loaded.getFloat(0, 0), 1e-6f);
        assertEquals(6.5f, loaded.getFloat(1, 2), 1e-6f);
    }

    @Test
    void testNpyRoundTripFloat64() throws IOException {
        Path file = tempDir.resolve("test_f64.npy");
        NDArray original = JNum.from(new double[]{10.1, 20.2, 30.3, 40.4}, 2, 2);

        JNumIO.writeNpy(original, file);
        NDArray loaded = JNumIO.readNpy(file);

        assertEquals(DType.f64, loaded.getDType());
        assertArrayEquals(new long[]{2, 2}, loaded.getShape());
        assertEquals(10.1, loaded.getDouble(0, 0), 1e-9);
        assertEquals(40.4, loaded.getDouble(1, 1), 1e-9);
    }

    @Test
    void testNpyRoundTripInt32() throws IOException {
        Path file = tempDir.resolve("test_i32.npy");
        NDArray original = JNum.from(new int[]{100, -200, 300, -400}, 4);

        JNumIO.writeNpy(original, file);
        NDArray loaded = JNumIO.readNpy(file);

        assertEquals(DType.i32, loaded.getDType());
        assertArrayEquals(new long[]{4}, loaded.getShape());
        assertEquals(100, loaded.getInt(0));
        assertEquals(-400, loaded.getInt(3));
    }

    @Test
    void testNpyRoundTripTransposed() throws IOException {
        Path file = tempDir.resolve("test_transposed.npy");
        NDArray original = JNum.from(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).transpose();

        JNumIO.writeNpy(original, file);
        NDArray loaded = JNumIO.readNpy(file);

        assertEquals(DType.f32, loaded.getDType());
        assertArrayEquals(new long[]{3, 2}, loaded.getShape());
        assertEquals(1f, loaded.getFloat(0, 0), 1e-6f);
        assertEquals(4f, loaded.getFloat(0, 1), 1e-6f);
        assertEquals(5f, loaded.getFloat(1, 1), 1e-6f);
    }

    @Test
    void testNpyMmapRead() throws IOException {
        Path file = tempDir.resolve("test_mmap.npy");
        NDArray original = JNum.from(new float[]{7.7f, 8.8f, 9.9f}, 3);

        JNumIO.writeNpy(original, file);

        try (MmapArray mmap = JNumIO.readNpyMmapScoped(file)) {
            NDArray loaded = mmap.array();
            assertEquals(DType.f32, loaded.getDType());
            assertArrayEquals(new long[]{3}, loaded.getShape());
            assertEquals(7.7f, loaded.getFloat(0), 1e-6f);
            assertEquals(9.9f, loaded.getFloat(2), 1e-6f);
        }
    }

    @Test
    void testNpzRoundTrip() throws IOException {
        Path file = tempDir.resolve("test_archive.npz");
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray b = JNum.from(new double[]{4.0, 5.0}, 2);

        Map<String, NDArray> map = Map.of("weights", a, "biases", b);
        JNumIO.writeNpz(map, file);

        Map<String, NDArray> loadedMap = JNumIO.readNpz(file);
        assertEquals(2, loadedMap.size());
        assertTrue(loadedMap.containsKey("weights"));
        assertTrue(loadedMap.containsKey("biases"));

        NDArray loadedA = loadedMap.get("weights");
        assertEquals(DType.f32, loadedA.getDType());
        assertEquals(2f, loadedA.getFloat(1), 1e-6f);

        NDArray loadedB = loadedMap.get("biases");
        assertEquals(DType.f64, loadedB.getDType());
        assertEquals(5.0, loadedB.getDouble(1), 1e-9);
    }

    @Test
    void testAsyncIO() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("test_async.npy");
        NDArray original = JNum.from(new float[]{42f}, 1);

        JNumIO.writeNpyAsync(original, file).get();
        NDArray loaded = JNumIO.readNpyAsync(file).get();

        assertEquals(42f, loaded.getFloat(0), 1e-6f);
    }
}
