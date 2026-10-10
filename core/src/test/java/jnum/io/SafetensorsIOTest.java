package jnum.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafetensorsIOTest {

    @TempDir
    Path tempDir;

    @Test
    void testSafetensorsRoundTrip() throws IOException {
        Path file = tempDir.resolve("model.safetensors");

        NDArray w1 = JNum.from(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
        NDArray b1 = JNum.from(new float[]{0.1f, 0.2f}, 2);
        NDArray w2 = JNum.from(new double[]{10.5, 20.5, 30.5, 40.5}, 2, 2);
        NDArray counts = JNum.from(new int[]{100, 200, 300}, 3);

        Map<String, NDArray> tensors = Map.of(
                "linear1.weight", w1,
                "linear1.bias", b1,
                "linear2.weight", w2,
                "step_counts", counts
        );

        Map<String, String> metadata = Map.of("format", "jnum", "author", "user");

        JNumIO.writeSafetensors(tensors, metadata, file);

        Map<String, NDArray> loaded = JNumIO.readSafetensors(file);
        assertEquals(4, loaded.size());

        NDArray loadedW1 = loaded.get("linear1.weight");
        assertNotNull(loadedW1);
        assertEquals(DType.f32, loadedW1.getDType());
        assertArrayEquals(new long[]{2, 3}, loadedW1.getShape());
        assertEquals(1.0f, loadedW1.getFloat(0, 0), 1e-6f);
        assertEquals(6.0f, loadedW1.getFloat(1, 2), 1e-6f);

        NDArray loadedW2 = loaded.get("linear2.weight");
        assertNotNull(loadedW2);
        assertEquals(DType.f64, loadedW2.getDType());
        assertArrayEquals(new long[]{2, 2}, loadedW2.getShape());
        assertEquals(10.5, loadedW2.getDouble(0, 0), 1e-9);
        assertEquals(40.5, loadedW2.getDouble(1, 1), 1e-9);

        NDArray loadedCounts = loaded.get("step_counts");
        assertNotNull(loadedCounts);
        assertEquals(DType.i32, loadedCounts.getDType());
        assertEquals(100, loadedCounts.getInt(0));
        assertEquals(300, loadedCounts.getInt(2));
    }

    @Test
    void testSafetensorsInspection() throws IOException {
        Path file = tempDir.resolve("inspect_model.safetensors");

        NDArray w = JNum.from(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
        Map<String, NDArray> tensors = Map.of("conv.weight", w);
        Map<String, String> metadata = Map.of("version", "1.0");

        JNumIO.writeSafetensors(tensors, metadata, file);

        SafetensorsInfo info = JNumIO.inspectSafetensors(file);
        assertEquals(1, info.tensors().size());
        assertTrue(info.tensors().containsKey("conv.weight"));

        SafetensorsInfo.TensorInfo tInfo = info.tensors().get("conv.weight");
        assertEquals("conv.weight", tInfo.name());
        assertEquals(DType.f32, tInfo.dtype());
        assertArrayEquals(new long[]{2, 2}, tInfo.shape());
        assertEquals(16L, tInfo.byteSize());

        assertEquals("1.0", info.userMetadata().get("version"));
        assertTrue(info.headerByteSize() > 0);
    }

    @Test
    void testSafetensorsMmap() throws IOException {
        Path file = tempDir.resolve("mmap_model.safetensors");

        NDArray w = JNum.from(new float[]{7.0f, 8.0f, 9.0f}, 3);
        JNumIO.writeSafetensors(Map.of("w", w), file);

        try (MmapTensors mmap = JNumIO.readSafetensorsMmapScoped(file)) {
            NDArray loaded = mmap.get("w");
            assertNotNull(loaded);
            assertEquals(DType.f32, loaded.getDType());
            assertArrayEquals(new long[]{3}, loaded.getShape());
            assertEquals(7.0f, loaded.getFloat(0), 1e-6f);
            assertEquals(9.0f, loaded.getFloat(2), 1e-6f);
        }
    }

    @Test
    void testSafetensorsSingleTensor() throws IOException {
        Path file = tempDir.resolve("single_model.safetensors");

        NDArray a = JNum.from(new float[]{1.0f, 2.0f}, 2);
        NDArray b = JNum.from(new float[]{3.0f, 4.0f}, 2);
        JNumIO.writeSafetensors(Map.of("tensorA", a, "tensorB", b), file);

        NDArray loadedB = JNumIO.readSafetensor(file, "tensorB");
        assertNotNull(loadedB);
        assertEquals(3.0f, loadedB.getFloat(0), 1e-6f);
        assertEquals(4.0f, loadedB.getFloat(1), 1e-6f);
    }

    @Test
    void testSafetensorsAsync() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("async_model.safetensors");

        NDArray a = JNum.from(new float[]{99.0f}, 1);
        Map<String, NDArray> tensors = Map.of("val", a);

        JNumIO.writeSafetensorsAsync(tensors, file).get();
        Map<String, NDArray> loaded = JNumIO.readSafetensorsAsync(file).get();

        assertEquals(99.0f, loaded.get("val").getFloat(0), 1e-6f);
    }

    @Test
    void testSafetensorsTransposed() throws IOException {
        Path file = tempDir.resolve("transposed.safetensors");

        NDArray transposed = JNum.from(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).transpose();
        JNumIO.writeSafetensors(Map.of("matrix", transposed), file);

        NDArray loaded = JNumIO.readSafetensor(file, "matrix");
        assertArrayEquals(new long[]{3, 2}, loaded.getShape());
        assertEquals(1f, loaded.getFloat(0, 0), 1e-6f);
        assertEquals(4f, loaded.getFloat(0, 1), 1e-6f);
        assertEquals(5f, loaded.getFloat(1, 1), 1e-6f);
    }
}
