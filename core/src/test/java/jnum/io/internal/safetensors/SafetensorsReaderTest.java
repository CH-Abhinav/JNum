package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafetensorsReaderTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<SafetensorsReader> constructor = SafetensorsReader.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testFileTooSmallThrows() throws IOException {
        Path tiny = tempDir.resolve("tiny.safetensors");
        Files.write(tiny, new byte[]{1, 2, 3});

        assertThrows(IllegalArgumentException.class, () ->
                SafetensorsReader.readMetadata(tiny));
    }

    @Test
    void testCorruptedHeaderLengthThrows() throws IOException {
        Path corrupt = tempDir.resolve("corrupt.safetensors");
        // 8 bytes representing huge header length 999999999L
        byte[] bytes = new byte[16];
        bytes[0] = (byte) 0xFF;
        bytes[1] = (byte) 0xFF;
        Files.write(corrupt, bytes);

        assertThrows(IllegalArgumentException.class, () ->
                SafetensorsReader.readMetadata(corrupt));
    }

    @Test
    void testReadMmapAndReadSingle() throws IOException {
        Path file = tempDir.resolve("model.safetensors");
        NDArray t1 = JNum.from(new float[]{1.0f, 2.0f}, 2);
        NDArray t2 = JNum.from(new double[]{3.0, 4.0}, 2);
        SafetensorsWriter.write(Map.of("t1", t1, "t2", t2), null, file);

        try (Arena arena = Arena.ofShared()) {
            Map<String, NDArray> map = SafetensorsReader.readMmap(file, arena);
            assertEquals(2, map.size());
            assertEquals(1.0f, map.get("t1").getFloat(0), 1e-6f);
            assertEquals(4.0, map.get("t2").getDouble(1), 1e-6);

            NDArray single = SafetensorsReader.readSingle(file, "t1", arena);
            assertEquals(2.0f, single.getFloat(1), 1e-6f);

            assertThrows(IllegalArgumentException.class, () ->
                    SafetensorsReader.readSingle(file, "missing", arena));
        }
    }
}
