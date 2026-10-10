package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafetensorsWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<SafetensorsWriter> constructor = SafetensorsWriter.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testWriteNonContiguousAndSpecialCharacters() throws IOException {
        Path file = tempDir.resolve("special.safetensors");
        NDArray arr = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        String tensorName = "model.layer_0.weight";
        Map<String, NDArray> tensors = Map.of(tensorName, arr);
        Map<String, String> metadata = Map.of("format", "jnum", "version", "1.0");

        SafetensorsWriter.write(tensors, metadata, file);
        assertTrue(Files.exists(file));

        SafetensorsMetadata meta = SafetensorsReader.readMetadata(file);
        assertEquals(1, meta.tensors().size());
        assertTrue(meta.tensors().containsKey(tensorName));
        assertEquals("1.0", meta.userMetadata().get("version"));
    }
}
