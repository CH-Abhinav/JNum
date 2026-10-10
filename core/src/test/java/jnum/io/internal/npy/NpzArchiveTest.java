package jnum.io.internal.npy;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NpzArchiveTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<NpzArchive> constructor = NpzArchive.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testArchiveRoundTripWithAndWithoutExtension() throws IOException {
        Path zip = tempDir.resolve("test.npz");

        NDArray a = JNum.from(new float[]{1.0f, 2.0f}, 2);
        NDArray b = JNum.from(new int[]{10, 20, 30}, 3);

        // One key with .npy, one without
        Map<String, NDArray> input = Map.of("arr1", a, "arr2.npy", b);

        NpzArchive.write(input, zip);
        assertTrue(Files.exists(zip));

        try (Arena arena = Arena.ofConfined()) {
            Map<String, NDArray> loaded = NpzArchive.read(zip, arena);
            assertEquals(2, loaded.size());
            assertTrue(loaded.containsKey("arr1"));
            assertTrue(loaded.containsKey("arr2"));

            assertEquals(1.0f, loaded.get("arr1").getFloat(0), 1e-6f);
            assertEquals(30, loaded.get("arr2").getInt(2));
        }
    }
}
