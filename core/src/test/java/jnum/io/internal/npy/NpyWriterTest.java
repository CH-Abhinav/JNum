package jnum.io.internal.npy;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NpyWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<NpyWriter> constructor = NpyWriter.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testWriteContiguousAndNonContiguous() throws IOException {
        Path p1 = tempDir.resolve("contig.npy");
        Path p2 = tempDir.resolve("noncontig.npy");

        NDArray orig = JNum.from(new int[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray transposed = orig.transpose();

        NpyWriter.write(orig, p1);
        NpyWriter.write(transposed, p2);

        assertTrue(Files.exists(p1));
        assertTrue(Files.exists(p2));

        NDArray r1 = NpyReader.read(p1, Arena.ofAuto());
        NDArray r2 = NpyReader.read(p2, Arena.ofAuto());

        assertArrayEquals(new long[]{2, 3}, r1.getShape());
        assertEquals(orig.getInt(0, 1), r1.getInt(0, 1));

        assertArrayEquals(new long[]{3, 2}, r2.getShape());
        assertEquals(transposed.getInt(1, 0), r2.getInt(1, 0));
    }

    @Test
    void testWriteToSegment() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray arr = JNum.from(new double[]{100.5, 200.5}, 2);
            MemorySegment seg = NpyWriter.writeToSegment(arr, arena);

            NpyHeader header = NpyHeaderParser.parse(seg);
            assertEquals(DType.f64, header.dtype());
            assertArrayEquals(new long[]{2}, header.shape());
            assertEquals(10L + header.headerLength(), header.payloadOffset());
        }
    }
}
