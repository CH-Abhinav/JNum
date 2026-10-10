package jnum.io.internal.csv;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import jnum.DType;
import jnum.NDArray;
import jnum.io.CsvOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CsvReaderTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<CsvReader> constructor = CsvReader.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testEmptyFileThrowsException() throws IOException {
        Path empty = tempDir.resolve("empty.csv");
        Files.writeString(empty, "");

        assertThrows(IllegalArgumentException.class, () ->
                CsvReader.read(empty, CsvOptions.DEFAULT, Arena.ofAuto()));
    }

    @Test
    void testNoDataThrowsException() throws IOException {
        Path commentsOnly = tempDir.resolve("comments.csv");
        Files.writeString(commentsOnly, "# Comment line 1\n# Comment line 2\n");

        CsvOptions options = CsvOptions.builder().commentPrefix('#').build();
        assertThrows(IllegalArgumentException.class, () ->
                CsvReader.read(commentsOnly, options, Arena.ofAuto()));
    }

    @Test
    void testReadVariousDTypes() throws IOException {
        Path file = tempDir.resolve("data.csv");
        Files.writeString(file, "1,0\n0,1\n");

        try (Arena arena = Arena.ofConfined()) {
            // Int32
            NDArray arrI32 = CsvReader.read(file, CsvOptions.builder().dtype(DType.i32).build(), arena);
            assertEquals(DType.i32, arrI32.getDType());
            assertEquals(1, arrI32.getInt(0, 0));
            assertEquals(0, arrI32.getInt(0, 1));

            // Bool
            NDArray arrBool = CsvReader.read(file, CsvOptions.builder().dtype(DType.bool).build(), arena);
            assertEquals(DType.bool, arrBool.getDType());
            assertTrue(arrBool.getBoolean(0, 0));
            assertFalse(arrBool.getBoolean(0, 1));
        }
    }

    @Test
    void testRaggedRowsPadding() throws IOException {
        Path ragged = tempDir.resolve("ragged.csv");
        // Row 1 has 3 columns, Row 2 has 2 columns
        Files.writeString(ragged, "1.0,2.0,3.0\n4.0,5.0\n");

        try (Arena arena = Arena.ofConfined()) {
            NDArray arr = CsvReader.read(ragged, CsvOptions.DEFAULT, arena);
            assertArrayEquals(new long[]{2, 3}, arr.getShape());
            assertEquals(1.0, arr.getDouble(0, 0), 1e-6);
            assertEquals(5.0, arr.getDouble(1, 1), 1e-6);
            assertTrue(Double.isNaN(arr.getDouble(1, 2))); // Padded with NaN
        }
    }

    @Test
    void testWindowsLineEndingsCRLF() throws IOException {
        Path crlf = tempDir.resolve("crlf.csv");
        Files.writeString(crlf, "10.0,20.0\r\n30.0,40.0\r\n");

        try (Arena arena = Arena.ofConfined()) {
            NDArray arr = CsvReader.read(crlf, CsvOptions.DEFAULT, arena);
            assertArrayEquals(new long[]{2, 2}, arr.getShape());
            assertEquals(10.0, arr.getDouble(0, 0), 1e-6);
            assertEquals(40.0, arr.getDouble(1, 1), 1e-6);
        }
    }
}
