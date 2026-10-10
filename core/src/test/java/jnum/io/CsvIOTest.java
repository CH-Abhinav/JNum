package jnum.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutionException;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CsvIOTest {

    @TempDir
    Path tempDir;

    @Test
    void testCsvRoundTripFloat64() throws IOException {
        Path file = tempDir.resolve("matrix.csv");
        NDArray original = JNum.from(new double[]{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, 2, 3);

        JNumIO.writeCsv(original, file);
        assertTrue(Files.exists(file));

        NDArray loaded = JNumIO.readCsv(file);

        assertEquals(DType.f64, loaded.getDType());
        assertArrayEquals(new long[]{2, 3}, loaded.getShape());
        assertEquals(1.5, loaded.getDouble(0, 0), 1e-5);
        assertEquals(3.5, loaded.getDouble(0, 2), 1e-5);
        assertEquals(6.5, loaded.getDouble(1, 2), 1e-5);
    }

    @Test
    void testTsvRoundTripFloat32() throws IOException {
        Path file = tempDir.resolve("matrix.tsv");
        NDArray original = JNum.from(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, 2, 2);

        CsvOptions options = CsvOptions.builder()
                .delimiter('\t')
                .dtype(DType.f32)
                .build();

        JNumIO.writeCsv(original, options, file);
        NDArray loaded = JNumIO.readCsv(file, options);

        assertEquals(DType.f32, loaded.getDType());
        assertArrayEquals(new long[]{2, 2}, loaded.getShape());
        assertEquals(10.0f, loaded.getFloat(0, 0), 1e-5f);
        assertEquals(40.0f, loaded.getFloat(1, 1), 1e-5f);
    }

    @Test
    void testHeaderAndCommentSkipping() throws IOException {
        Path file = tempDir.resolve("with_metadata.csv");
        String content = """
                # This is a leading comment
                # Another metadata line
                col_a,col_b
                1.0,2.0
                3.0,4.0
                # Inline comment to ignore
                5.0,6.0
                """;
        Files.writeString(file, content);

        CsvOptions options = CsvOptions.builder()
                .hasHeader(true)
                .commentPrefix('#')
                .dtype(DType.f64)
                .build();

        NDArray loaded = JNumIO.readCsv(file, options);

        assertArrayEquals(new long[]{3, 2}, loaded.getShape());
        assertEquals(1.0, loaded.getDouble(0, 0), 1e-6);
        assertEquals(4.0, loaded.getDouble(1, 1), 1e-6);
        assertEquals(5.0, loaded.getDouble(2, 0), 1e-6);
        assertEquals(6.0, loaded.getDouble(2, 1), 1e-6);
    }

    @Test
    void testScientificNotation() throws IOException {
        Path file = tempDir.resolve("scientific.csv");
        String content = """
                1.23e-4,-5.67E2
                1e5,-2.5e-1
                """;
        Files.writeString(file, content);

        NDArray loaded = JNumIO.readCsv(file, DType.f64);

        assertArrayEquals(new long[]{2, 2}, loaded.getShape());
        assertEquals(1.23e-4, loaded.getDouble(0, 0), 1e-8);
        assertEquals(-567.0, loaded.getDouble(0, 1), 1e-6);
        assertEquals(100000.0, loaded.getDouble(1, 0), 1e-3);
        assertEquals(-0.25, loaded.getDouble(1, 1), 1e-6);
    }

    @Test
    void testMissingValuesAndNaN() throws IOException {
        Path file = tempDir.resolve("missing.csv");
        String content = """
                1.0,NA,3.0
                NaN,5.0,null
                """;
        Files.writeString(file, content);

        CsvOptions options = CsvOptions.builder()
                .naString("NA")
                .build();

        NDArray loaded = JNumIO.readCsv(file, options);

        assertArrayEquals(new long[]{2, 3}, loaded.getShape());
        assertEquals(1.0, loaded.getDouble(0, 0), 1e-6);
        assertTrue(Double.isNaN(loaded.getDouble(0, 1)));
        assertEquals(3.0, loaded.getDouble(0, 2), 1e-6);
        assertTrue(Double.isNaN(loaded.getDouble(1, 0)));
        assertEquals(5.0, loaded.getDouble(1, 1), 1e-6);
        assertTrue(Double.isNaN(loaded.getDouble(1, 2)));
    }

    @Test
    void testAsyncCsv() throws ExecutionException, InterruptedException {
        Path file = tempDir.resolve("async.csv");
        NDArray original = JNum.from(new double[]{42.0, 84.0}, 1, 2);

        JNumIO.writeCsvAsync(original, file).get();
        NDArray loaded = JNumIO.readCsvAsync(file).get();

        assertArrayEquals(new long[]{1, 2}, loaded.getShape());
        assertEquals(42.0, loaded.getDouble(0, 0), 1e-5);
        assertEquals(84.0, loaded.getDouble(0, 1), 1e-5);
    }
}
