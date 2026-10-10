package jnum.io.internal.csv;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.io.CsvOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CsvWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<CsvWriter> constructor = CsvWriter.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testDimensionGreaterThan2ThrowsException() {
        NDArray arr3D = JNum.zeros(DType.f64, 2, 2, 2);
        Path out = tempDir.resolve("out.csv");

        assertThrows(IllegalArgumentException.class, () ->
                CsvWriter.write(arr3D, CsvOptions.DEFAULT, out));
    }

    @Test
    void testWrite1DArray() throws IOException {
        Path out = tempDir.resolve("out_1d.csv");
        NDArray arr1D = JNum.from(new int[]{1, 2, 3}, 3);

        CsvWriter.write(arr1D, CsvOptions.DEFAULT, out);
        List<String> lines = Files.readAllLines(out);
        assertEquals(3, lines.size());
        assertEquals("1", lines.get(0));
        assertEquals("2", lines.get(1));
        assertEquals("3", lines.get(2));
    }

    @Test
    void testWriteNaNsWithNaString() throws IOException {
        Path out = tempDir.resolve("out_nan.csv");
        NDArray arr = JNum.from(new double[]{1.0, Double.NaN, 3.0, Double.NaN}, 2, 2);

        CsvOptions options = CsvOptions.builder().naString("MISSING").build();
        CsvWriter.write(arr, options, out);

        List<String> lines = Files.readAllLines(out);
        assertEquals("1.000000,MISSING", lines.get(0));
        assertEquals("3.000000,MISSING", lines.get(1));
    }

    @Test
    void testWriteBooleans() throws IOException {
        Path out = tempDir.resolve("out_bool.csv");
        NDArray arr = JNum.from(new boolean[]{true, false, false, true}, 2, 2);

        CsvWriter.write(arr, CsvOptions.builder().delimiter('\t').build(), out);
        List<String> lines = Files.readAllLines(out);
        assertEquals("1\t0", lines.get(0));
        assertEquals("0\t1", lines.get(1));
    }

    @Test
    void testWriteNonContiguousArray() throws IOException {
        Path out = tempDir.resolve("out_noncontig.csv");
        NDArray arr = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();

        CsvWriter.write(arr, CsvOptions.DEFAULT, out);
        List<String> lines = Files.readAllLines(out);
        assertEquals("1.000000,3.000000", lines.get(0));
        assertEquals("2.000000,4.000000", lines.get(1));
    }
}
