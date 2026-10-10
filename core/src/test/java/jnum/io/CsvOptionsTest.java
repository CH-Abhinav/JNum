package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import jnum.DType;
import org.junit.jupiter.api.Test;

class CsvOptionsTest {

    @Test
    void testDefaultValues() {
        CsvOptions def = CsvOptions.DEFAULT;
        assertEquals(',', def.delimiter());
        assertFalse(def.hasHeader());
        assertEquals(0, def.skipRows());
        assertEquals('\0', def.commentPrefix());
        assertEquals("NaN", def.naString());
        assertEquals(DType.f64, def.dtype());
        assertEquals("%.6f", def.floatFormat());
    }

    @Test
    void testPredefinedConstants() {
        CsvOptions tsv = CsvOptions.TSV;
        assertEquals('\t', tsv.delimiter());
        assertEquals(DType.f64, tsv.dtype());

        CsvOptions ws = CsvOptions.WHITESPACE;
        assertEquals(' ', ws.delimiter());
        assertEquals(DType.f64, ws.dtype());
    }

    @Test
    void testBuilderCustomization() {
        CsvOptions options = CsvOptions.builder()
                .delimiter(';')
                .hasHeader(true)
                .skipRows(3)
                .commentPrefix('#')
                .naString("NULL")
                .dtype(DType.f32)
                .floatFormat("%.2f")
                .build();

        assertEquals(';', options.delimiter());
        assertTrue(options.hasHeader());
        assertEquals(3, options.skipRows());
        assertEquals('#', options.commentPrefix());
        assertEquals("NULL", options.naString());
        assertEquals(DType.f32, options.dtype());
        assertEquals("%.2f", options.floatFormat());
    }

    @Test
    void testBuilderNegativeSkipRowsClamped() {
        CsvOptions options = CsvOptions.builder()
                .skipRows(-5)
                .build();

        assertEquals(0, options.skipRows());
    }

    @Test
    void testRecordEqualityAndHashCode() {
        CsvOptions opt1 = CsvOptions.builder().delimiter('|').dtype(DType.i32).build();
        CsvOptions opt2 = CsvOptions.builder().delimiter('|').dtype(DType.i32).build();
        CsvOptions opt3 = CsvOptions.builder().delimiter(',').dtype(DType.i32).build();

        assertEquals(opt1, opt2);
        assertEquals(opt1.hashCode(), opt2.hashCode());
        assertNotEquals(opt1, opt3);
        assertTrue(opt1.toString().contains("delimiter=|"));
    }
}
