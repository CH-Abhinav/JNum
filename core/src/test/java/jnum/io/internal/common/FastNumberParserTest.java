package jnum.io.internal.common;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class FastNumberParserTest {

    private MemorySegment toSegment(String s) {
        return MemorySegment.ofArray(s.getBytes(StandardCharsets.US_ASCII));
    }

    private double parse(String s) {
        MemorySegment seg = toSegment(s);
        return FastNumberParser.parseDouble(seg, 0, seg.byteSize(), "NA", Double.NaN);
    }

    private double parseWithNa(String s, String naString, double naVal) {
        MemorySegment seg = toSegment(s);
        return FastNumberParser.parseDouble(seg, 0, seg.byteSize(), naString, naVal);
    }

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<FastNumberParser> constructor = FastNumberParser.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testStandardNumbers() {
        assertEquals(123.0, parse("123"), 1e-9);
        assertEquals(-456.0, parse("-456"), 1e-9);
        assertEquals(789.0, parse("+789"), 1e-9);
        assertEquals(123.456, parse("123.456"), 1e-9);
        assertEquals(-0.789, parse("-0.789"), 1e-9);
        assertEquals(0.0, parse("0"), 1e-9);
        assertEquals(0.0, parse("0.0"), 1e-9);
    }

    @Test
    void testWhitespaceTrimming() {
        assertEquals(42.5, parse("   42.5  \r\n\t"), 1e-9);
        assertEquals(-10.0, parse("\t-10.0 \n"), 1e-9);
    }

    @Test
    void testEmptyOrWhitespaceOnly() {
        assertTrue(Double.isNaN(parse("")));
        assertTrue(Double.isNaN(parse("    \t\r\n")));
        assertEquals(-999.0, parseWithNa("   ", "NA", -999.0), 1e-9);
    }

    @Test
    void testScientificNotation() {
        assertEquals(1e5, parse("1e5"), 1e-9);
        assertEquals(1.23e-4, parse("1.23e-4"), 1e-9);
        assertEquals(-5.67e2, parse("-5.67E2"), 1e-9);
        assertEquals(3.14e+3, parse("3.14E+3"), 1e-9);
    }

    @Test
    void testSentinelNAValues() {
        assertEquals(-99.0, parseWithNa("NA", "NA", -99.0), 1e-9);
        assertEquals(-99.0, parseWithNa("NULL", "NULL", -99.0), 1e-9);
        assertEquals(12.0, parseWithNa("12", "NULL", -99.0), 1e-9);
    }

    @Test
    void testSpecialFloatingPointLiterals() {
        assertTrue(Double.isNaN(parse("nan")));
        assertTrue(Double.isNaN(parse("NaN")));
        assertTrue(Double.isNaN(parse("NAN")));

        assertEquals(Double.POSITIVE_INFINITY, parse("inf"));
        assertEquals(Double.POSITIVE_INFINITY, parse("+INF"));
        assertEquals(Double.POSITIVE_INFINITY, parse("Infinity"));
        assertEquals(Double.NEGATIVE_INFINITY, parse("-inf"));
        assertEquals(Double.NEGATIVE_INFINITY, parse("-Infinity"));
    }

    @Test
    void testMalformedFallback() {
        assertTrue(Double.isNaN(parse("not_a_number")));
        assertEquals(-1.0, parseWithNa("xyz", "NA", -1.0), 1e-9);
    }
}
