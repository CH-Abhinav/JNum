package jnum.io.internal.npy;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import jnum.DType;
import org.junit.jupiter.api.Test;

class NpyHeaderParserTest {

    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<NpyHeaderParser> constructor = NpyHeaderParser.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testSegmentTooSmall() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(8);
            assertThrows(IllegalArgumentException.class, () -> NpyHeaderParser.parse(seg));
        }
    }

    @Test
    void testMagicMismatch() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(20);
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 0, (byte) 'B');
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 1, (byte) 'A');
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 2, (byte) 'D');
            assertThrows(IllegalArgumentException.class, () -> NpyHeaderParser.parse(seg));
        }
    }

    @Test
    void testParseV1Header() {
        String dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }\n";
        byte[] dictBytes = dict.getBytes(StandardCharsets.US_ASCII);
        int headerLen = dictBytes.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(10 + headerLen);
            for (int i = 0; i < 6; i++) seg.set(ValueLayout.JAVA_BYTE, i, NpyHeaderParser.MAGIC[i]);
            seg.set(ValueLayout.JAVA_BYTE, 6, (byte) 1); // v1
            seg.set(ValueLayout.JAVA_BYTE, 7, (byte) 0); // .0
            seg.set(LE_SHORT, 8, (short) headerLen);
            MemorySegment.copy(MemorySegment.ofArray(dictBytes), 0, seg, 10, headerLen);

            NpyHeader header = NpyHeaderParser.parse(seg);
            assertEquals(1, header.majorVersion());
            assertEquals(0, header.minorVersion());
            assertEquals(DType.f32, header.dtype());
            assertEquals(ByteOrder.LITTLE_ENDIAN, header.byteOrder());
            assertFalse(header.fortranOrder());
            assertArrayEquals(new long[]{2, 3}, header.shape());
            assertEquals(10 + headerLen, header.payloadOffset());
        }
    }

    @Test
    void testParseV2HeaderBigEndianFortranScalar() {
        String dict = "{'descr': '>f8', 'fortran_order': True, 'shape': (), }\n";
        byte[] dictBytes = dict.getBytes(StandardCharsets.US_ASCII);
        int headerLen = dictBytes.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(12 + headerLen);
            for (int i = 0; i < 6; i++) seg.set(ValueLayout.JAVA_BYTE, i, NpyHeaderParser.MAGIC[i]);
            seg.set(ValueLayout.JAVA_BYTE, 6, (byte) 2); // v2
            seg.set(ValueLayout.JAVA_BYTE, 7, (byte) 0); // .0
            seg.set(LE_INT, 8, headerLen);
            MemorySegment.copy(MemorySegment.ofArray(dictBytes), 0, seg, 12, headerLen);

            NpyHeader header = NpyHeaderParser.parse(seg);
            assertEquals(2, header.majorVersion());
            assertEquals(0, header.minorVersion());
            assertEquals(DType.f64, header.dtype());
            assertEquals(ByteOrder.BIG_ENDIAN, header.byteOrder());
            assertTrue(header.fortranOrder());
            assertArrayEquals(new long[0], header.shape());
            assertEquals(1L, header.totalElements());
        }
    }

    @Test
    void testUnsupportedVersionThrows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(20);
            for (int i = 0; i < 6; i++) seg.set(ValueLayout.JAVA_BYTE, i, NpyHeaderParser.MAGIC[i]);
            seg.set(ValueLayout.JAVA_BYTE, 6, (byte) 5); // v5 unsupported
            seg.set(ValueLayout.JAVA_BYTE, 7, (byte) 0);

            assertThrows(UnsupportedOperationException.class, () -> NpyHeaderParser.parse(seg));
        }
    }
}
