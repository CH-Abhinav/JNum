package jnum.io.internal.npy;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteOrder;
import jnum.DType;
import org.junit.jupiter.api.Test;

class NpyHeaderTest {

    @Test
    void testScalarHeader() {
        NpyHeader header = new NpyHeader(1, 0, 64, DType.f32, ByteOrder.LITTLE_ENDIAN, false, new long[0], 74L);
        assertEquals(1L, header.totalElements());
        assertEquals(4L, header.payloadByteSize());
        assertEquals(1, header.majorVersion());
        assertEquals(0, header.minorVersion());
        assertEquals(64, header.headerLength());
        assertEquals(74L, header.payloadOffset());
        assertFalse(header.fortranOrder());
    }

    @Test
    void testMultidimensionalHeader() {
        NpyHeader header = new NpyHeader(2, 0, 128, DType.f64, ByteOrder.BIG_ENDIAN, true, new long[]{2, 3, 4}, 140L);
        assertEquals(24L, header.totalElements());
        assertEquals(24L * 8, header.payloadByteSize());
        assertEquals(ByteOrder.BIG_ENDIAN, header.byteOrder());
        assertTrue(header.fortranOrder());
        assertArrayEquals(new long[]{2, 3, 4}, header.shape());
    }
}
