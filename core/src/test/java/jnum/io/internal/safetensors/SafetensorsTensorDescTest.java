package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsTensorDescTest {

    @Test
    void testDescriptorFieldsAndByteSize() {
        SafetensorsTensorDesc desc = new SafetensorsTensorDesc("model.layer", DType.f64, new long[]{4, 4}, 100L, 228L);

        assertEquals("model.layer", desc.name());
        assertEquals(DType.f64, desc.dtype());
        assertArrayEquals(new long[]{4, 4}, desc.shape());
        assertEquals(100L, desc.startOffset());
        assertEquals(228L, desc.endOffset());
        assertEquals(128L, desc.byteSize());
    }
}
