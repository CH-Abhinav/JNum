package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsMetadataTest {

    @Test
    void testRecordFields() {
        SafetensorsTensorDesc desc = new SafetensorsTensorDesc("t", DType.f32, new long[]{1}, 0L, 4L);
        Map<String, SafetensorsTensorDesc> tensors = Map.of("t", desc);
        Map<String, String> meta = Map.of("ver", "1.0");

        SafetensorsMetadata metadata = new SafetensorsMetadata(tensors, meta, 80L);

        assertEquals(tensors, metadata.tensors());
        assertEquals(meta, metadata.userMetadata());
        assertEquals(80L, metadata.headerLength());
    }
}
