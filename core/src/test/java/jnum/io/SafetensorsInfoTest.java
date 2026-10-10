package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsInfoTest {

    @Test
    void testTensorInfoRecord() {
        long[] shape = new long[]{3, 4};
        SafetensorsInfo.TensorInfo tensorInfo = new SafetensorsInfo.TensorInfo("layer.weight", DType.f32, shape, 48L);

        assertEquals("layer.weight", tensorInfo.name());
        assertEquals(DType.f32, tensorInfo.dtype());
        assertArrayEquals(shape, tensorInfo.shape());
        assertEquals(48L, tensorInfo.byteSize());
    }

    @Test
    void testSafetensorsInfoRecord() {
        SafetensorsInfo.TensorInfo t1 = new SafetensorsInfo.TensorInfo("w", DType.f64, new long[]{2, 2}, 32L);
        Map<String, SafetensorsInfo.TensorInfo> tensors = Map.of("w", t1);
        Map<String, String> metadata = Map.of("format", "pt", "author", "user");

        SafetensorsInfo info = new SafetensorsInfo(tensors, metadata, 128L);

        assertEquals(1, info.tensors().size());
        assertEquals(t1, info.tensors().get("w"));
        assertEquals(2, info.userMetadata().size());
        assertEquals("pt", info.userMetadata().get("format"));
        assertEquals("user", info.userMetadata().get("author"));
        assertEquals(128L, info.headerByteSize());
    }
}
