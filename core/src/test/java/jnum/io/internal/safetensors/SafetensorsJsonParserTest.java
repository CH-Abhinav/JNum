package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsJsonParserTest {

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<SafetensorsJsonParser> constructor = SafetensorsJsonParser.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testParseValidJsonWithMetadata() {
        String json = """
                {
                    "__metadata__": {
                        "author": "tester",
                        "framework": "jnum"
                    },
                    "dense.weight": {
                        "dtype": "F32",
                        "shape": [2, 3],
                        "data_offsets": [0, 24]
                    },
                    "dense.bias": {
                        "dtype": "I32",
                        "shape": [3],
                        "data_offsets": [24, 36]
                    }
                }
                """;

        SafetensorsMetadata meta = SafetensorsJsonParser.parse(json, 256L);
        assertEquals(256L, meta.headerLength());
        assertEquals(2, meta.userMetadata().size());
        assertEquals("tester", meta.userMetadata().get("author"));
        assertEquals("jnum", meta.userMetadata().get("framework"));

        assertEquals(2, meta.tensors().size());
        SafetensorsTensorDesc w = meta.tensors().get("dense.weight");
        assertNotNull(w);
        assertEquals("dense.weight", w.name());
        assertEquals(DType.f32, w.dtype());
        assertArrayEquals(new long[]{2, 3}, w.shape());
        assertEquals(0L, w.startOffset());
        assertEquals(24L, w.endOffset());
        assertEquals(24L, w.byteSize());
    }

    @Test
    void testParseAllSupportedDTypes() {
        String json = """
                {
                    "t_f32": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "t_f64": {"dtype": "F64", "shape": [1], "data_offsets": [4, 12]},
                    "t_i32": {"dtype": "I32", "shape": [1], "data_offsets": [12, 16]},
                    "t_bool": {"dtype": "BOOL", "shape": [1], "data_offsets": [16, 17]}
                }
                """;

        SafetensorsMetadata meta = SafetensorsJsonParser.parse(json, 100L);
        assertEquals(DType.f32, meta.tensors().get("t_f32").dtype());
        assertEquals(DType.f64, meta.tensors().get("t_f64").dtype());
        assertEquals(DType.i32, meta.tensors().get("t_i32").dtype());
        assertEquals(DType.bool, meta.tensors().get("t_bool").dtype());
    }

    @Test
    void testInvalidRootThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                SafetensorsJsonParser.parse("[1, 2, 3]", 10L));
    }

    @Test
    void testInvalidOffsetsLengthThrows() {
        String json = """
                {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}
                """;
        assertThrows(IllegalArgumentException.class, () ->
                SafetensorsJsonParser.parse(json, 10L));
    }

    @Test
    void testUnsupportedDTypeThrows() {
        String json = """
                {"t": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
                """;
        assertThrows(UnsupportedOperationException.class, () ->
                SafetensorsJsonParser.parse(json, 10L));
    }
}
