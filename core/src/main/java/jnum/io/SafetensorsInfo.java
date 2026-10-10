package jnum.io;

import java.util.Map;
import jnum.DType;

/**
 * Inspection record containing metadata of a .safetensors file without loading tensor payload.
 */
public record SafetensorsInfo(
        Map<String, TensorInfo> tensors,
        Map<String, String> userMetadata,
        long headerByteSize
) {
    public record TensorInfo(
            String name,
            DType dtype,
            long[] shape,
            long byteSize
    ) {}
}
