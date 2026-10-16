package jnum.io;

import java.util.Map;
import jnum.DType;

/**
 * Inspection record containing metadata of a {@code .safetensors} file without loading tensor payload bytes.
 *
 * <p>Enables ultra-fast header inspection of multi-gigabyte neural network checkpoints in microseconds.</p>
 *
 * @param tensors map of tensor names to their metadata descriptors
 * @param userMetadata key-value string metadata embedded in the header (e.g. format version, model architecture)
 * @param headerByteSize size in bytes of the JSON header prefix
 */
public record SafetensorsInfo(
        Map<String, TensorInfo> tensors,
        Map<String, String> userMetadata,
        long headerByteSize
) {
    /**
     * Metadata descriptor for an individual tensor within a {@code .safetensors} file.
     *
     * @param name unique name of the tensor (e.g. {@code "model.layers.0.weight"})
     * @param dtype data type descriptor
     * @param shape dimensional extent array
     * @param byteSize total payload size in bytes
     */
    public record TensorInfo(
            String name,
            DType dtype,
            long[] shape,
            long byteSize
    ) {}
}

