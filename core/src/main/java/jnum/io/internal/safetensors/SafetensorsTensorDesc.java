package jnum.io.internal.safetensors;

import jnum.DType;

/**
 * Metadata descriptor for an individual tensor within a .safetensors container.
 */
public record SafetensorsTensorDesc(
        String name,
        DType dtype,
        long[] shape,
        long startOffset,
        long endOffset
) {
    public long byteSize() {
        return endOffset - startOffset;
    }
}
