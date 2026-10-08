package jnum.io.internal.npy;

import java.nio.ByteOrder;
import jnum.DType;

/**
 * Strongly-typed representation of NumPy .npy header metadata.
 */
public record NpyHeader(
        int majorVersion,
        int minorVersion,
        int headerLength,
        DType dtype,
        ByteOrder byteOrder,
        boolean fortranOrder,
        long[] shape,
        long payloadOffset
) {
    public long totalElements() {
        if (shape.length == 0) return 1L; // 0-D scalar
        long total = 1L;
        for (long dim : shape) total *= dim;
        return total;
    }

    public long payloadByteSize() {
        return totalElements() * dtype.layout.byteSize();
    }
}