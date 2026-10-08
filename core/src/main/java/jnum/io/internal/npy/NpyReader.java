package jnum.io.internal.npy;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import jnum.NDArray;
import jnum.internal.layout.ShapeUtil;
import jnum.io.internal.common.ByteSwapUtil;

/**
 * Pure FFM reader for NumPy .npy files.
 */
public final class NpyReader {

    private NpyReader() {
        throw new AssertionError("NpyReader cannot be instantiated.");
    }

    /**
     * Reads a .npy file into an independent MemorySegment allocated in the provided Arena.
     */
    public static NDArray read(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            // Map temporarily to parse header and copy payload
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, tempArena);
                NpyHeader header = NpyHeaderParser.parse(fileSegment);

                long payloadBytes = header.payloadByteSize();
                MemorySegment dataSegment = arena.allocate(payloadBytes, 64);
                MemorySegment.copy(fileSegment, header.payloadOffset(), dataSegment, 0, payloadBytes);

                if (header.byteOrder() == ByteOrder.BIG_ENDIAN) {
                    ByteSwapUtil.swapInPlace(dataSegment, header.dtype(), header.totalElements());
                }

                long[] strides = calculateStrides(header.shape(), header.fortranOrder());
                NDArray array = NDArray.ofRaw(dataSegment, header.shape(), strides, header.dtype());

                return header.fortranOrder() ? array.contiguous(arena) : array;
            }
        }
    }

    /**
     * Reads directly from an in-memory segment (used by .npz and streaming).
     */
    public static NDArray readFromSegment(MemorySegment sourceSegment, Arena arena) {
        NpyHeader header = NpyHeaderParser.parse(sourceSegment);
        long payloadBytes = header.payloadByteSize();

        MemorySegment dataSegment = arena.allocate(payloadBytes, 64);
        MemorySegment.copy(sourceSegment, header.payloadOffset(), dataSegment, 0, payloadBytes);

        if (header.byteOrder() == ByteOrder.BIG_ENDIAN) {
            ByteSwapUtil.swapInPlace(dataSegment, header.dtype(), header.totalElements());
        }

        long[] strides = calculateStrides(header.shape(), header.fortranOrder());
        NDArray array = NDArray.ofRaw(dataSegment, header.shape(), strides, header.dtype());

        return header.fortranOrder() ? array.contiguous(arena) : array;
    }

    /**
     * Zero-copy memory-mapped load: maps the file directly into virtual memory.
     */
    public static NDArray readMmap(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
            NpyHeader header = NpyHeaderParser.parse(fileSegment);

            MemorySegment payloadSlice = fileSegment.asSlice(header.payloadOffset(), header.payloadByteSize());

            if (header.byteOrder() == ByteOrder.BIG_ENDIAN) {
                MemorySegment swapped = arena.allocate(header.payloadByteSize(), 64);
                MemorySegment.copy(payloadSlice, 0, swapped, 0, header.payloadByteSize());
                ByteSwapUtil.swapInPlace(swapped, header.dtype(), header.totalElements());
                payloadSlice = swapped;
            }

            long[] strides = calculateStrides(header.shape(), header.fortranOrder());
            return NDArray.ofRaw(payloadSlice, header.shape(), strides, header.dtype());
        }
    }

    private static long[] calculateStrides(long[] shape, boolean fortranOrder) {
        if (!fortranOrder) {
            return ShapeUtil.calculateDefaultStrides(shape);
        }
        long[] strides = new long[shape.length];
        long step = 1;
        for (int i = 0; i < shape.length; i++) {
            strides[i] = step;
            step *= shape[i];
        }
        return strides;
    }
}