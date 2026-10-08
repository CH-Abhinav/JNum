package jnum.io.internal.npy;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import jnum.NDArray;

/**
 * Pure FFM writer serializing NDArrays to disk via memory mapping.
 */
public final class NpyWriter {

    private NpyWriter() {
        throw new AssertionError("NpyWriter cannot be instantiated.");
    }

    public static void write(NDArray array, Path path) throws IOException {
        NDArray contiguous = array.isContiguous() ? array : array.contiguous();
        long payloadBytes = contiguous.getSize() * contiguous.getDType().layout.byteSize();

        NpyHeaderWriter.HeaderInfo info = NpyHeaderWriter.prepareHeader(
                contiguous.getDType(), contiguous.getShape(), false);
        long totalFileSize = info.totalHeaderBytes() + payloadBytes;

        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.READ,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalFileSize, arena);
                NpyHeaderWriter.writeHeaderToSegment(fileSegment, info);
                MemorySegment.copy(contiguous.getData(), 0, fileSegment, info.totalHeaderBytes(), payloadBytes);
            }
        }
    }

    /**
     * Serializes an array into a newly allocated in-memory segment (used by .npz archiving).
     */
    public static MemorySegment writeToSegment(NDArray array, Arena arena) {
        NDArray contiguous = array.isContiguous() ? array : array.contiguous();
        long payloadBytes = contiguous.getSize() * contiguous.getDType().layout.byteSize();

        NpyHeaderWriter.HeaderInfo info = NpyHeaderWriter.prepareHeader(
                contiguous.getDType(), contiguous.getShape(), false);
        long totalBytes = info.totalHeaderBytes() + payloadBytes;

        MemorySegment segment = arena.allocate(totalBytes, 64);
        NpyHeaderWriter.writeHeaderToSegment(segment, info);
        MemorySegment.copy(contiguous.getData(), 0, segment, info.totalHeaderBytes(), payloadBytes);

        return segment;
    }
}
