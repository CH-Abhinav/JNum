package jnum.io.internal.safetensors;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import jnum.NDArray;
import jnum.internal.layout.ShapeUtil;

/**
 * Pure FFM reader for Hugging Face .safetensors files supporting instant zero-copy mmap.
 */
public final class SafetensorsReader {

    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private SafetensorsReader() {
        throw new AssertionError("SafetensorsReader cannot be instantiated.");
    }

    public static SafetensorsMetadata readMetadata(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            try (Arena tempArena = Arena.ofConfined()) {
                long fileSize = channel.size();
                if (fileSize < 8) {
                    throw new IllegalArgumentException("Invalid Safetensors file: size is less than 8 bytes.");
                }
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, tempArena);
                long headerLen = fileSegment.get(LE_LONG, 0);

                if (headerLen <= 0 || (8L + headerLen) > fileSize) {
                    throw new IllegalArgumentException("Corrupt Safetensors header length: " + headerLen);
                }

                byte[] jsonBytes = fileSegment.asSlice(8L, headerLen).toArray(ValueLayout.JAVA_BYTE);
                String jsonStr = new String(jsonBytes, StandardCharsets.UTF_8);
                return SafetensorsJsonParser.parse(jsonStr, headerLen);
            }
        }
    }

    /**
     * Reads all tensors into independent off-heap MemorySegments allocated in the provided Arena.
     */
    public static Map<String, NDArray> read(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, tempArena);
                long headerLen = fileSegment.get(LE_LONG, 0);
                byte[] jsonBytes = fileSegment.asSlice(8L, headerLen).toArray(ValueLayout.JAVA_BYTE);
                SafetensorsMetadata metadata = SafetensorsJsonParser.parse(new String(jsonBytes, StandardCharsets.UTF_8), headerLen);

                long dataBufferOffset = 8L + headerLen;
                Map<String, NDArray> result = new HashMap<>();

                for (SafetensorsTensorDesc desc : metadata.tensors().values()) {
                    long tensorBytes = desc.byteSize();
                    MemorySegment dataSegment = arena.allocate(tensorBytes, 64);
                    MemorySegment.copy(fileSegment, dataBufferOffset + desc.startOffset(), dataSegment, 0, tensorBytes);

                    long[] strides = ShapeUtil.calculateDefaultStrides(desc.shape());
                    result.put(desc.name(), NDArray.ofRaw(dataSegment, desc.shape(), strides, desc.dtype()));
                }
                return result;
            }
        }
    }

    /**
     * Instant zero-copy memory-mapped load: slices the mapped file segment for all tensors.
     */
    public static Map<String, NDArray> readMmap(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);

            long headerLen = fileSegment.get(LE_LONG, 0);
            byte[] jsonBytes = fileSegment.asSlice(8L, headerLen).toArray(ValueLayout.JAVA_BYTE);
            SafetensorsMetadata metadata = SafetensorsJsonParser.parse(new String(jsonBytes, StandardCharsets.UTF_8), headerLen);

            long dataBufferOffset = 8L + headerLen;
            Map<String, NDArray> result = new HashMap<>();

            for (SafetensorsTensorDesc desc : metadata.tensors().values()) {
                MemorySegment tensorSlice = fileSegment.asSlice(dataBufferOffset + desc.startOffset(), desc.byteSize());
                long[] strides = ShapeUtil.calculateDefaultStrides(desc.shape());
                result.put(desc.name(), NDArray.ofRaw(tensorSlice, desc.shape(), strides, desc.dtype()));
            }
            return result;
        }
    }

    /**
     * Reads a single specific tensor by name from the file.
     */
    public static NDArray readSingle(Path path, String tensorName, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, tempArena);
                long headerLen = fileSegment.get(LE_LONG, 0);
                byte[] jsonBytes = fileSegment.asSlice(8L, headerLen).toArray(ValueLayout.JAVA_BYTE);
                SafetensorsMetadata metadata = SafetensorsJsonParser.parse(new String(jsonBytes, StandardCharsets.UTF_8), headerLen);

                SafetensorsTensorDesc desc = metadata.tensors().get(tensorName);
                if (desc == null) {
                    throw new IllegalArgumentException("Tensor '" + tensorName + "' not found in " + path);
                }

                long dataBufferOffset = 8L + headerLen;
                long tensorBytes = desc.byteSize();
                MemorySegment dataSegment = arena.allocate(tensorBytes, 64);
                MemorySegment.copy(fileSegment, dataBufferOffset + desc.startOffset(), dataSegment, 0, tensorBytes);

                long[] strides = ShapeUtil.calculateDefaultStrides(desc.shape());
                return NDArray.ofRaw(dataSegment, desc.shape(), strides, desc.dtype());
            }
        }
    }
}
