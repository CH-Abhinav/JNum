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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import jnum.NDArray;

/**
 * Pure FFM writer serializing Map<String, NDArray> to Hugging Face .safetensors files.
 */
public final class SafetensorsWriter {

    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private SafetensorsWriter() {
        throw new AssertionError("SafetensorsWriter cannot be instantiated.");
    }

    public static void write(Map<String, NDArray> tensors, Map<String, String> userMetadata, Path path) throws IOException {
        // 1. Prepare tensor contiguous layouts and compute cumulative byte offsets
        List<String> names = new ArrayList<>(tensors.keySet());
        List<NDArray> contiguousTensors = new ArrayList<>(names.size());
        List<Long> byteSizes = new ArrayList<>(names.size());
        List<Long> startOffsets = new ArrayList<>(names.size());

        long currentOffset = 0;
        for (String name : names) {
            NDArray arr = tensors.get(name);
            NDArray cont = arr.isContiguous() ? arr : arr.contiguous();
            contiguousTensors.add(cont);
            long bytes = cont.getSize() * cont.getDType().layout.byteSize();
            byteSizes.add(bytes);
            startOffsets.add(currentOffset);
            currentOffset += bytes;
        }
        long totalPayloadBytes = currentOffset;

        // 2. Build JSON header
        StringBuilder json = new StringBuilder("{");
        boolean hasMetadata = (userMetadata != null && !userMetadata.isEmpty());
        if (hasMetadata) {
            json.append("\"__metadata__\":{");
            int mCount = 0;
            for (Map.Entry<String, String> me : userMetadata.entrySet()) {
                if (mCount++ > 0) json.append(",");
                json.append("\"").append(escape(me.getKey())).append("\":\"").append(escape(me.getValue())).append("\"");
            }
            json.append("}");
        }

        for (int i = 0; i < names.size(); i++) {
            if (i > 0 || hasMetadata) {
                json.append(",");
            }
            String name = names.get(i);
            NDArray arr = contiguousTensors.get(i);
            long start = startOffsets.get(i);
            long end = start + byteSizes.get(i);

            String dtypeStr = switch (arr.getDType()) {
                case f32 -> "F32";
                case f64 -> "F64";
                case i32 -> "I32";
                case bool -> "BOOL";
            };

            json.append("\"").append(escape(name)).append("\":{")
                    .append("\"dtype\":\"").append(dtypeStr).append("\",")
                    .append("\"shape\":").append(Arrays.toString(arr.getShape())).append(",")
                    .append("\"data_offsets\":[").append(start).append(",").append(end).append("]}");
        }
        json.append("}");

        // 3. Align header to 64 bytes (so binary payload starts on an aligned boundary)
        int rawLen = json.toString().getBytes(StandardCharsets.UTF_8).length;
        // Total prefix: 8 bytes. (8 + N) % 64 must be 0
        int pad = (64 - ((8 + rawLen) % 64)) % 64;
        for (int i = 0; i < pad; i++) json.append(' ');

        byte[] jsonBytes = json.toString().getBytes(StandardCharsets.UTF_8);
        long headerLen = jsonBytes.length;
        long totalFileSize = 8L + headerLen + totalPayloadBytes;

        // 4. Memory-map destination file and write directly
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.READ,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalFileSize, arena);

                // Write 8-byte header length
                fileSegment.set(LE_LONG, 0, headerLen);

                // Copy JSON header bytes
                MemorySegment.copy(MemorySegment.ofArray(jsonBytes), 0, fileSegment, 8L, headerLen);

                // Copy all tensor payloads
                long dataBufferOffset = 8L + headerLen;
                for (int i = 0; i < names.size(); i++) {
                    NDArray cont = contiguousTensors.get(i);
                    long offset = dataBufferOffset + startOffsets.get(i);
                    long len = byteSizes.get(i);
                    MemorySegment.copy(cont.getData(), 0, fileSegment, offset, len);
                }
            }
        }
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
