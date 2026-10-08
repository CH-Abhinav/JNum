package jnum.io.internal.npy;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import jnum.NDArray;

/**
 * Handles multi-array .npz ZIP archives by wrapping zip entry bytes in MemorySegments.
 */
public final class NpzArchive {

    private NpzArchive() {
        throw new AssertionError("NpzArchive cannot be instantiated.");
    }

    public static Map<String, NDArray> read(Path path, Arena arena) throws IOException {
        Map<String, NDArray> result = new HashMap<>();
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(path))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String name = entry.getName();
                if (name.endsWith(".npy")) {
                    name = name.substring(0, name.length() - 4);
                }

                byte[] entryBytes = zis.readAllBytes();
                // Wrap directly in a MemorySegment for parsing
                MemorySegment entrySegment = MemorySegment.ofArray(entryBytes);
                NDArray array = NpyReader.readFromSegment(entrySegment, arena);
                result.put(name, array);
                zis.closeEntry();
            }
        }
        return result;
    }

    public static void write(Map<String, NDArray> arrays, Path path) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(path))) {
            try (Arena tempArena = Arena.ofConfined()) {
                for (Map.Entry<String, NDArray> e : arrays.entrySet()) {
                    String entryName = e.getKey().endsWith(".npy") ? e.getKey() : e.getKey() + ".npy";
                    ZipEntry entry = new ZipEntry(entryName);
                    zos.putNextEntry(entry);

                    // Serialize to segment, then write to zip entry
                    MemorySegment segment = NpyWriter.writeToSegment(e.getValue(), tempArena);
                    byte[] bytes = segment.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                    zos.write(bytes);
                    zos.closeEntry();
                }
            }
        }
    }
}