package jnum.io.internal.csv;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import jnum.DType;
import jnum.NDArray;
import jnum.internal.layout.ShapeUtil;
import jnum.io.CsvOptions;
import jnum.io.internal.common.FastNumberParser;

/**
 * High-performance, zero-GC tabular text reader parsing directly into off-heap memory.
 */
public final class CsvReader {

    private CsvReader() {
        throw new AssertionError("CsvReader cannot be instantiated.");
    }

    public static NDArray read(Path path, CsvOptions options, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize == 0) {
                throw new IllegalArgumentException("Cannot parse empty CSV file: " + path);
            }

            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, tempArena);

                byte delim = (byte) options.delimiter();
                byte comment = (byte) options.commentPrefix();
                int skipRows = options.skipRows();
                boolean hasHeader = options.hasHeader();

                // =============================================================
                // Pass 1: Discover rows and columns count
                // =============================================================
                long pos = 0;
                int totalLines = 0;
                boolean headerSkipped = false;
                int numCols = -1;
                int numRows = 0;

                while (pos < fileSize) {
                    long lineStart = pos;
                    while (pos < fileSize && fileSegment.get(ValueLayout.JAVA_BYTE, pos) != '\n') {
                        pos++;
                    }
                    long lineEnd = pos;
                    if (pos < fileSize && fileSegment.get(ValueLayout.JAVA_BYTE, pos) == '\n') {
                        pos++; // skip '\n'
                    }

                    // Trim trailing '\r'
                    if (lineEnd > lineStart && fileSegment.get(ValueLayout.JAVA_BYTE, lineEnd - 1) == '\r') {
                        lineEnd--;
                    }

                    totalLines++;
                    if (totalLines <= skipRows) continue;

                    long firstNonWs = findFirstNonWhitespace(fileSegment, lineStart, lineEnd);
                    if (firstNonWs == lineEnd) continue; // blank line

                    if (comment != 0 && fileSegment.get(ValueLayout.JAVA_BYTE, firstNonWs) == comment) {
                        continue; // comment line
                    }

                    if (hasHeader && !headerSkipped) {
                        headerSkipped = true;
                        continue; // skip column names header
                    }

                    if (numCols == -1) {
                        numCols = countColumns(fileSegment, lineStart, lineEnd, delim);
                    }
                    numRows++;
                }

                if (numRows == 0 || numCols <= 0) {
                    throw new IllegalArgumentException("No valid tabular data found in file: " + path);
                }

                // =============================================================
                // Pass 2: Allocate target segment & parse directly
                // =============================================================
                long[] shape = new long[]{numRows, numCols};
                DType dtype = options.dtype();
                long totalElements = (long) numRows * numCols;
                MemorySegment dataSegment = arena.allocate(totalElements * dtype.layout.byteSize(), 64);

                pos = 0;
                totalLines = 0;
                headerSkipped = false;
                int currentRow = 0;

                while (pos < fileSize && currentRow < numRows) {
                    long lineStart = pos;
                    while (pos < fileSize && fileSegment.get(ValueLayout.JAVA_BYTE, pos) != '\n') {
                        pos++;
                    }
                    long lineEnd = pos;
                    if (pos < fileSize && fileSegment.get(ValueLayout.JAVA_BYTE, pos) == '\n') {
                        pos++;
                    }
                    if (lineEnd > lineStart && fileSegment.get(ValueLayout.JAVA_BYTE, lineEnd - 1) == '\r') {
                        lineEnd--;
                    }

                    totalLines++;
                    if (totalLines <= skipRows) continue;

                    long firstNonWs = findFirstNonWhitespace(fileSegment, lineStart, lineEnd);
                    if (firstNonWs == lineEnd) continue;
                    if (comment != 0 && fileSegment.get(ValueLayout.JAVA_BYTE, firstNonWs) == comment) continue;
                    if (hasHeader && !headerSkipped) {
                        headerSkipped = true;
                        continue;
                    }

                    // Parse line tokens
                    long tokStart = lineStart;
                    int currentCol = 0;

                    for (long i = lineStart; i <= lineEnd; i++) {
                        boolean isEnd = (i == lineEnd);
                        boolean isDelim = !isEnd && (fileSegment.get(ValueLayout.JAVA_BYTE, i) == delim);

                        if (isDelim || isEnd) {
                            if (currentCol < numCols) {
                                double val = FastNumberParser.parseDouble(fileSegment, tokStart, i, options.naString(), Double.NaN);
                                long flatIndex = (long) currentRow * numCols + currentCol;
                                setElement(dataSegment, flatIndex, val, dtype);
                                currentCol++;
                            }
                            tokStart = i + 1;
                        }
                    }

                    // Pad missing trailing columns with NaN if line was shorter
                    while (currentCol < numCols) {
                        long flatIndex = (long) currentRow * numCols + currentCol;
                        setElement(dataSegment, flatIndex, Double.NaN, dtype);
                        currentCol++;
                    }

                    currentRow++;
                }

                return NDArray.ofRaw(dataSegment, shape, ShapeUtil.calculateDefaultStrides(shape), dtype);
            }
        }
    }

    private static void setElement(MemorySegment seg, long index, double val, DType dtype) {
        switch (dtype) {
            case f64 -> seg.setAtIndex(ValueLayout.JAVA_DOUBLE, index, val);
            case f32 -> seg.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) val);
            case i32 -> seg.setAtIndex(ValueLayout.JAVA_INT, index, (int) val);
            case bool -> seg.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (val != 0.0 && !Double.isNaN(val) ? 1 : 0));
        }
    }

    private static long findFirstNonWhitespace(MemorySegment seg, long start, long end) {
        for (long i = start; i < end; i++) {
            byte b = seg.get(ValueLayout.JAVA_BYTE, i);
            if (b != ' ' && b != '\t' && b != '\r') return i;
        }
        return end;
    }

    private static int countColumns(MemorySegment seg, long start, long end, byte delim) {
        int count = 1;
        for (long i = start; i < end; i++) {
            if (seg.get(ValueLayout.JAVA_BYTE, i) == delim) {
                count++;
            }
        }
        return count;
    }
}
