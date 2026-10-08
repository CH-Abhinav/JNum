package jnum.io.internal.npy;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import jnum.DType;

/**
 * Builds and writes NumPy headers directly into MemorySegments with 64-byte alignment.
 */
public final class NpyHeaderWriter {

    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private NpyHeaderWriter() {
        throw new AssertionError("NpyHeaderWriter cannot be instantiated.");
    }

    public static record HeaderInfo(byte[] dictBytes, int headerLen, long totalHeaderBytes) {}

    public static HeaderInfo prepareHeader(DType dtype, long[] shape, boolean fortranOrder) {
        String descr = switch (dtype) {
            case f32 -> "<f4";
            case f64 -> "<f8";
            case i32 -> "<i4";
            case bool -> "|b1";
        };

        StringBuilder sbShape = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++) {
            sbShape.append(shape[i]);
            if (shape.length == 1 || i < shape.length - 1) {
                sbShape.append(", ");
            }
        }
        sbShape.append(")");

        String dictStr = String.format("{'descr': '%s', 'fortran_order': %s, 'shape': %s, }",
                descr, fortranOrder ? "True" : "False", sbShape);

        // v1.0 prefix: 10 bytes. (10 + headerLen) % 64 must be 0
        int unpaddedLen = 10 + dictStr.length() + 1; // +1 for '\n'
        int pad = (64 - (unpaddedLen % 64)) % 64;

        StringBuilder padded = new StringBuilder(dictStr);
        for (int i = 0; i < pad; i++) padded.append(' ');
        padded.append('\n');

        byte[] dictBytes = padded.toString().getBytes(StandardCharsets.US_ASCII);
        int headerLen = dictBytes.length;
        long totalHeaderBytes = 10L + headerLen;

        return new HeaderInfo(dictBytes, headerLen, totalHeaderBytes);
    }

    public static void writeHeaderToSegment(MemorySegment target, HeaderInfo info) {
        // Write 6 magic bytes
        for (int i = 0; i < 6; i++) {
            target.set(ValueLayout.JAVA_BYTE, i, NpyHeaderParser.MAGIC[i]);
        }
        target.set(ValueLayout.JAVA_BYTE, 6, (byte) 1); // Major v1
        target.set(ValueLayout.JAVA_BYTE, 7, (byte) 0); // Minor v0
        target.set(LE_SHORT, 8, (short) info.headerLen());
        MemorySegment.copy(MemorySegment.ofArray(info.dictBytes()), 0, target, 10L, info.headerLen());
    }
}