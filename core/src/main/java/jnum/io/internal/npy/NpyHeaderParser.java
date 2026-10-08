package jnum.io.internal.npy;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import jnum.DType;

/**
 * Pure FFM parser reading NumPy headers directly from MemorySegments.
 */
public final class NpyHeaderParser {

    public static final byte[] MAGIC = new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private NpyHeaderParser() {
        throw new AssertionError("NpyHeaderParser cannot be instantiated.");
    }

    public static NpyHeader parse(MemorySegment segment) {
        if (segment.byteSize() < 10) {
            throw new IllegalArgumentException("Segment is too small to contain a valid .npy header.");
        }

        // Verify magic bytes
        for (int i = 0; i < 6; i++) {
            if (segment.get(ValueLayout.JAVA_BYTE, i) != MAGIC[i]) {
                throw new IllegalArgumentException("Invalid .npy file: magic header prefix mismatch.");
            }
        }

        int major = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, 6));
        int minor = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, 7));

        int headerLen;
        long payloadOffset;

        if (major == 1) {
            headerLen = Short.toUnsignedInt(segment.get(LE_SHORT, 8));
            payloadOffset = 10L + headerLen;
        } else if (major == 2 || major == 3) {
            if (segment.byteSize() < 12) {
                throw new IllegalArgumentException("Segment is too small to contain a .npy v2.0 header.");
            }
            headerLen = segment.get(LE_INT, 8);
            payloadOffset = 12L + headerLen;
        } else {
            throw new UnsupportedOperationException("Unsupported .npy version: " + major + "." + minor);
        }

        long prefixLen = (major == 1) ? 10L : 12L;
        byte[] dictBytes = segment.asSlice(prefixLen, headerLen).toArray(ValueLayout.JAVA_BYTE);
        String dictStr = new String(dictBytes, major == 3 ? StandardCharsets.UTF_8 : StandardCharsets.US_ASCII);

        return parseDict(dictStr, major, minor, headerLen, payloadOffset);
    }

    private static NpyHeader parseDict(String dict, int major, int minor, int headerLen, long payloadOffset) {
        String descr = extractStringValue(dict, "descr");
        boolean fortranOrder = extractBooleanValue(dict, "fortran_order");
        long[] shape = extractShape(dict);

        DType dtype;
        ByteOrder order = ByteOrder.LITTLE_ENDIAN;

        if (descr.startsWith(">")) {
            order = ByteOrder.BIG_ENDIAN;
        }

        String typeCode = descr.replace("<", "").replace(">", "").replace("|", "").replace("=", "").trim();
        dtype = switch (typeCode) {
            case "f4" -> DType.f32;
            case "f8" -> DType.f64;
            case "i4" -> DType.i32;
            case "b1", "u1", "?" -> DType.bool;
            default -> throw new UnsupportedOperationException("Unsupported dtype in .npy descr: '" + descr + "'");
        };

        return new NpyHeader(major, minor, headerLen, dtype, order, fortranOrder, shape, payloadOffset);
    }

    private static String extractStringValue(String dict, String key) {
        int idx = dict.indexOf("'" + key + "'");
        if (idx == -1) idx = dict.indexOf("\"" + key + "\"");
        if (idx == -1) throw new IllegalArgumentException("Missing '" + key + "' in .npy header: " + dict);

        int colon = dict.indexOf(':', idx);
        int quoteStart = -1;
        char quoteChar = 0;
        for (int i = colon + 1; i < dict.length(); i++) {
            char c = dict.charAt(i);
            if (c == '\'' || c == '"') {
                quoteStart = i;
                quoteChar = c;
                break;
            }
        }
        int quoteEnd = dict.indexOf(quoteChar, quoteStart + 1);
        return dict.substring(quoteStart + 1, quoteEnd);
    }

    private static boolean extractBooleanValue(String dict, String key) {
        int idx = dict.indexOf("'" + key + "'");
        if (idx == -1) idx = dict.indexOf("\"" + key + "\"");
        if (idx == -1) return false;

        int colon = dict.indexOf(':', idx);
        String sub = dict.substring(colon + 1).trim();
        return sub.startsWith("True") || sub.startsWith("true");
    }

    private static long[] extractShape(String dict) {
        int idx = dict.indexOf("'shape'");
        if (idx == -1) idx = dict.indexOf("\"shape\"");
        if (idx == -1) throw new IllegalArgumentException("Missing 'shape' in .npy header: " + dict);

        int openParen = dict.indexOf('(', idx);
        int closeParen = dict.indexOf(')', openParen);
        String inside = dict.substring(openParen + 1, closeParen).trim();
        if (inside.isEmpty()) {
            return new long[0]; // 0-D scalar
        }

        String[] parts = inside.split(",");
        List<Long> dims = new ArrayList<>();
        for (String p : parts) {
            String trimmed = p.trim();
            if (!trimmed.isEmpty()) {
                dims.add(Long.parseLong(trimmed));
            }
        }
        long[] result = new long[dims.size()];
        for (int i = 0; i < dims.size(); i++) result[i] = dims.get(i);
        return result;
    }
}