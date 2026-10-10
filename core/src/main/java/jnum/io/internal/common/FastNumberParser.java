package jnum.io.internal.common;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * High-performance, zero-allocation ASCII byte-to-number parser.
 * Parses doubles, floats, and ints directly from MemorySegments without creating String objects.
 */
public final class FastNumberParser {

    private FastNumberParser() {
        throw new AssertionError("FastNumberParser cannot be instantiated.");
    }

    public static double parseDouble(MemorySegment segment, long start, long end, String naString, double naValue) {
        // 1. Trim leading and trailing whitespace
        while (start < end && isWhitespace(segment.get(ValueLayout.JAVA_BYTE, start))) {
            start++;
        }
        while (end > start && isWhitespace(segment.get(ValueLayout.JAVA_BYTE, end - 1))) {
            end--;
        }

        long len = end - start;
        if (len == 0) return naValue;

        // 2. Check for NA / missing sentinel string
        if (naString != null && matchesString(segment, start, end, naString)) {
            return naValue;
        }

        // 3. Handle sign
        boolean negative = false;
        byte first = segment.get(ValueLayout.JAVA_BYTE, start);
        if (first == '-') {
            negative = true;
            start++;
            len--;
        } else if (first == '+') {
            start++;
            len--;
        }

        if (len == 0) return naValue;

        // 4. Check for special constants (NaN, Inf, Infinity)
        if (matchesIgnoreCase(segment, start, end, "nan")) {
            return Double.NaN;
        }
        if (matchesIgnoreCase(segment, start, end, "inf") || matchesIgnoreCase(segment, start, end, "infinity")) {
            return negative ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        }

        // 5. Fast integer and fractional parsing
        try {
            long intPart = 0;
            boolean hasDigits = false;
            long i = start;

            while (i < end) {
                byte b = segment.get(ValueLayout.JAVA_BYTE, i);
                if (b >= '0' && b <= '9') {
                    intPart = intPart * 10 + (b - '0');
                    hasDigits = true;
                    i++;
                } else {
                    break;
                }
            }

            double val = intPart;

            // Fractional part
            if (i < end && segment.get(ValueLayout.JAVA_BYTE, i) == '.') {
                i++;
                double factor = 0.1;
                while (i < end) {
                    byte b = segment.get(ValueLayout.JAVA_BYTE, i);
                    if (b >= '0' && b <= '9') {
                        val += (b - '0') * factor;
                        factor *= 0.1;
                        hasDigits = true;
                        i++;
                    } else {
                        break;
                    }
                }
            }

            // Exponent part (e or E)
            if (i < end && (segment.get(ValueLayout.JAVA_BYTE, i) == 'e' || segment.get(ValueLayout.JAVA_BYTE, i) == 'E')) {
                i++;
                boolean expNeg = false;
                if (i < end && segment.get(ValueLayout.JAVA_BYTE, i) == '-') {
                    expNeg = true;
                    i++;
                } else if (i < end && segment.get(ValueLayout.JAVA_BYTE, i) == '+') {
                    i++;
                }
                int exp = 0;
                while (i < end) {
                    byte b = segment.get(ValueLayout.JAVA_BYTE, i);
                    if (b >= '0' && b <= '9') {
                        exp = exp * 10 + (b - '0');
                        i++;
                    } else {
                        break;
                    }
                }
                val *= Math.pow(10, expNeg ? -exp : exp);
            }

            if (!hasDigits) {
                return fallbackParse(segment, start - (negative ? 1 : 0), end, naValue);
            }

            return negative ? -val : val;

        } catch (Exception e) {
            return fallbackParse(segment, start - (negative ? 1 : 0), end, naValue);
        }
    }

    private static double fallbackParse(MemorySegment segment, long start, long end, double naValue) {
        try {
            byte[] bytes = segment.asSlice(start, end - start).toArray(ValueLayout.JAVA_BYTE);
            String str = new String(bytes, StandardCharsets.US_ASCII).trim();
            if (str.isEmpty()) return naValue;
            return Double.parseDouble(str);
        } catch (Exception ignored) {
            return naValue;
        }
    }

    private static boolean isWhitespace(byte b) {
        return b == ' ' || b == '\t' || b == '\r' || b == '\n';
    }

    private static boolean matchesString(MemorySegment segment, long start, long end, String target) {
        if (end - start != target.length()) return false;
        for (int i = 0; i < target.length(); i++) {
            if (segment.get(ValueLayout.JAVA_BYTE, start + i) != (byte) target.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static boolean matchesIgnoreCase(MemorySegment segment, long start, long end, String target) {
        if (end - start != target.length()) return false;
        for (int i = 0; i < target.length(); i++) {
            byte b = segment.get(ValueLayout.JAVA_BYTE, start + i);
            char c = target.charAt(i);
            if (Character.toLowerCase((char) b) != Character.toLowerCase(c)) {
                return false;
            }
        }
        return true;
    }
}
