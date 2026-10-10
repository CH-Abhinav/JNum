package jnum.io.internal.safetensors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jnum.DType;

/**
 * Lightweight, zero-dependency JSON parser specialized for .safetensors headers.
 */
public final class SafetensorsJsonParser {

    private SafetensorsJsonParser() {
        throw new AssertionError("SafetensorsJsonParser cannot be instantiated.");
    }

    public static SafetensorsMetadata parse(String json, long headerLength) {
        String trimmed = json.trim();
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            throw new IllegalArgumentException("Invalid Safetensors header: JSON root must be an object.");
        }

        Map<String, SafetensorsTensorDesc> tensors = new HashMap<>();
        Map<String, String> userMetadata = new HashMap<>();

        int pos = 1; // Skip opening '{'
        int len = trimmed.length();

        while (pos < len) {
            pos = skipWhitespace(trimmed, pos);
            if (pos >= len || trimmed.charAt(pos) == '}') break;

            // 1. Parse top-level key (tensor name or __metadata__)
            if (trimmed.charAt(pos) != '"') {
                pos++;
                continue;
            }
            int keyEnd = findClosingQuote(trimmed, pos + 1);
            String key = trimmed.substring(pos + 1, keyEnd);
            pos = skipWhitespace(trimmed, keyEnd + 1);

            // Expect ':'
            if (pos < len && trimmed.charAt(pos) == ':') pos++;
            pos = skipWhitespace(trimmed, pos);

            // 2. Parse object value
            if (pos < len && trimmed.charAt(pos) == '{') {
                int objEnd = findMatchingBrace(trimmed, pos);
                String objContent = trimmed.substring(pos, objEnd + 1);

                if ("__metadata__".equals(key)) {
                    parseUserMetadata(objContent, userMetadata);
                } else {
                    SafetensorsTensorDesc desc = parseTensorDescriptor(key, objContent);
                    tensors.put(key, desc);
                }
                pos = objEnd + 1;
            }

            // Skip trailing comma
            pos = skipWhitespace(trimmed, pos);
            if (pos < len && trimmed.charAt(pos) == ',') pos++;
        }

        return new SafetensorsMetadata(Collections.unmodifiableMap(tensors),
                Collections.unmodifiableMap(userMetadata), headerLength);
    }

    private static SafetensorsTensorDesc parseTensorDescriptor(String name, String obj) {
        String dtypeStr = extractStringField(obj, "dtype");
        long[] shape = extractLongArrayField(obj, "shape");
        long[] offsets = extractLongArrayField(obj, "data_offsets");

        if (offsets.length != 2) {
            throw new IllegalArgumentException("Tensor '" + name + "' must have exactly 2 data_offsets [start, end].");
        }

        DType dtype = switch (dtypeStr.toUpperCase()) {
            case "F32" -> DType.f32;
            case "F64" -> DType.f64;
            case "I32" -> DType.i32;
            case "BOOL" -> DType.bool;
            default -> throw new UnsupportedOperationException("Unsupported Safetensors dtype: '" + dtypeStr +
                    "' for tensor '" + name + "'. Supported: F32, F64, I32, BOOL");
        };

        return new SafetensorsTensorDesc(name, dtype, shape, offsets[0], offsets[1]);
    }

    private static void parseUserMetadata(String obj, Map<String, String> out) {
        int pos = 1;
        int len = obj.length();
        while (pos < len) {
            pos = skipWhitespace(obj, pos);
            if (pos >= len || obj.charAt(pos) == '}') break;

            if (obj.charAt(pos) == '"') {
                int keyEnd = findClosingQuote(obj, pos + 1);
                String k = obj.substring(pos + 1, keyEnd);
                pos = skipWhitespace(obj, keyEnd + 1);

                if (pos < len && obj.charAt(pos) == ':') pos++;
                pos = skipWhitespace(obj, pos);

                if (pos < len && obj.charAt(pos) == '"') {
                    int valEnd = findClosingQuote(obj, pos + 1);
                    String v = obj.substring(pos + 1, valEnd);
                    out.put(k, v);
                    pos = valEnd + 1;
                }
            }
            pos = skipWhitespace(obj, pos);
            if (pos < len && obj.charAt(pos) == ',') pos++;
        }
    }

    private static String extractStringField(String obj, String fieldName) {
        int idx = obj.indexOf("\"" + fieldName + "\"");
        if (idx == -1) throw new IllegalArgumentException("Missing field '" + fieldName + "' in " + obj);

        int colon = obj.indexOf(':', idx);
        int quoteStart = obj.indexOf('"', colon);
        int quoteEnd = findClosingQuote(obj, quoteStart + 1);
        return obj.substring(quoteStart + 1, quoteEnd);
    }

    private static long[] extractLongArrayField(String obj, String fieldName) {
        int idx = obj.indexOf("\"" + fieldName + "\"");
        if (idx == -1) throw new IllegalArgumentException("Missing field '" + fieldName + "' in " + obj);

        int open = obj.indexOf('[', idx);
        int close = obj.indexOf(']', open);
        String inside = obj.substring(open + 1, close).trim();
        if (inside.isEmpty()) return new long[0];

        String[] parts = inside.split(",");
        List<Long> nums = new ArrayList<>();
        for (String p : parts) {
            String trimmed = p.trim();
            if (!trimmed.isEmpty()) nums.add(Long.parseLong(trimmed));
        }
        long[] result = new long[nums.size()];
        for (int i = 0; i < nums.size(); i++) result[i] = nums.get(i);
        return result;
    }

    private static int skipWhitespace(String s, int pos) {
        while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++;
        return pos;
    }

    private static int findClosingQuote(String s, int start) {
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) == '"' && s.charAt(i - 1) != '\\') return i;
        }
        throw new IllegalArgumentException("Unterminated string in Safetensors JSON: " + s);
    }

    private static int findMatchingBrace(String s, int openPos) {
        int depth = 0;
        boolean inQuotes = false;
        for (int i = openPos; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQuotes = !inQuotes;
            if (!inQuotes) {
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) return i;
                }
            }
        }
        throw new IllegalArgumentException("Unterminated object in Safetensors JSON: " + s);
    }
}
