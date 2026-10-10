package jnum.io.internal.safetensors;

import java.util.Map;

/**
 * Container holding the parsed metadata header of a .safetensors file.
 */
public record SafetensorsMetadata(
        Map<String, SafetensorsTensorDesc> tensors,
        Map<String, String> userMetadata,
        long headerLength
) {}
