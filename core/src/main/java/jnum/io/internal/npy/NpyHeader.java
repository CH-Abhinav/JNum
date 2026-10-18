/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.io.internal.npy;

import java.nio.ByteOrder;
import jnum.DType;

/**
 * Strongly-typed representation of NumPy .npy header metadata.
 */
public record NpyHeader(
        int majorVersion,
        int minorVersion,
        int headerLength,
        DType dtype,
        ByteOrder byteOrder,
        boolean fortranOrder,
        long[] shape,
        long payloadOffset
) {
    public long totalElements() {
        if (shape.length == 0) return 1L; // 0-D scalar
        long total = 1L;
        for (long dim : shape) total *= dim;
        return total;
    }

    public long payloadByteSize() {
        return totalElements() * dtype.layout.byteSize();
    }
}