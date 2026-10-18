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
package jnum;

import java.lang.foreign.ValueLayout;

/**
 * Supported data types for tensors in JNum.
 *
 * <p>Each data type encapsulates a Foreign Function &amp; Memory (FFM) {@link ValueLayout},
 * defining the memory footprint, binary layout, and alignment constraints when reading
 * from or writing to native {@link java.lang.foreign.MemorySegment} buffers.</p>
 */

public enum DType{
    /** 32-bit signed two's complement integer (4 bytes, {@link ValueLayout#JAVA_INT}).*/
    i32(ValueLayout.JAVA_INT),
    /** 32-bit IEEE 754 single-precision floating point (4 bytes, {@link ValueLayout#JAVA_FLOAT}). */
    f32(ValueLayout.JAVA_FLOAT),
    /** 64-bit IEEE 754 double-precision floating point (8 bytes, {@link ValueLayout#JAVA_DOUBLE}). */
    f64(ValueLayout.JAVA_DOUBLE),
    /** Boolean truth value stored as an 8-bit byte (1 byte, {@link ValueLayout#JAVA_BYTE}, where 0 is false and 1 is true). */
    bool(ValueLayout.JAVA_BYTE);

    /** The FFM value layout descriptor used for off-heap/native memory access. */
    public final ValueLayout layout;

    /**
     * Constructs a {@code DType} with the specified memory value layout.
     *
     * @param layout the {@link ValueLayout} corresponding to this data type
     */
    private DType(ValueLayout layout) {
        this.layout = layout;
    }
    
}
