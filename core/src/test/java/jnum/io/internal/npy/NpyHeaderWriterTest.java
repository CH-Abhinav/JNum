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

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import jnum.DType;
import org.junit.jupiter.api.Test;

class NpyHeaderWriterTest {

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<NpyHeaderWriter> constructor = NpyHeaderWriter.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testHeaderAlignment64Bytes() {
        for (DType dtype : DType.values()) {
            NpyHeaderWriter.HeaderInfo info1 = NpyHeaderWriter.prepareHeader(dtype, new long[]{10}, false);
            assertEquals(0, info1.totalHeaderBytes() % 64, "Total header bytes must be aligned to 64 bytes");

            NpyHeaderWriter.HeaderInfo info2 = NpyHeaderWriter.prepareHeader(dtype, new long[]{100, 200, 300}, true);
            assertEquals(0, info2.totalHeaderBytes() % 64, "Total header bytes must be aligned to 64 bytes");
        }
    }

    @Test
    void testWriteHeaderToSegmentAndParseBack() {
        NpyHeaderWriter.HeaderInfo info = NpyHeaderWriter.prepareHeader(DType.f32, new long[]{3, 5}, false);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(info.totalHeaderBytes());
            NpyHeaderWriter.writeHeaderToSegment(target, info);

            NpyHeader parsed = NpyHeaderParser.parse(target);
            assertEquals(1, parsed.majorVersion());
            assertEquals(0, parsed.minorVersion());
            assertEquals(DType.f32, parsed.dtype());
            assertFalse(parsed.fortranOrder());
            assertArrayEquals(new long[]{3, 5}, parsed.shape());
            assertEquals(info.totalHeaderBytes(), parsed.payloadOffset());
        }
    }
}
