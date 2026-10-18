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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NpyReaderTest {

    @TempDir
    Path tempDir;

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<NpyReader> constructor = NpyReader.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testReadFromFileAndFromSegment() throws IOException {
        Path file = tempDir.resolve("test.npy");
        NDArray original = JNum.from(new float[]{1.1f, 2.2f, 3.3f, 4.4f}, 2, 2);

        NpyWriter.write(original, file);

        try (Arena arena = Arena.ofConfined()) {
            NDArray readFromFile = NpyReader.read(file, arena);
            assertArrayEquals(new long[]{2, 2}, readFromFile.getShape());
            assertEquals(1.1f, readFromFile.getFloat(0, 0), 1e-6f);
            assertEquals(4.4f, readFromFile.getFloat(1, 1), 1e-6f);

            // In-memory segment read
            MemorySegment memSeg = NpyWriter.writeToSegment(original, arena);
            NDArray readFromSeg = NpyReader.readFromSegment(memSeg, arena);
            assertEquals(2.2f, readFromSeg.getFloat(0, 1), 1e-6f);
            assertEquals(3.3f, readFromSeg.getFloat(1, 0), 1e-6f);
        }
    }

    @Test
    void testReadMmap() throws IOException {
        Path file = tempDir.resolve("test_mmap.npy");
        NDArray original = JNum.from(new double[]{10.0, 20.0, 30.0}, 3);

        NpyWriter.write(original, file);

        try (Arena arena = Arena.ofShared()) {
            NDArray mmapped = NpyReader.readMmap(file, arena);
            assertEquals(DType.f64, mmapped.getDType());
            assertEquals(3, mmapped.getSize());
            assertEquals(10.0, mmapped.getDouble(0), 1e-9);
            assertEquals(30.0, mmapped.getDouble(2), 1e-9);
        }
    }
}
