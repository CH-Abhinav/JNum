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
package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class MmapTensorsTest {

    @Test
    void testMmapTensorsAccessAndClose() {
        Arena arena = Arena.ofShared();
        NDArray t1 = JNum.from(new float[]{1.0f, 2.0f}, 2);
        NDArray t2 = JNum.from(new int[]{3, 4, 5}, 3);
        Map<String, NDArray> map = Map.of("weight", t1, "bias", t2);

        MmapTensors tensors = new MmapTensors(map, arena);

        assertEquals(2, tensors.tensors().size());
        assertSame(t1, tensors.get("weight"));
        assertSame(t2, tensors.get("bias"));
        assertNull(tensors.get("nonexistent"));
        assertSame(arena, tensors.arena());
        assertTrue(arena.scope().isAlive());

        tensors.close();
        assertFalse(arena.scope().isAlive());
    }

    @Test
    void testTryWithResources() {
        Arena arena = Arena.ofShared();
        NDArray t1 = JNum.from(new double[]{1.0}, 1);
        try (MmapTensors tensors = new MmapTensors(Map.of("x", t1), arena)) {
            assertNotNull(tensors.get("x"));
            assertTrue(tensors.arena().scope().isAlive());
        }
        assertFalse(arena.scope().isAlive());
    }
}
