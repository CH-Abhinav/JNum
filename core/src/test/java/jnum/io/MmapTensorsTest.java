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
