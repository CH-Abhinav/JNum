package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class MmapArrayTest {

    @Test
    void testMmapArrayAccessAndClose() {
        Arena arena = Arena.ofShared();
        NDArray arr = JNum.from(new float[]{1.0f, 2.0f, 3.0f}, 3);
        MmapArray mmap = new MmapArray(arr, arena);

        assertSame(arr, mmap.array());
        assertSame(arena, mmap.arena());
        assertTrue(arena.scope().isAlive());

        mmap.close();
        assertFalse(arena.scope().isAlive());
    }

    @Test
    void testTryWithResources() {
        Arena arena = Arena.ofShared();
        NDArray arr = JNum.from(new double[]{4.0, 5.0}, 2);
        try (MmapArray mmap = new MmapArray(arr, arena)) {
            assertEquals(2, mmap.array().getSize());
            assertTrue(mmap.arena().scope().isAlive());
        }
        assertFalse(arena.scope().isAlive());
    }
}
