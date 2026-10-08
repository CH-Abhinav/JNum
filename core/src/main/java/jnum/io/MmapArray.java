package jnum.io;

import java.lang.foreign.Arena;
import jnum.NDArray;

/**
 * Safe AutoCloseable container holding a memory-mapped NDArray and its backing Arena.
 * Deterministically closes and unmaps memory on close().
 */
public final class MmapArray implements AutoCloseable {

    private final NDArray array;
    private final Arena arena;

    public MmapArray(NDArray array, Arena arena) {
        this.array = array;
        this.arena = arena;
    }

    public NDArray array() {
        return array;
    }

    public Arena arena() {
        return arena;
    }

    @Override
    public void close() {
        arena.close();
    }
}