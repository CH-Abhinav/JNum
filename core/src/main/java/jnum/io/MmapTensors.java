package jnum.io;

import java.lang.foreign.Arena;
import java.util.Map;
import jnum.NDArray;

/**
 * Safe AutoCloseable container holding memory-mapped tensors and their backing Arena.
 */
public final class MmapTensors implements AutoCloseable {

    private final Map<String, NDArray> tensors;
    private final Arena arena;

    public MmapTensors(Map<String, NDArray> tensors, Arena arena) {
        this.tensors = tensors;
        this.arena = arena;
    }

    public Map<String, NDArray> tensors() {
        return tensors;
    }

    public NDArray get(String name) {
        return tensors.get(name);
    }

    public Arena arena() {
        return arena;
    }

    @Override
    public void close() {
        arena.close();
    }
}
