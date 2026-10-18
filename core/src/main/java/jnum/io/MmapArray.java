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

import java.lang.foreign.Arena;
import jnum.NDArray;

/**
 * Safe {@link AutoCloseable} container holding a memory-mapped {@link NDArray} and its backing {@link Arena}.
 *
 * <p>Deterministic lifecycle management: closing this resource unmaps the native memory buffer
 * immediately via {@link Arena#close()}, freeing virtual memory back to the OS without waiting
 * for garbage collection.</p>
 */
public final class MmapArray implements AutoCloseable {

    private final NDArray array;
    private final Arena arena;

    /**
     * Constructs a new container wrapping the mapped tensor and arena.
     *
     * @param array the memory-mapped tensor
     * @param arena the arena governing the memory mapping lifecycle
     */
    public MmapArray(NDArray array, Arena arena) {
        this.array = array;
        this.arena = arena;
    }

    /**
     * Retrieves the memory-mapped {@link NDArray}.
     *
     * @return the mapped tensor
     */
    public NDArray array() {
        return array;
    }

    /**
     * Retrieves the underlying memory {@link Arena}.
     *
     * @return the arena managing the mapped segment
     */
    public Arena arena() {
        return arena;
    }

    /**
     * Closes the backing arena and unmaps the physical memory segment from process address space.
     * Subsequent access to the wrapped tensor will throw {@link IllegalStateException}.
     */
    @Override
    public void close() {
        arena.close();
    }
}