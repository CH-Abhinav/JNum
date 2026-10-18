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
import java.util.Map;
import jnum.NDArray;

/**
 * Safe {@link AutoCloseable} container holding multiple memory-mapped tensors and their backing {@link Arena}.
 *
 * <p>Ideal for zero-copy loading of entire neural network checkpoint weights from {@code .safetensors} files.
 * Calling {@link #close()} deterministically unmaps the entire archive from memory.</p>
 */
public final class MmapTensors implements AutoCloseable {

    private final Map<String, NDArray> tensors;
    private final Arena arena;

    /**
     * Constructs a container wrapping the named tensor dictionary and its governing arena.
     *
     * @param tensors map of tensor names to memory-mapped tensors
     * @param arena the arena governing memory mapping lifecycle
     */
    public MmapTensors(Map<String, NDArray> tensors, Arena arena) {
        this.tensors = tensors;
        this.arena = arena;
    }

    /**
     * Returns an unmodifiable view of all mapped tensors.
     *
     * @return map of tensor names to {@link NDArray} instances
     */
    public Map<String, NDArray> tensors() {
        return tensors;
    }

    /**
     * Looks up a mapped tensor by its weight name.
     *
     * @param name tensor name (e.g. {@code "model.embed_tokens.weight"})
     * @return the mapped {@link NDArray}, or {@code null} if not found
     */
    public NDArray get(String name) {
        return tensors.get(name);
    }

    /**
     * Retrieves the underlying memory {@link Arena}.
     *
     * @return the arena managing the mapped file segment
     */
    public Arena arena() {
        return arena;
    }

    /**
     * Closes the arena and unmaps all tensor memory from process address space.
     * Subsequent access to any of the contained tensors will throw {@link IllegalStateException}.
     */
    @Override
    public void close() {
        arena.close();
    }
}

