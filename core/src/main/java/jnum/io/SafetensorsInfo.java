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

import java.util.Map;
import java.util.Set;
import jnum.DType;

/**
 * Inspection record containing metadata of a {@code .safetensors} file without loading tensor payload bytes.
 *
 * <p>Enables ultra-fast header inspection of multi-gigabyte neural network checkpoints in microseconds.</p>
 *
 * @param tensors map of tensor names to their metadata descriptors
 * @param userMetadata key-value string metadata embedded in the header (e.g. format version, model architecture)
 * @param headerByteSize size in bytes of the JSON header prefix
 */
public record SafetensorsInfo(
        Map<String, TensorInfo> tensors,
        Map<String, String> userMetadata,
        long headerByteSize
) {
    /**
     * Returns the total number of tensors in this safetensors file.
     *
     * @return the tensor count
     */
    public int tensorCount() {
        return tensors.size();
    }

    /**
     * Returns the set of tensor names contained in this safetensors file.
     *
     * @return set of tensor names
     */
    public Set<String> tensorNames() {
        return tensors.keySet();
    }

    /**
     * Returns the dimensional extent shape of the tensor with the given name, or {@code null} if not found.
     *
     * @param tensorName the name of the tensor
     * @return the shape array, or null if absent
     */
    public long[] getShape(String tensorName) {
        TensorInfo info = tensors.get(tensorName);
        return info != null ? info.shape() : null;
    }

    /**
     * Returns the string representation of the data type of the tensor with the given name, or {@code null} if not found.
     *
     * @param tensorName the name of the tensor
     * @return the dtype name string, or null if absent
     */
    public String getDType(String tensorName) {
        TensorInfo info = tensors.get(tensorName);
        return (info != null && info.dtype() != null) ? info.dtype().name() : null;
    }
    /**
     * Metadata descriptor for an individual tensor within a {@code .safetensors} file.
     *
     * @param name unique name of the tensor (e.g. {@code "model.layers.0.weight"})
     * @param dtype data type descriptor
     * @param shape dimensional extent array
     * @param byteSize total payload size in bytes
     */
    public record TensorInfo(
            String name,
            DType dtype,
            long[] shape,
            long byteSize
    ) {}
}

