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
package jnum.nn;

import jnum.NDArray;

/**
 * Rectified Linear Unit (ReLU) activation function: {@code ReLU(x) = max(0, x)}.
 *
 * <p>Clamps negative values to zero while passing positive values through unchanged.</p>
 */
public class ReLU implements Module {

    /**
     * Default constructor.
     */
    public ReLU() {}

    /**
     * Applies the ReLU activation function element-wise to the input tensor.
     *
     * @param input the input tensor
     * @return a new tensor containing {@code max(0, x)} for each element
     * @throws IllegalArgumentException if the data type does not support comparison (e.g. boolean)
     */
    @Override
    public NDArray forward(NDArray input) {
        return switch (input.getDType()) {
            case f32 -> input.maximum(0.0f);
            case f64 -> input.maximum(0.0);
            case i32 -> input.maximum(0);
            default -> throw new IllegalArgumentException("ReLU unsupported for dtype: " + input.getDType());
        };
    }
}
