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
 * Base interface for all neural network modules and computational layers.
 *
 * <p>A module encapsulates parameterized transformations or mathematical operations that map
 * an input tensor to an output tensor during forward inference.</p>
 */
public interface Module {

    /**
     * Executes the forward computation pass of this module.
     *
     * @param input the input tensor to the layer
     * @return the computed output tensor
     * @throws IllegalArgumentException if the input tensor shape or dtype is incompatible with this module
     */
    NDArray forward(NDArray input);
}
