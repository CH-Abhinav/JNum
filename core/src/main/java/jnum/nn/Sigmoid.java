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
 * Logistic Sigmoid activation function: {@code Sigmoid(x) = 1 / (1 + e^-x)}.
 *
 * <p>Compresses real-valued inputs into the probability interval {@code (0, 1)}.</p>
 */
public class Sigmoid implements Module {

    /**
     * Default constructor.
     */
    public Sigmoid() {}

    /**
     * Applies the logistic sigmoid function element-wise to the input tensor.
     *
     * @param input the input tensor
     * @return a new tensor containing sigmoid activations
     */
    @Override
    public NDArray forward(NDArray input) {
        return input.sigmoid();
    }
}