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
 * Normalized exponential (Softmax) activation function:
 * {@code Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))}.
 *
 * <p>Rescales elements along a specified axis so that they lie in the range {@code [0, 1]}
 * and sum to {@code 1.0}. Employs the standard numerical stability trick (subtracting the axis maximum)
 * to prevent floating-point overflow during exponentiation.</p>
 */
public class Softmax implements Module {

    private final int axis;

    /**
     * Constructs a Softmax module operating along the specified axis.
     *
     * @param axis the dimension along which softmax normalization is performed (supports negative indices)
     */
    public Softmax(int axis) {
        this.axis = axis;
    }

    /**
     * Constructs a Softmax module operating along the trailing dimension ({@code axis = -1}).
     */
    public Softmax() {
        this.axis = -1;
    }

    /**
     * Returns the axis along which softmax is applied.
     *
     * @return target normalization axis
     */
    public int axis() {
        return axis;
    }

    /**
     * Applies softmax probability normalization along the configured axis.
     *
     * @param input the unnormalized logits tensor
     * @return a normalized probability distribution tensor with the same shape
     */
    @Override
    public NDArray forward(NDArray input) {
        int targetAxis = axis < 0 ? axis + input.dim() : axis;

        // 1. Numerical Stability: Subtract the maximum value along the axis
        // keepDims=true ensures maxVals is [Batch, 1], broadcasting safely against [Batch, Features]
        NDArray maxVals = input.max(targetAxis, true);
        NDArray shifted = input.sub(maxVals);

        // 2. Exponentiate the shifted values
        NDArray expVals = shifted.exp();

        // 3. Normalize by dividing by the sum of the exponents
        NDArray sumExp = expVals.sum(targetAxis, true);

        return expVals.div(sumExp);
    }
}