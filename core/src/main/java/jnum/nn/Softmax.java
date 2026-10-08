package jnum.nn;

import jnum.NDArray;

/**
 * Applies the Softmax function to an n-dimensional input Tensor.
 * Rescales elements so they lie in the range [0, 1] and sum to 1.
 */
public class Softmax implements Module {
    private final int axis;

    /**
     * @param axis The dimension softmax would be performed on.
     */
    public Softmax(int axis) {
        this.axis = axis;
    }

    /**
     * Defaults to the last dimension (axis = -1).
     */
    public Softmax() {
        this.axis = -1;
    }

    @Override
    public NDArray forward(NDArray input) {
        int targetAxis = axis < 0 ? axis + input.dim() : axis;

        // 1. Numerical Stability: Subtract the maximum value along the axis
        // keepdims=true ensures maxVals is [Batch, 1], broadcasting safely against [Batch, Features]
        NDArray maxVals = input.max(targetAxis, true);
        NDArray shifted = input.sub(maxVals);

        // 2. Exponentiate the shifted values
        NDArray expVals = shifted.exp();

        // 3. Normalize by dividing by the sum of the exponents
        NDArray sumExp = expVals.sum(targetAxis, true);

        return expVals.div(sumExp);
    }
}