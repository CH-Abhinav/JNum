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
