package jnum.nn;

import jnum.NDArray;

/**
 * Hyperbolic Tangent (Tanh) activation function: {@code Tanh(x) = (e^x - e^-x) / (e^x + e^-x)}.
 *
 * <p>Compresses real-valued inputs into the zero-centered range {@code (-1, 1)}.</p>
 */
public class Tanh implements Module {

    /**
     * Default constructor.
     */
    public Tanh() {}

    /**
     * Applies the hyperbolic tangent function element-wise to the input tensor.
     *
     * @param input the input tensor
     * @return a new tensor containing tanh activations
     */
    @Override
    public NDArray forward(NDArray input) {
        return input.tanh();
    }
}