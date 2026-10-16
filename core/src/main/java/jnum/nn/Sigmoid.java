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