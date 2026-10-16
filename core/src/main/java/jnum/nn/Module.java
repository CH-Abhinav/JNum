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
