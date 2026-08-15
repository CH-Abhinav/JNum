package jnum.nn;

import jnum.NDArray;

public interface Module {
    /**
     * Executes the forward pass of this module
     * @param input The input Array
     * @return The computed output Array
     **/
    NDArray forward(NDArray input);
}
