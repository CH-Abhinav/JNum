package jnum.nn;

import jnum.NDArray;

public class Tanh implements Module {
    @Override
    public NDArray forward(NDArray input) {
        return input.tanh();
    }
}