package jnum.nn;

import jnum.NDArray;

public class Sigmoid implements Module {
    @Override
    public NDArray forward(NDArray input) {
        return input.sigmoid();
    }
}