package jnum.nn;

import jnum.NDArray;

public class ReLU implements Module {
    @Override
    public NDArray forward(NDArray input){
        return switch (input.getDType()) {
            case f32 -> input.maximum(0.0f);
            case f64 -> input.maximum(0.0);
            case i32 -> input.maximum(0);
            default -> throw new IllegalArgumentException("ReLU unsupported for dtype: " + input.getDType());
        };
    }
}
