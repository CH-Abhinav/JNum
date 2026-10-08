package jnum.nn;

import jnum.NDArray;
import java.util.Arrays;
import java.util.List;

/**
 * A sequential container. Modules will be added to it in the order they are passed in the constructor.
 */
public class Sequential implements Module {
    private final List<Module> layers;

    public Sequential(Module... layers) {
        this.layers = Arrays.asList(layers);
    }

    @Override
    public NDArray forward(NDArray input) {
        NDArray output = input;
        for (Module layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
}