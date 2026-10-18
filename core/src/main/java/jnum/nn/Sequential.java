/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.nn;

import jnum.NDArray;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A sequential container module that chains multiple {@link Module} layers in the order they are provided.
 *
 * <p>During the forward pass, the output of each layer is passed sequentially as the input to the next layer:
 * {@code output = layer_n(...(layer_2(layer_1(input))))}.</p>
 */
public class Sequential implements Module {

    private final List<Module> layers;

    /**
     * Constructs a sequential pipeline of layers.
     *
     * @param layers ordered sequence of computational modules
     */
    public Sequential(Module... layers) {
        this.layers = Arrays.asList(layers);
    }

    /**
     * Constructs a sequential pipeline from a list of layers.
     *
     * @param layers ordered list of computational modules
     */
    public Sequential(List<Module> layers) {
        this.layers = List.copyOf(layers);
    }

    /**
     * Returns an unmodifiable view of the constituent layers.
     *
     * @return unmodifiable list of layers
     */
    public List<Module> layers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * Returns the number of layers in this sequential container.
     *
     * @return number of constituent layers
     */
    public int size() {
        return layers.size();
    }

    /**
     * Executes forward propagation sequentially through every layer in the container.
     *
     * @param input the input tensor to the first layer
     * @return the final output tensor produced by the last layer
     */
    @Override
    public NDArray forward(NDArray input) {
        NDArray output = input;
        for (Module layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
}