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
package jnum.internal.kernel.stat;

import jnum.NDArray;
import jnum.stat.QuantileMethod;

/**
 * High-performance Median kernels utilizing O(N) Introselect.
 */
public final class Median {

    private Median() {
        throw new AssertionError("Median kernel cannot be instantiated.");
    }

    public static double medianFloat(NDArray a) {
        return Quantile.quantileFloat(a, 0.5, QuantileMethod.LINEAR);
    }

    public static double medianDouble(NDArray a) {
        return Quantile.quantileDouble(a, 0.5, QuantileMethod.LINEAR);
    }

    public static double medianInt(NDArray a) {
        return Quantile.quantileInt(a, 0.5, QuantileMethod.LINEAR);
    }

    public static NDArray medianFloatAxis(NDArray a, int axis, NDArray resArray) {
        return Quantile.quantileFloatAxis(a, axis, 0.5, QuantileMethod.LINEAR, resArray);
    }

    public static NDArray medianDoubleAxis(NDArray a, int axis, NDArray resArray) {
        return Quantile.quantileDoubleAxis(a, axis, 0.5, QuantileMethod.LINEAR, resArray);
    }

    public static NDArray medianIntAxis(NDArray a, int axis, NDArray resArray) {
        return Quantile.quantileIntAxis(a, axis, 0.5, QuantileMethod.LINEAR, resArray);
    }
}
