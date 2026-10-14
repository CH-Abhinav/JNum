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
