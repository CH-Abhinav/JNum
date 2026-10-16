package jnum.stat;

/**
 * Interpolation methods for quantile and percentile computation, matching NumPy conventions.
 *
 * <p>When computing quantile {@code q in [0, 1]} on a sorted array of length {@code N}, the virtual index
 * is given by {@code index = q * (N - 1)}. Let {@code i = floor(index)}, {@code j = ceil(index)}, and
 * {@code fraction = index - i}. The interpolation method determines how the estimated value is computed
 * between values at index {@code i} and index {@code j}.</p>
 */
public enum QuantileMethod {

    /**
     * Linear interpolation: {@code x[i] + (x[j] - x[i]) * fraction}. Standard NumPy default.
     */
    LINEAR,

    /**
     * Takes the lower bound: {@code x[i]}.
     */
    LOWER,

    /**
     * Takes the upper bound: {@code x[j]}.
     */
    HIGHER,

    /**
     * Takes the midpoint between lower and upper bounds: {@code (x[i] + x[j]) / 2.0}.
     */
    MIDPOINT,

    /**
     * Takes the element at the index closest to the virtual position (half rounds to nearest even).
     */
    NEAREST
}

