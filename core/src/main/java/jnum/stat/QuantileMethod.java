package jnum.stat;

/**
 * Interpolation methods for quantile and percentile computation, matching NumPy conventions.
 */
public enum QuantileMethod {
    /**
     * Linear interpolation: i + (j - i) * fraction (NumPy default).
     */
    LINEAR,

    /**
     * Takes the lower index: i.
     */
    LOWER,

    /**
     * Takes the higher index: j.
     */
    HIGHER,

    /**
     * Midpoint between lower and higher: (i + j) / 2.
     */
    MIDPOINT,

    /**
     * Takes the nearest index to the virtual position.
     */
    NEAREST
}
