package jnum.stat;

import java.lang.foreign.Arena;
import jnum.NDArray;
import jnum.internal.ops.StatOps;

/**
 * High-performance statistical operations for JNum.
 * All statistical reductions and summary metrics are housed exclusively in this facade.
 */
public final class Stats {

    private Stats() {
        throw new AssertionError("Stats facade cannot be instantiated.");
    }

    // =========================================================================
    // ARITHMETIC MEAN
    // =========================================================================

    /**
     * Computes the global arithmetic mean of all elements in the array.
     */
    public static double mean(NDArray a) {
        return StatOps.mean(a);
    }

    /**
     * Computes the arithmetic mean along the specified axis.
     */
    public static NDArray mean(NDArray a, int axis) {
        return StatOps.mean(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the arithmetic mean along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray mean(NDArray a, int axis, boolean keepDims) {
        return StatOps.mean(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the arithmetic mean along the specified axis with a custom memory Arena.
     */
    public static NDArray mean(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.mean(a, axis, keepDims, arena);
    }

    // =========================================================================
    // WEIGHTED AVERAGE
    // =========================================================================

    /**
     * Computes the unweighted arithmetic mean of all elements (delegates to mean).
     */
    public static double average(NDArray a) {
        return StatOps.mean(a);
    }

    /**
     * Computes the weighted average of all elements: sum(a * weights) / sum(weights).
     */
    public static double average(NDArray a, NDArray weights) {
        return StatOps.average(a, weights);
    }

    /**
     * Computes the weighted average along the specified axis.
     */
    public static NDArray average(NDArray a, int axis, NDArray weights) {
        return StatOps.average(a, axis, weights, false, Arena.ofAuto());
    }

    /**
     * Computes the weighted average along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray average(NDArray a, int axis, NDArray weights, boolean keepDims) {
        return StatOps.average(a, axis, weights, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the weighted average along the specified axis with a custom memory Arena.
     */
    public static NDArray average(NDArray a, int axis, NDArray weights, boolean keepDims, Arena arena) {
        return StatOps.average(a, axis, weights, keepDims, arena);
    }

    // =========================================================================
    // VARIANCE
    // =========================================================================

    /**
     * Computes the population variance (ddof = 0) of all elements in the array.
     */
    public static double var(NDArray a) {
        return StatOps.var(a, 0);
    }

    /**
     * Computes the sample variance (ddof = 1) of all elements in the array.
     */
    public static double sampleVar(NDArray a) {
        return StatOps.var(a, 1);
    }

    /**
     * Computes the scalar variance of all elements with custom delta degrees of freedom (ddof).
     */
    public static double varScalar(NDArray a, int ddof) {
        return StatOps.var(a, ddof);
    }

    /**
     * Computes the population variance (ddof = 0) along the specified axis.
     */
    public static NDArray var(NDArray a, int axis) {
        return StatOps.var(a, axis, 0, false, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with the specified ddof.
     */
    public static NDArray var(NDArray a, int axis, int ddof) {
        return StatOps.var(a, axis, ddof, false, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with the specified ddof, optionally preserving reduced dimensions.
     */
    public static NDArray var(NDArray a, int axis, int ddof, boolean keepDims) {
        return StatOps.var(a, axis, ddof, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with a custom memory Arena.
     */
    public static NDArray var(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        return StatOps.var(a, axis, ddof, keepDims, arena);
    }

    // =========================================================================
    // STANDARD DEVIATION
    // =========================================================================

    /**
     * Computes the population standard deviation (ddof = 0) of all elements in the array.
     */
    public static double std(NDArray a) {
        return StatOps.std(a, 0);
    }

    /**
     * Computes the sample standard deviation (ddof = 1) of all elements in the array.
     */
    public static double sampleStd(NDArray a) {
        return StatOps.std(a, 1);
    }

    /**
     * Computes the scalar standard deviation of all elements with custom delta degrees of freedom (ddof).
     */
    public static double stdScalar(NDArray a, int ddof) {
        return StatOps.std(a, ddof);
    }

    /**
     * Computes the population standard deviation (ddof = 0) along the specified axis.
     */
    public static NDArray std(NDArray a, int axis) {
        return StatOps.std(a, axis, 0, false, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with the specified ddof.
     */
    public static NDArray std(NDArray a, int axis, int ddof) {
        return StatOps.std(a, axis, ddof, false, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with the specified ddof, optionally preserving reduced dimensions.
     */
    public static NDArray std(NDArray a, int axis, int ddof, boolean keepDims) {
        return StatOps.std(a, axis, ddof, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with a custom memory Arena.
     */
    public static NDArray std(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        return StatOps.std(a, axis, ddof, keepDims, arena);
    }

    // =========================================================================
    // PEAK-TO-PEAK (PTP: MAX - MIN)
    // =========================================================================

    /**
     * Computes the peak-to-peak range (maximum - minimum) across all elements in the array.
     */
    public static double ptp(NDArray a) {
        return StatOps.ptp(a);
    }

    /**
     * Computes the peak-to-peak range along the specified axis.
     */
    public static NDArray ptp(NDArray a, int axis) {
        return StatOps.ptp(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the peak-to-peak range along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray ptp(NDArray a, int axis, boolean keepDims) {
        return StatOps.ptp(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the peak-to-peak range along the specified axis with a custom memory Arena.
     */
    public static NDArray ptp(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.ptp(a, axis, keepDims, arena);
    }

    // =========================================================================
    // MEDIAN
    // =========================================================================

    /**
     * Computes the median of all elements in the array using O(N) Introselect.
     */
    public static double median(NDArray a) {
        return StatOps.median(a);
    }

    /**
     * Computes the median along the specified axis using O(N) Introselect.
     */
    public static NDArray median(NDArray a, int axis) {
        return StatOps.median(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the median along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray median(NDArray a, int axis, boolean keepDims) {
        return StatOps.median(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the median along the specified axis with a custom memory Arena.
     */
    public static NDArray median(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.median(a, axis, keepDims, arena);
    }

    // =========================================================================
    // QUANTILE & PERCENTILE
    // =========================================================================

    /**
     * Computes the q-th quantile of all elements (q in [0.0, 1.0]) with linear interpolation.
     */
    public static double quantile(NDArray a, double q) {
        return StatOps.quantile(a, q, QuantileMethod.LINEAR);
    }

    /**
     * Computes the q-th quantile of all elements with the specified interpolation method.
     */
    public static double quantile(NDArray a, double q, QuantileMethod method) {
        return StatOps.quantile(a, q, method);
    }

    /**
     * Computes the q-th quantile along the specified axis with linear interpolation.
     */
    public static NDArray quantile(NDArray a, double q, int axis) {
        return StatOps.quantile(a, q, axis, QuantileMethod.LINEAR, false, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray quantile(NDArray a, double q, int axis, boolean keepDims) {
        return StatOps.quantile(a, q, axis, QuantileMethod.LINEAR, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with the specified interpolation method.
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method) {
        return StatOps.quantile(a, q, axis, method, false, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with the specified interpolation method and keepDims.
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method, boolean keepDims) {
        return StatOps.quantile(a, q, axis, method, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with a custom memory Arena.
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method, boolean keepDims, Arena arena) {
        return StatOps.quantile(a, q, axis, method, keepDims, arena);
    }

    /**
     * Computes the p-th percentile of all elements (p in [0.0, 100.0]) with linear interpolation.
     */
    public static double percentile(NDArray a, double p) {
        return quantile(a, p / 100.0, QuantileMethod.LINEAR);
    }

    /**
     * Computes the p-th percentile of all elements with the specified interpolation method.
     */
    public static double percentile(NDArray a, double p, QuantileMethod method) {
        return quantile(a, p / 100.0, method);
    }

    /**
     * Computes the p-th percentile along the specified axis with linear interpolation.
     */
    public static NDArray percentile(NDArray a, double p, int axis) {
        return quantile(a, p / 100.0, axis, QuantileMethod.LINEAR, false, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray percentile(NDArray a, double p, int axis, boolean keepDims) {
        return quantile(a, p / 100.0, axis, QuantileMethod.LINEAR, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with the specified interpolation method.
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method) {
        return quantile(a, p / 100.0, axis, method, false, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with the specified interpolation method and keepDims.
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method, boolean keepDims) {
        return quantile(a, p / 100.0, axis, method, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with a custom memory Arena.
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method, boolean keepDims, Arena arena) {
        return quantile(a, p / 100.0, axis, method, keepDims, arena);
    }

    // =========================================================================
    // INTERQUARTILE RANGE (IQR: Q75 - Q25)
    // =========================================================================

    /**
     * Computes the interquartile range (75th percentile - 25th percentile) across all elements.
     */
    public static double iqr(NDArray a) {
        return StatOps.iqr(a);
    }

    /**
     * Computes the interquartile range along the specified axis.
     */
    public static NDArray iqr(NDArray a, int axis) {
        return StatOps.iqr(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the interquartile range along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray iqr(NDArray a, int axis, boolean keepDims) {
        return StatOps.iqr(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the interquartile range along the specified axis with a custom memory Arena.
     */
    public static NDArray iqr(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.iqr(a, axis, keepDims, arena);
    }

    // =========================================================================
    // MODE
    // =========================================================================

    /**
     * Computes the mode (most frequent value) across all elements.
     * Ties are broken by choosing the smallest value.
     */
    public static double mode(NDArray a) {
        return StatOps.mode(a);
    }

    /**
     * Computes the mode along the specified axis.
     */
    public static NDArray mode(NDArray a, int axis) {
        return StatOps.mode(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the mode along the specified axis, optionally preserving reduced dimensions.
     */
    public static NDArray mode(NDArray a, int axis, boolean keepDims) {
        return StatOps.mode(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the mode along the specified axis with a custom memory Arena.
     */
    public static NDArray mode(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.mode(a, axis, keepDims, arena);
    }

    // =========================================================================
    // COVARIANCE
    // =========================================================================

    /**
     * Computes sample covariance between two 1D arrays (ddof = 1).
     */
    public static double cov(NDArray x, NDArray y) {
        return StatOps.cov(x, y, 1);
    }

    /**
     * Computes covariance between two 1D arrays with specified degrees of freedom.
     */
    public static double cov(NDArray x, NDArray y, int ddof) {
        return StatOps.cov(x, y, ddof);
    }

    /**
     * Estimates the covariance matrix of a 2D array where rows are variables and columns are observations (ddof = 1).
     */
    public static NDArray cov(NDArray m) {
        return StatOps.cov(m, true, 1, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix of a 2D array with custom variable orientation (ddof = 1).
     *
     * @param m 2D input array.
     * @param rowvar If true, rows represent variables; if false, columns represent variables.
     */
    public static NDArray cov(NDArray m, boolean rowvar) {
        return StatOps.cov(m, rowvar, 1, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix of a 2D array with custom variable orientation and degrees of freedom.
     */
    public static NDArray cov(NDArray m, boolean rowvar, int ddof) {
        return StatOps.cov(m, rowvar, ddof, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix of a 2D array with custom variable orientation, degrees of freedom, and memory Arena.
     */
    public static NDArray cov(NDArray m, boolean rowvar, int ddof, Arena arena) {
        return StatOps.cov(m, rowvar, ddof, arena);
    }

    // =========================================================================
    // CUMULATIVE SUM (CUMSUM)
    // =========================================================================

    /**
     * Computes the cumulative sum of elements over the flattened array.
     */
    public static NDArray cumsum(NDArray a) {
        return StatOps.cumsum(a, Arena.ofAuto());
    }

    /**
     * Computes the cumulative sum of elements over the flattened array with a custom memory Arena.
     */
    public static NDArray cumsum(NDArray a, Arena arena) {
        return StatOps.cumsum(a, arena);
    }

    /**
     * Computes the cumulative sum of elements along the specified axis.
     */
    public static NDArray cumsum(NDArray a, int axis) {
        return StatOps.cumsum(a, axis, Arena.ofAuto());
    }

    /**
     * Computes the cumulative sum of elements along the specified axis with a custom memory Arena.
     */
    public static NDArray cumsum(NDArray a, int axis, Arena arena) {
        return StatOps.cumsum(a, axis, arena);
    }

    // =========================================================================
    // CUMULATIVE PRODUCT (CUMPROD)
    // =========================================================================

    /**
     * Computes the cumulative product of elements over the flattened array.
     */
    public static NDArray cumprod(NDArray a) {
        return StatOps.cumprod(a, Arena.ofAuto());
    }

    /**
     * Computes the cumulative product of elements over the flattened array with a custom memory Arena.
     */
    public static NDArray cumprod(NDArray a, Arena arena) {
        return StatOps.cumprod(a, arena);
    }

    /**
     * Computes the cumulative product of elements along the specified axis.
     */
    public static NDArray cumprod(NDArray a, int axis) {
        return StatOps.cumprod(a, axis, Arena.ofAuto());
    }

    /**
     * Computes the cumulative product of elements along the specified axis with a custom memory Arena.
     */
    public static NDArray cumprod(NDArray a, int axis, Arena arena) {
        return StatOps.cumprod(a, axis, arena);
    }

    // =========================================================================
    // DISCRETE DIFFERENCE (DIFF)
    // =========================================================================

    /**
     * Computes the first discrete difference along the last axis.
     */
    public static NDArray diff(NDArray a) {
        return StatOps.diff(a, 1, -1, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the last axis.
     */
    public static NDArray diff(NDArray a, int n) {
        return StatOps.diff(a, n, -1, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the specified axis.
     */
    public static NDArray diff(NDArray a, int n, int axis) {
        return StatOps.diff(a, n, axis, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the specified axis with a custom memory Arena.
     */
    public static NDArray diff(NDArray a, int n, int axis, Arena arena) {
        return StatOps.diff(a, n, axis, arena);
    }
}
