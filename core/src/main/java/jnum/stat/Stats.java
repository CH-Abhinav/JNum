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
     * Computes the global arithmetic mean across all elements in the array.
     *
     * @param a the input tensor
     * @return the scalar arithmetic mean promoted to {@code double}
     */
    public static double mean(NDArray a) {
        return StatOps.mean(a);
    }

    /**
     * Computes the arithmetic mean along the specified axis, collapsing that dimension.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute the mean (supports negative indices)
     * @return a newly allocated {@link NDArray} with rank reduced by 1
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public static NDArray mean(NDArray a, int axis) {
        return StatOps.mean(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the arithmetic mean along the specified axis, optionally retaining the reduced dimension as size 1.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute the mean
     * @param keepDims whether to preserve the reduced dimension with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray mean(NDArray a, int axis, boolean keepDims) {
        return StatOps.mean(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the arithmetic mean along the specified axis using a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute the mean
     * @param keepDims whether to preserve the reduced dimension with size 1
     * @param arena memory arena governing the lifecycle of the returned tensor
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray mean(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.mean(a, axis, keepDims, arena);
    }

    // =========================================================================
    // WEIGHTED AVERAGE
    // =========================================================================

    /**
     * Computes the unweighted arithmetic mean across all elements (delegates to {@link #mean(NDArray)}).
     *
     * @param a the input tensor
     * @return the scalar mean
     */
    public static double average(NDArray a) {
        return StatOps.mean(a);
    }

    /**
     * Computes the weighted average across all elements: {@code sum(a * weights) / sum(weights)}.
     *
     * @param a the input tensor
     * @param weights tensor of weights broadcast-compatible with {@code a}
     * @return the scalar weighted average
     * @throws IllegalArgumentException if weights cannot be aligned or sum of weights is zero
     */
    public static double average(NDArray a, NDArray weights) {
        return StatOps.average(a, weights);
    }

    /**
     * Computes the weighted average along the specified axis, collapsing that dimension.
     *
     * @param a the input tensor
     * @param axis dimension along which to average
     * @param weights tensor of weights
     * @return reduced {@link NDArray} with rank reduced by 1
     * @throws IllegalArgumentException if {@code axis} is out of bounds or weights are incompatible
     */
    public static NDArray average(NDArray a, int axis, NDArray weights) {
        return StatOps.average(a, axis, weights, false, Arena.ofAuto());
    }

    /**
     * Computes the weighted average along the specified axis, optionally retaining the reduced dimension as size 1.
     *
     * @param a the input tensor
     * @param axis dimension along which to average
     * @param weights tensor of weights
     * @param keepDims whether to preserve the reduced dimension with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds or weights are incompatible
     */
    public static NDArray average(NDArray a, int axis, NDArray weights, boolean keepDims) {
        return StatOps.average(a, axis, weights, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the weighted average along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to average
     * @param weights tensor of weights
     * @param keepDims whether to preserve the reduced dimension with size 1
     * @param arena memory arena governing lifecycle of the result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds or weights are incompatible
     */
    public static NDArray average(NDArray a, int axis, NDArray weights, boolean keepDims, Arena arena) {
        return StatOps.average(a, axis, weights, keepDims, arena);
    }

    // =========================================================================
    // VARIANCE
    // =========================================================================

    /**
     * Computes the population variance ({@code ddof = 0}) across all elements in the array.
     *
     * @param a the input tensor
     * @return the scalar population variance
     */
    public static double var(NDArray a) {
        return StatOps.var(a, 0);
    }

    /**
     * Computes the sample variance ({@code ddof = 1}) across all elements in the array.
     *
     * @param a the input tensor
     * @return the scalar sample variance
     */
    public static double sampleVar(NDArray a) {
        return StatOps.var(a, 1);
    }

    /**
     * Computes the scalar variance across all elements with custom delta degrees of freedom ({@code ddof}).
     *
     * <p>The divisor used in calculation is {@code N - ddof}, where {@code N} is the total element count.</p>
     *
     * @param a the input tensor
     * @param ddof delta degrees of freedom
     * @return the scalar variance
     */
    public static double varScalar(NDArray a, int ddof) {
        return StatOps.var(a, ddof);
    }

    /**
     * Computes the population variance ({@code ddof = 0}) along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute variance
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray var(NDArray a, int axis) {
        return StatOps.var(a, axis, 0, false, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with specified delta degrees of freedom ({@code ddof}).
     *
     * @param a the input tensor
     * @param axis dimension along which to compute variance
     * @param ddof delta degrees of freedom
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray var(NDArray a, int axis, int ddof) {
        return StatOps.var(a, axis, ddof, false, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with specified {@code ddof}, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute variance
     * @param ddof delta degrees of freedom
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray var(NDArray a, int axis, int ddof, boolean keepDims) {
        return StatOps.var(a, axis, ddof, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the variance along the specified axis with specified {@code ddof} and custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute variance
     * @param ddof delta degrees of freedom
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray var(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        return StatOps.var(a, axis, ddof, keepDims, arena);
    }

    // =========================================================================
    // STANDARD DEVIATION
    // =========================================================================

    /**
     * Computes the population standard deviation ({@code ddof = 0}) across all elements in the array.
     *
     * @param a the input tensor
     * @return the scalar standard deviation
     */
    public static double std(NDArray a) {
        return StatOps.std(a, 0);
    }

    /**
     * Computes the sample standard deviation ({@code ddof = 1}) across all elements in the array.
     *
     * @param a the input tensor
     * @return the scalar sample standard deviation
     */
    public static double sampleStd(NDArray a) {
        return StatOps.std(a, 1);
    }

    /**
     * Computes the scalar standard deviation across all elements with custom delta degrees of freedom ({@code ddof}).
     *
     * @param a the input tensor
     * @param ddof delta degrees of freedom
     * @return the scalar standard deviation
     */
    public static double stdScalar(NDArray a, int ddof) {
        return StatOps.std(a, ddof);
    }

    /**
     * Computes the population standard deviation ({@code ddof = 0}) along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute standard deviation
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray std(NDArray a, int axis) {
        return StatOps.std(a, axis, 0, false, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with specified {@code ddof}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute standard deviation
     * @param ddof delta degrees of freedom
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray std(NDArray a, int axis, int ddof) {
        return StatOps.std(a, axis, ddof, false, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with specified {@code ddof}, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute standard deviation
     * @param ddof delta degrees of freedom
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray std(NDArray a, int axis, int ddof, boolean keepDims) {
        return StatOps.std(a, axis, ddof, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the standard deviation along the specified axis with custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute standard deviation
     * @param ddof delta degrees of freedom
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray std(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        return StatOps.std(a, axis, ddof, keepDims, arena);
    }

    // =========================================================================
    // PEAK-TO-PEAK (PTP: MAX - MIN)
    // =========================================================================

    /**
     * Computes the peak-to-peak range ({@code maximum - minimum}) across all elements in the array.
     *
     * @param a the input tensor
     * @return scalar range (max - min)
     */
    public static double ptp(NDArray a) {
        return StatOps.ptp(a);
    }

    /**
     * Computes the peak-to-peak range ({@code max - min}) along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute range
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray ptp(NDArray a, int axis) {
        return StatOps.ptp(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the peak-to-peak range along the specified axis, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute range
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray ptp(NDArray a, int axis, boolean keepDims) {
        return StatOps.ptp(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the peak-to-peak range along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute range
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray ptp(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.ptp(a, axis, keepDims, arena);
    }

    // =========================================================================
    // MEDIAN
    // =========================================================================

    /**
     * Computes the median across all elements in the array using an $O(N)$ Introselect algorithm.
     *
     * @param a the input tensor
     * @return the scalar median value
     */
    public static double median(NDArray a) {
        return StatOps.median(a);
    }

    /**
     * Computes the median along the specified axis using $O(N)$ Introselect.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute median
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray median(NDArray a, int axis) {
        return StatOps.median(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the median along the specified axis, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute median
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray median(NDArray a, int axis, boolean keepDims) {
        return StatOps.median(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the median along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute median
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray median(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.median(a, axis, keepDims, arena);
    }

    // =========================================================================
    // QUANTILE & PERCENTILE
    // =========================================================================

    /**
     * Computes the q-th quantile across all elements ({@code q in [0.0, 1.0]}) using linear interpolation.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @return scalar quantile value
     * @throws IllegalArgumentException if {@code q} is not within {@code [0.0, 1.0]}
     */
    public static double quantile(NDArray a, double q) {
        return StatOps.quantile(a, q, QuantileMethod.LINEAR);
    }

    /**
     * Computes the q-th quantile across all elements using the specified {@link QuantileMethod} interpolation.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param method interpolation method
     * @return scalar quantile value
     * @throws IllegalArgumentException if {@code q} is not within {@code [0.0, 1.0]}
     */
    public static double quantile(NDArray a, double q, QuantileMethod method) {
        return StatOps.quantile(a, q, method);
    }

    /**
     * Computes the q-th quantile along the specified axis using linear interpolation.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param axis dimension along which to compute quantile
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code q} is out of range or {@code axis} is invalid
     */
    public static NDArray quantile(NDArray a, double q, int axis) {
        return StatOps.quantile(a, q, axis, QuantileMethod.LINEAR, false, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis using linear interpolation, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param axis dimension along which to compute quantile
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code q} is out of range or {@code axis} is invalid
     */
    public static NDArray quantile(NDArray a, double q, int axis, boolean keepDims) {
        return StatOps.quantile(a, q, axis, QuantileMethod.LINEAR, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with the specified interpolation method.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param axis dimension along which to compute quantile
     * @param method interpolation method
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code q} is out of range or {@code axis} is invalid
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method) {
        return StatOps.quantile(a, q, axis, method, false, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with custom interpolation and keepDims.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param axis dimension along which to compute quantile
     * @param method interpolation method
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code q} is out of range or {@code axis} is invalid
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method, boolean keepDims) {
        return StatOps.quantile(a, q, axis, method, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the q-th quantile along the specified axis with custom interpolation, keepDims, and memory {@link Arena}.
     *
     * @param a the input tensor
     * @param q quantile value in range {@code [0.0, 1.0]}
     * @param axis dimension along which to compute quantile
     * @param method interpolation method
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code q} is out of range or {@code axis} is invalid
     */
    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method, boolean keepDims, Arena arena) {
        return StatOps.quantile(a, q, axis, method, keepDims, arena);
    }

    /**
     * Computes the p-th percentile of all elements ({@code p in [0.0, 100.0]}) with linear interpolation.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @return scalar percentile value
     * @throws IllegalArgumentException if {@code p} is not within {@code [0.0, 100.0]}
     */
    public static double percentile(NDArray a, double p) {
        return quantile(a, p / 100.0, QuantileMethod.LINEAR);
    }

    /**
     * Computes the p-th percentile of all elements with the specified interpolation method.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param method interpolation method
     * @return scalar percentile value
     * @throws IllegalArgumentException if {@code p} is not within {@code [0.0, 100.0]}
     */
    public static double percentile(NDArray a, double p, QuantileMethod method) {
        return quantile(a, p / 100.0, method);
    }

    /**
     * Computes the p-th percentile along the specified axis with linear interpolation.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param axis dimension along which to compute percentile
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code p} is out of range or {@code axis} is invalid
     */
    public static NDArray percentile(NDArray a, double p, int axis) {
        return quantile(a, p / 100.0, axis, QuantileMethod.LINEAR, false, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param axis dimension along which to compute percentile
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code p} is out of range or {@code axis} is invalid
     */
    public static NDArray percentile(NDArray a, double p, int axis, boolean keepDims) {
        return quantile(a, p / 100.0, axis, QuantileMethod.LINEAR, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with the specified interpolation method.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param axis dimension along which to compute percentile
     * @param method interpolation method
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code p} is out of range or {@code axis} is invalid
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method) {
        return quantile(a, p / 100.0, axis, method, false, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with custom interpolation and keepDims.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param axis dimension along which to compute percentile
     * @param method interpolation method
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code p} is out of range or {@code axis} is invalid
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method, boolean keepDims) {
        return quantile(a, p / 100.0, axis, method, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the p-th percentile along the specified axis with custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param p percentile value in range {@code [0.0, 100.0]}
     * @param axis dimension along which to compute percentile
     * @param method interpolation method
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code p} is out of range or {@code axis} is invalid
     */
    public static NDArray percentile(NDArray a, double p, int axis, QuantileMethod method, boolean keepDims, Arena arena) {
        return quantile(a, p / 100.0, axis, method, keepDims, arena);
    }

    // =========================================================================
    // INTERQUARTILE RANGE (IQR: Q75 - Q25)
    // =========================================================================

    /**
     * Computes the interquartile range (75th percentile - 25th percentile) across all elements.
     *
     * @param a the input tensor
     * @return scalar interquartile range
     */
    public static double iqr(NDArray a) {
        return StatOps.iqr(a);
    }

    /**
     * Computes the interquartile range (75th percentile - 25th percentile) along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute IQR
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray iqr(NDArray a, int axis) {
        return StatOps.iqr(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the interquartile range along the specified axis, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute IQR
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray iqr(NDArray a, int axis, boolean keepDims) {
        return StatOps.iqr(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the interquartile range along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute IQR
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray iqr(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.iqr(a, axis, keepDims, arena);
    }

    // =========================================================================
    // MODE
    // =========================================================================

    /**
     * Computes the statistical mode (most frequent value) across all elements in the array.
     *
     * <p>Ties are broken deterministically by selecting the smallest numerical value.</p>
     *
     * @param a the input tensor
     * @return scalar mode value
     */
    public static double mode(NDArray a) {
        return StatOps.mode(a);
    }

    /**
     * Computes the mode along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute mode
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray mode(NDArray a, int axis) {
        return StatOps.mode(a, axis, false, Arena.ofAuto());
    }

    /**
     * Computes the mode along the specified axis, optionally preserving reduced dimensions.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute mode
     * @param keepDims whether to retain reduced dimensions with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray mode(NDArray a, int axis, boolean keepDims) {
        return StatOps.mode(a, axis, keepDims, Arena.ofAuto());
    }

    /**
     * Computes the mode along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which to compute mode
     * @param keepDims whether to retain reduced dimensions with size 1
     * @param arena memory arena governing lifecycle of result
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray mode(NDArray a, int axis, boolean keepDims, Arena arena) {
        return StatOps.mode(a, axis, keepDims, arena);
    }

    // =========================================================================
    // COVARIANCE
    // =========================================================================

    /**
     * Computes the sample covariance between two 1D vectors with default sample degrees of freedom ({@code ddof = 1}).
     *
     * @param x the first 1D vector
     * @param y the second 1D vector
     * @return scalar sample covariance
     * @throws IllegalArgumentException if tensors are not 1D or their lengths differ
     */
    public static double cov(NDArray x, NDArray y) {
        return StatOps.cov(x, y, 1);
    }

    /**
     * Computes the covariance between two 1D vectors with custom delta degrees of freedom ({@code ddof}).
     *
     * @param x the first 1D vector
     * @param y the second 1D vector
     * @param ddof delta degrees of freedom
     * @return scalar covariance
     * @throws IllegalArgumentException if tensors are not 1D or their lengths differ
     */
    public static double cov(NDArray x, NDArray y, int ddof) {
        return StatOps.cov(x, y, ddof);
    }

    /**
     * Estimates the covariance matrix of a 2D matrix where rows represent variables and columns represent observations ({@code ddof = 1}).
     *
     * @param m 2D input matrix
     * @return square covariance matrix of shape {@code [N, N]} where {@code N = m.shape[0]}
     * @throws IllegalArgumentException if {@code m} is not a 2D matrix
     */
    public static NDArray cov(NDArray m) {
        return StatOps.cov(m, true, 1, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix of a 2D matrix with custom variable orientation ({@code ddof = 1}).
     *
     * @param m 2D input matrix
     * @param rowvar if true, rows represent variables; if false, columns represent variables
     * @return square covariance matrix
     * @throws IllegalArgumentException if {@code m} is not a 2D matrix
     */
    public static NDArray cov(NDArray m, boolean rowvar) {
        return StatOps.cov(m, rowvar, 1, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix of a 2D matrix with custom variable orientation and degrees of freedom.
     *
     * @param m 2D input matrix
     * @param rowvar if true, rows represent variables; if false, columns represent variables
     * @param ddof delta degrees of freedom
     * @return square covariance matrix
     * @throws IllegalArgumentException if {@code m} is not a 2D matrix
     */
    public static NDArray cov(NDArray m, boolean rowvar, int ddof) {
        return StatOps.cov(m, rowvar, ddof, Arena.ofAuto());
    }

    /**
     * Estimates the covariance matrix with custom variable orientation, degrees of freedom, and memory {@link Arena}.
     *
     * @param m 2D input matrix
     * @param rowvar if true, rows represent variables; if false, columns represent variables
     * @param ddof delta degrees of freedom
     * @param arena memory arena governing lifecycle of result
     * @return square covariance matrix
     * @throws IllegalArgumentException if {@code m} is not a 2D matrix
     */
    public static NDArray cov(NDArray m, boolean rowvar, int ddof, Arena arena) {
        return StatOps.cov(m, rowvar, ddof, arena);
    }

    // =========================================================================
    // CUMULATIVE SUM (CUMSUM)
    // =========================================================================

    /**
     * Computes the cumulative sum of elements over the flattened array.
     *
     * @param a the input tensor
     * @return 1D tensor containing the cumulative running sums
     */
    public static NDArray cumsum(NDArray a) {
        return StatOps.cumsum(a, Arena.ofAuto());
    }

    /**
     * Computes the cumulative sum of elements over the flattened array with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param arena memory arena governing lifecycle of result
     * @return 1D tensor containing the cumulative running sums
     */
    public static NDArray cumsum(NDArray a, Arena arena) {
        return StatOps.cumsum(a, arena);
    }

    /**
     * Computes the cumulative sum of elements along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which running sums are accumulated
     * @return tensor of identical shape with running sums along {@code axis}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray cumsum(NDArray a, int axis) {
        return StatOps.cumsum(a, axis, Arena.ofAuto());
    }

    /**
     * Computes the cumulative sum along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which running sums are accumulated
     * @param arena memory arena governing lifecycle of result
     * @return tensor of identical shape with running sums along {@code axis}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray cumsum(NDArray a, int axis, Arena arena) {
        return StatOps.cumsum(a, axis, arena);
    }

    // =========================================================================
    // CUMULATIVE PRODUCT (CUMPROD)
    // =========================================================================

    /**
     * Computes the cumulative product of elements over the flattened array.
     *
     * @param a the input tensor
     * @return 1D tensor containing the cumulative running products
     */
    public static NDArray cumprod(NDArray a) {
        return StatOps.cumprod(a, Arena.ofAuto());
    }

    /**
     * Computes the cumulative product of elements over the flattened array with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param arena memory arena governing lifecycle of result
     * @return 1D tensor containing the cumulative running products
     */
    public static NDArray cumprod(NDArray a, Arena arena) {
        return StatOps.cumprod(a, arena);
    }

    /**
     * Computes the cumulative product of elements along the specified axis.
     *
     * @param a the input tensor
     * @param axis dimension along which running products are accumulated
     * @return tensor of identical shape with running products along {@code axis}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray cumprod(NDArray a, int axis) {
        return StatOps.cumprod(a, axis, Arena.ofAuto());
    }

    /**
     * Computes the cumulative product along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param axis dimension along which running products are accumulated
     * @param arena memory arena governing lifecycle of result
     * @return tensor of identical shape with running products along {@code axis}
     * @throws IllegalArgumentException if {@code axis} is out of bounds
     */
    public static NDArray cumprod(NDArray a, int axis, Arena arena) {
        return StatOps.cumprod(a, axis, arena);
    }

    // =========================================================================
    // DISCRETE DIFFERENCE (DIFF)
    // =========================================================================

    /**
     * Computes the first discrete difference along the last axis ({@code out[i] = a[i+1] - a[i]}).
     *
     * @param a the input tensor
     * @return tensor containing discrete differences
     */
    public static NDArray diff(NDArray a) {
        return StatOps.diff(a, 1, -1, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the last axis by recursively calculating differences {@code n} times.
     *
     * @param a the input tensor
     * @param n the order of the difference (must be non-negative)
     * @return tensor containing n-th order differences
     * @throws IllegalArgumentException if {@code n < 0}
     */
    public static NDArray diff(NDArray a, int n) {
        return StatOps.diff(a, n, -1, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the specified axis.
     *
     * @param a the input tensor
     * @param n the order of the difference
     * @param axis dimension along which differences are calculated
     * @return tensor containing differences with {@code axis} dimension reduced by {@code n}
     * @throws IllegalArgumentException if {@code n < 0} or {@code axis} is out of bounds
     */
    public static NDArray diff(NDArray a, int n, int axis) {
        return StatOps.diff(a, n, axis, Arena.ofAuto());
    }

    /**
     * Computes the n-th discrete difference along the specified axis with a custom memory {@link Arena}.
     *
     * @param a the input tensor
     * @param n the order of the difference
     * @param axis dimension along which differences are calculated
     * @param arena memory arena governing lifecycle of result
     * @return tensor containing differences with {@code axis} dimension reduced by {@code n}
     * @throws IllegalArgumentException if {@code n < 0} or {@code axis} is out of bounds
     */
    public static NDArray diff(NDArray a, int n, int axis, Arena arena) {
        return StatOps.diff(a, n, axis, arena);
    }
}
