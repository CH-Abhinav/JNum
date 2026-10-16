/**
 * High-performance descriptive and summary statistics for tensors.
 *
 * <p>Provides full NumPy-compatible statistical reductions and aggregations:
 * <ul>
 *   <li><b>Central Tendency:</b> {@link jnum.stat.Stats#mean}, {@link jnum.stat.Stats#average} (weighted),
 *       {@link jnum.stat.Stats#median} ($O(N)$ Introselect), and {@link jnum.stat.Stats#mode}.</li>
 *   <li><b>Dispersion:</b> {@link jnum.stat.Stats#var} and {@link jnum.stat.Stats#std} with configurable delta degrees
 *       of freedom ({@code ddof}), {@link jnum.stat.Stats#ptp} (peak-to-peak range), and {@link jnum.stat.Stats#iqr} (interquartile range).</li>
 *   <li><b>Order Statistics:</b> {@link jnum.stat.Stats#quantile} and {@link jnum.stat.Stats#percentile} supporting 5 standard
 *       interpolation strategies via {@link jnum.stat.QuantileMethod}.</li>
 *   <li><b>Cumulative &amp; Difference Operations:</b> {@link jnum.stat.Stats#cumsum}, {@link jnum.stat.Stats#cumprod},
 *       and n-th order discrete differences {@link jnum.stat.Stats#diff}.</li>
 *   <li><b>Multivariate Statistics:</b> Pairwise and 2D matrix sample covariance {@link jnum.stat.Stats#cov}.</li>
 * </ul>
 * </p>
 */
package jnum.stat;