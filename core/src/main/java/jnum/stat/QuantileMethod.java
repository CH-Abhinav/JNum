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

