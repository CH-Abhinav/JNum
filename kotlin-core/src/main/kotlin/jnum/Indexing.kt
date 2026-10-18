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
@file:Suppress("NOTHING_TO_INLINE")

package jnum

/**
 * High-performance indexing, mutation, and Python-style multidimensional slicing for Kotlin.
 *
 * <p>All indexers and slice operators are marked {@code inline} to ensure zero call-frame overhead
 * and zero heap allocation during element access.</p>
 */

/** Full-axis slice token representing Python's {@code :}. Example: {@code arr[all, 0..4]} */
val all: Slice = Slice.all()

/** Short-hand full-axis slice token representing Python's {@code :}. Example: {@code arr[`_`, 0..4]} */
val `_`: Slice = Slice.all()

@PublishedApi
internal fun progressionToSlice(prog: IntProgression): Slice {
    val step = prog.step.toLong()
    val start = prog.first.toLong()
    val stop = if (step > 0) {
        prog.last.toLong() + 1
    } else {
        if (prog.last <= 0) Slice.UNBOUNDED_STOP else prog.last.toLong() - 1
    }
    return Slice.range(start, stop, step)
}

@PublishedApi
internal fun singleIndexToSlice(idx: Int): Slice =
    if (idx == -1) Slice.from(-1) else Slice.range(idx.toLong(), idx.toLong() + 1)

// =============================================================================
// Element Access (Direct Flat Offset, Zero Allocations)
// =============================================================================

/** Retrieves the element at coordinate [x] in a 1D tensor without heap allocation. */
inline operator fun NDArray.get(x: Int): Double = this.getDouble(x.toLong())

/** Retrieves the element at coordinates [x, y] in a 2D tensor without heap allocation. */
inline operator fun NDArray.get(x: Int, y: Int): Double = this.getDouble(x.toLong(), y.toLong())

/** Retrieves the element at coordinates [x, y, z] in a 3D tensor without heap allocation. */
inline operator fun NDArray.get(x: Int, y: Int, z: Int): Double = this.getDouble(x.toLong(), y.toLong(), z.toLong())

/** Retrieves the element at N-dimensional [indices]. */
operator fun NDArray.get(vararg indices: Int): Double =
    this.get(*LongArray(indices.size) { indices[it].toLong() })

// =============================================================================
// Element Mutation (a[i] = v, a[i, j] = v, a[i, j, k] = v)
// =============================================================================

/** Writes a [Double] value to coordinate [x] in a 1D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, value: Double) = this.setDouble(value, x.toLong())

/** Writes a [Float] value to coordinate [x] in a 1D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, value: Float) = this.setFloat(value, x.toLong())

/** Writes an [Int] value to coordinate [x] in a 1D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, value: Int) = this.setInt(value, x.toLong())

/** Writes a [Double] value to coordinates [x, y] in a 2D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, value: Double) = this.setDouble(value, x.toLong(), y.toLong())

/** Writes a [Float] value to coordinates [x, y] in a 2D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, value: Float) = this.setFloat(value, x.toLong(), y.toLong())

/** Writes an [Int] value to coordinates [x, y] in a 2D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, value: Int) = this.setInt(value, x.toLong(), y.toLong())

/** Writes a [Double] value to coordinates [x, y, z] in a 3D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Double) = this.setDouble(value, x.toLong(), y.toLong(), z.toLong())

/** Writes a [Float] value to coordinates [x, y, z] in a 3D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Float) = this.setFloat(value, x.toLong(), y.toLong(), z.toLong())

/** Writes an [Int] value to coordinates [x, y, z] in a 3D tensor without heap allocation. */
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Int) = this.setInt(value, x.toLong(), y.toLong(), z.toLong())

// =============================================================================
// Python-Style Slicing (Ranges, Progressions, Steps, and Full-Axis Token '_')
// =============================================================================

/** Slices this tensor along axis 0 using a [Slice] descriptor. */
inline operator fun NDArray.get(slice: Slice): NDArray = this.slice(slice)

/** Slices this tensor along axes 0 and 1 using [Slice] descriptors. */
inline operator fun NDArray.get(s1: Slice, s2: Slice): NDArray = this.slice(s1, s2)

/** Slices this tensor along axes 0, 1, and 2 using [Slice] descriptors. */
inline operator fun NDArray.get(s1: Slice, s2: Slice, s3: Slice): NDArray = this.slice(s1, s2, s3)

/** Slices this tensor along axis 0 using a Kotlin [IntProgression] (e.g. `arr[0..10 step 2]`). */
inline operator fun NDArray.get(prog: IntProgression): NDArray =
    this.slice(progressionToSlice(prog))

/** Slices this tensor along 2 axes using Kotlin [IntProgression] ranges. */
inline operator fun NDArray.get(p1: IntProgression, p2: IntProgression): NDArray =
    this.slice(progressionToSlice(p1), progressionToSlice(p2))

/** Slices this tensor along 3 axes using Kotlin [IntProgression] ranges. */
inline operator fun NDArray.get(p1: IntProgression, p2: IntProgression, p3: IntProgression): NDArray =
    this.slice(progressionToSlice(p1), progressionToSlice(p2), progressionToSlice(p3))

/** Extracts rows and sliced columns using a fixed row index and a column progression. */
inline operator fun NDArray.get(row: Int, p2: IntProgression): NDArray =
    this.slice(singleIndexToSlice(row), progressionToSlice(p2))

/** Extracts sliced rows and a fixed column. */
inline operator fun NDArray.get(p1: IntProgression, col: Int): NDArray =
    this.slice(progressionToSlice(p1), singleIndexToSlice(col))

/** Extracts a fixed row and sliced columns using a [Slice]. */
inline operator fun NDArray.get(row: Int, slice: Slice): NDArray =
    this.slice(singleIndexToSlice(row), slice)

/** Extracts sliced rows using a [Slice] and a fixed column. */
inline operator fun NDArray.get(slice: Slice, col: Int): NDArray =
    this.slice(slice, singleIndexToSlice(col))

/** Slices using a [Slice] on axis 0 and an [IntProgression] on axis 1. */
inline operator fun NDArray.get(slice: Slice, prog: IntProgression): NDArray =
    this.slice(slice, progressionToSlice(prog))

/** Slices using an [IntProgression] on axis 0 and a [Slice] on axis 1. */
inline operator fun NDArray.get(prog: IntProgression, slice: Slice): NDArray =
    this.slice(progressionToSlice(prog), slice)

// =============================================================================
// 3D Mixed Slices (Slice and IntProgression combinations)
// =============================================================================

/** Slices a 3D tensor using two progressions and one slice (e.g. `image[340..740, 760..1160, all]`). */
inline operator fun NDArray.get(p1: IntProgression, p2: IntProgression, s3: Slice): NDArray =
    this.slice(progressionToSlice(p1), progressionToSlice(p2), s3)

/** Slices a 3D tensor using a slice and two progressions. */
inline operator fun NDArray.get(s1: Slice, p2: IntProgression, p3: IntProgression): NDArray =
    this.slice(s1, progressionToSlice(p2), progressionToSlice(p3))

/** Slices a 3D tensor using a progression, a slice, and a progression. */
inline operator fun NDArray.get(p1: IntProgression, s2: Slice, p3: IntProgression): NDArray =
    this.slice(progressionToSlice(p1), s2, progressionToSlice(p3))

/** Slices a 3D tensor using two slices and a progression. */
inline operator fun NDArray.get(s1: Slice, s2: Slice, p3: IntProgression): NDArray =
    this.slice(s1, s2, progressionToSlice(p3))

/** Slices a 3D tensor using a slice, a progression, and a slice. */
inline operator fun NDArray.get(s1: Slice, p2: IntProgression, s3: Slice): NDArray =
    this.slice(s1, progressionToSlice(p2), s3)

/** Slices a 3D tensor using a progression and two slices. */
inline operator fun NDArray.get(p1: IntProgression, s2: Slice, s3: Slice): NDArray =
    this.slice(progressionToSlice(p1), s2, s3)

