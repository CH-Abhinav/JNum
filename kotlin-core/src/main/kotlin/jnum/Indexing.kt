@file:Suppress("NOTHING_TO_INLINE")

package jnum

/**
 * High-performance indexing, mutation, and Python-style slicing.
 * Zero string parsing, zero heap allocations for element access.
 */

/** Full-axis slice token representing Python's ':' */
val all: Slice = Slice.all()
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

// =============================================================================
// Element Access (Direct Flat Offset, Zero Allocations)
// =============================================================================

inline operator fun NDArray.get(x: Int): Double = this.getDouble(x.toLong())
inline operator fun NDArray.get(x: Int, y: Int): Double = this.getDouble(x.toLong(), y.toLong())
inline operator fun NDArray.get(x: Int, y: Int, z: Int): Double = this.getDouble(x.toLong(), y.toLong(), z.toLong())

operator fun NDArray.get(vararg indices: Int): Double =
    this.get(*indices.map { it.toLong() }.toLongArray())

// =============================================================================
// Element Mutation (a[i] = v, a[i, j] = v, a[i, j, k] = v)
// In Kotlin, the assigned value is always the last argument.
// =============================================================================

// 1D Set
inline operator fun NDArray.set(x: Int, value: Double) = this.setDouble(value, x.toLong())
inline operator fun NDArray.set(x: Int, value: Float) = this.setFloat(value, x.toLong())
inline operator fun NDArray.set(x: Int, value: Int) = this.setInt(value, x.toLong())

// 2D Set
inline operator fun NDArray.set(x: Int, y: Int, value: Double) = this.setDouble(value, x.toLong(), y.toLong())
inline operator fun NDArray.set(x: Int, y: Int, value: Float) = this.setFloat(value, x.toLong(), y.toLong())
inline operator fun NDArray.set(x: Int, y: Int, value: Int) = this.setInt(value, x.toLong(), y.toLong())

// 3D Set
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Double) = this.setDouble(value, x.toLong(), y.toLong(), z.toLong())
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Float) = this.setFloat(value, x.toLong(), y.toLong(), z.toLong())
inline operator fun NDArray.set(x: Int, y: Int, z: Int, value: Int) = this.setInt(value, x.toLong(), y.toLong(), z.toLong())

// =============================================================================
// Python-Style Slicing (Ranges, Progressions, Steps, and Full-Axis Token '_')
// =============================================================================

inline operator fun NDArray.get(slice: Slice): NDArray = this.slice(slice)
inline operator fun NDArray.get(s1: Slice, s2: Slice): NDArray = this.slice(s1, s2)
inline operator fun NDArray.get(s1: Slice, s2: Slice, s3: Slice): NDArray = this.slice(s1, s2, s3)

inline operator fun NDArray.get(prog: IntProgression): NDArray =
    this.slice(progressionToSlice(prog))

inline operator fun NDArray.get(p1: IntProgression, p2: IntProgression): NDArray =
    this.slice(progressionToSlice(p1), progressionToSlice(p2))

inline operator fun NDArray.get(p1: IntProgression, p2: IntProgression, p3: IntProgression): NDArray =
    this.slice(progressionToSlice(p1), progressionToSlice(p2), progressionToSlice(p3))

inline operator fun NDArray.get(row: Int, p2: IntProgression): NDArray =
    this.slice(Slice.range(row.toLong(), row.toLong() + 1), progressionToSlice(p2))

inline operator fun NDArray.get(p1: IntProgression, col: Int): NDArray =
    this.slice(progressionToSlice(p1), Slice.range(col.toLong(), col.toLong() + 1))

inline operator fun NDArray.get(slice: Slice, prog: IntProgression): NDArray =
    this.slice(slice, progressionToSlice(prog))

inline operator fun NDArray.get(prog: IntProgression, slice: Slice): NDArray =
    this.slice(progressionToSlice(prog), slice)
