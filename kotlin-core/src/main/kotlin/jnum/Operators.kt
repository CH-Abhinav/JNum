@file:Suppress("NOTHING_TO_INLINE", "EXTENSION_SHADOWED_BY_MEMBER")

package jnum

/**
 * High-performance operator overloads for JNum NDArray.
 * All operators are inlined for zero call-frame overhead.
 * Primitive overloads prevent JVM autoboxing.
 */

// =============================================================================
// Array - Array Binary Operators
// =============================================================================

inline operator fun NDArray.plus(other: NDArray): NDArray = this.add(other)
inline operator fun NDArray.minus(other: NDArray): NDArray = this.sub(other)
inline operator fun NDArray.times(other: NDArray): NDArray = this.mul(other)
inline operator fun NDArray.div(other: NDArray): NDArray = this.div(other)

// =============================================================================
// Unary Operators
// =============================================================================

inline operator fun NDArray.unaryMinus(): NDArray = this.mul(-1.0)
inline operator fun NDArray.unaryPlus(): NDArray = this
inline operator fun NDArray.not(): NDArray = this.not()

// =============================================================================
// Boolean Infix Operators (and, or, xor)
// =============================================================================

inline infix fun NDArray.and(other: NDArray): NDArray = this.and(other)
inline infix fun NDArray.or(other: NDArray): NDArray = this.or(other)
inline infix fun NDArray.xor(other: NDArray): NDArray = this.xor(other)

// =============================================================================
// Array - Scalar Binary Operators (Right-Hand Scalar)
// =============================================================================

// Double
inline operator fun NDArray.plus(scalar: Double): NDArray = this.add(scalar)
inline operator fun NDArray.minus(scalar: Double): NDArray = this.sub(scalar)
inline operator fun NDArray.times(scalar: Double): NDArray = this.mul(scalar)
inline operator fun NDArray.div(scalar: Double): NDArray = this.div(scalar)

// Float
inline operator fun NDArray.plus(scalar: Float): NDArray = this.add(scalar)
inline operator fun NDArray.minus(scalar: Float): NDArray = this.sub(scalar)
inline operator fun NDArray.times(scalar: Float): NDArray = this.mul(scalar)
inline operator fun NDArray.div(scalar: Float): NDArray = this.div(scalar)

// Int
inline operator fun NDArray.plus(scalar: Int): NDArray = this.add(scalar)
inline operator fun NDArray.minus(scalar: Int): NDArray = this.sub(scalar)
inline operator fun NDArray.times(scalar: Int): NDArray = this.mul(scalar)
inline operator fun NDArray.div(scalar: Int): NDArray = this.div(scalar)

// =============================================================================
// Scalar - Array Binary Operators (Left-Hand Scalar: e.g. 2.0 + a, 10.0 - a)
// =============================================================================

// Double
inline operator fun Double.plus(arr: NDArray): NDArray = arr.add(this)
inline operator fun Double.minus(arr: NDArray): NDArray = arr.mul(-1.0).add(this)
inline operator fun Double.times(arr: NDArray): NDArray = arr.mul(this)

// Float
inline operator fun Float.plus(arr: NDArray): NDArray = arr.add(this)
inline operator fun Float.minus(arr: NDArray): NDArray = arr.mul(-1.0f).add(this)
inline operator fun Float.times(arr: NDArray): NDArray = arr.mul(this)

// Int
inline operator fun Int.plus(arr: NDArray): NDArray = arr.add(this)
inline operator fun Int.minus(arr: NDArray): NDArray = arr.mul(-1).add(this)
inline operator fun Int.times(arr: NDArray): NDArray = arr.mul(this)
