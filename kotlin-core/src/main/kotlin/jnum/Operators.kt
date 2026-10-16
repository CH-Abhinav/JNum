@file:Suppress("NOTHING_TO_INLINE", "EXTENSION_SHADOWED_BY_MEMBER")

package jnum

/**
 * High-performance operator overloads for JNum [NDArray] in Kotlin.
 *
 * <p>All operators are marked {@code inline} to ensure zero call-frame overhead.
 * Primitive overloads prevent JVM autoboxing.</p>
 */

// =============================================================================
// Array - Array Binary Operators
// =============================================================================

/** Element-wise addition (`a + b`) with broadcasting support. */
inline operator fun NDArray.plus(other: NDArray): NDArray = this.add(other)

/** Element-wise subtraction (`a - b`) with broadcasting support. */
inline operator fun NDArray.minus(other: NDArray): NDArray = this.sub(other)

/** Element-wise multiplication (`a * b`) with broadcasting support. */
inline operator fun NDArray.times(other: NDArray): NDArray = this.mul(other)

/** Element-wise division (`a / b`) with broadcasting support. */
inline operator fun NDArray.div(other: NDArray): NDArray = this.div(other)

// =============================================================================
// Unary Operators
// =============================================================================

/** Element-wise negation (`-a`). */
inline operator fun NDArray.unaryMinus(): NDArray = this.mul(-1.0)

/** Unary plus identity (`+a`). */
inline operator fun NDArray.unaryPlus(): NDArray = this

/** Element-wise logical inversion (`!a`). */
inline operator fun NDArray.not(): NDArray = this.not()

// =============================================================================
// Boolean Infix Operators (and, or, xor)
// =============================================================================

/** Infix logical AND between boolean tensors (`a and b`). */
inline infix fun NDArray.and(other: NDArray): NDArray = this.and(other)

/** Infix logical OR between boolean tensors (`a or b`). */
inline infix fun NDArray.or(other: NDArray): NDArray = this.or(other)

/** Infix logical XOR between boolean tensors (`a xor b`). */
inline infix fun NDArray.xor(other: NDArray): NDArray = this.xor(other)

// =============================================================================
// Array - Scalar Binary Operators (Right-Hand Scalar)
// =============================================================================

/** Adds a [Double] scalar to every element (`a + 2.0`). */
inline operator fun NDArray.plus(scalar: Double): NDArray = this.add(scalar)

/** Subtracts a [Double] scalar from every element (`a - 2.0`). */
inline operator fun NDArray.minus(scalar: Double): NDArray = this.sub(scalar)

/** Multiplies every element by a [Double] scalar (`a * 2.0`). */
inline operator fun NDArray.times(scalar: Double): NDArray = this.mul(scalar)

/** Divides every element by a [Double] scalar (`a / 2.0`). */
inline operator fun NDArray.div(scalar: Double): NDArray = this.div(scalar)

/** Adds a [Float] scalar to every element (`a + 2.0f`). */
inline operator fun NDArray.plus(scalar: Float): NDArray = this.add(scalar)

/** Subtracts a [Float] scalar from every element (`a - 2.0f`). */
inline operator fun NDArray.minus(scalar: Float): NDArray = this.sub(scalar)

/** Multiplies every element by a [Float] scalar (`a * 2.0f`). */
inline operator fun NDArray.times(scalar: Float): NDArray = this.mul(scalar)

/** Divides every element by a [Float] scalar (`a / 2.0f`). */
inline operator fun NDArray.div(scalar: Float): NDArray = this.div(scalar)

/** Adds an [Int] scalar to every element (`a + 2`). */
inline operator fun NDArray.plus(scalar: Int): NDArray = this.add(scalar)

/** Subtracts an [Int] scalar from every element (`a - 2`). */
inline operator fun NDArray.minus(scalar: Int): NDArray = this.sub(scalar)

/** Multiplies every element by an [Int] scalar (`a * 2`). */
inline operator fun NDArray.times(scalar: Int): NDArray = this.mul(scalar)

/** Divides every element by an [Int] scalar (`a / 2`). */
inline operator fun NDArray.div(scalar: Int): NDArray = this.div(scalar)

// =============================================================================
// Scalar - Array Binary Operators (Left-Hand Scalar: e.g. 2.0 + a, 10.0 - a)
// =============================================================================

/** Adds a [Double] scalar to an array (`2.0 + a`). */
inline operator fun Double.plus(arr: NDArray): NDArray = arr.add(this)

/** Subtracts an array from a [Double] scalar (`10.0 - a`). */
inline operator fun Double.minus(arr: NDArray): NDArray = arr.mul(-1.0).add(this)

/** Multiplies an array by a [Double] scalar (`2.0 * a`). */
inline operator fun Double.times(arr: NDArray): NDArray = arr.mul(this)

/** Adds a [Float] scalar to an array (`2.0f + a`). */
inline operator fun Float.plus(arr: NDArray): NDArray = arr.add(this)

/** Subtracts an array from a [Float] scalar (`10.0f - a`). */
inline operator fun Float.minus(arr: NDArray): NDArray = arr.mul(-1.0f).add(this)

/** Multiplies an array by a [Float] scalar (`2.0f * a`). */
inline operator fun Float.times(arr: NDArray): NDArray = arr.mul(this)

/** Adds an [Int] scalar to an array (`2 + a`). */
inline operator fun Int.plus(arr: NDArray): NDArray = arr.add(this)

/** Subtracts an array from an [Int] scalar (`10 - a`). */
inline operator fun Int.minus(arr: NDArray): NDArray = arr.mul(-1).add(this)

/** Multiplies an array by an [Int] scalar (`2 * a`). */
inline operator fun Int.times(arr: NDArray): NDArray = arr.mul(this)

