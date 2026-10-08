@file:Suppress("NOTHING_TO_INLINE")

package jnum

/**
 * In-place augmented assignment operators.
 * Directly invokes SIMD-accelerated addi, subi, muli, divi.
 * Modifies the underlying memory buffer in-place with zero reallocation.
 */

// =============================================================================
// In-Place Array Assignments
// =============================================================================

inline operator fun NDArray.plusAssign(other: NDArray) { this.addi(other) }
inline operator fun NDArray.minusAssign(other: NDArray) { this.subi(other) }
inline operator fun NDArray.timesAssign(other: NDArray) { this.muli(other) }
inline operator fun NDArray.divAssign(other: NDArray) { this.divi(other) }

// =============================================================================
// In-Place Scalar Assignments (Double, Float, Int)
// =============================================================================

// Double
inline operator fun NDArray.plusAssign(scalar: Double) { this.addi(scalar) }
inline operator fun NDArray.minusAssign(scalar: Double) { this.subi(scalar) }
inline operator fun NDArray.timesAssign(scalar: Double) { this.muli(scalar) }
inline operator fun NDArray.divAssign(scalar: Double) { this.divi(scalar) }

// Float
inline operator fun NDArray.plusAssign(scalar: Float) { this.addi(scalar) }
inline operator fun NDArray.minusAssign(scalar: Float) { this.subi(scalar) }
inline operator fun NDArray.timesAssign(scalar: Float) { this.muli(scalar) }
inline operator fun NDArray.divAssign(scalar: Float) { this.divi(scalar) }

// Int
inline operator fun NDArray.plusAssign(scalar: Int) { this.addi(scalar) }
inline operator fun NDArray.minusAssign(scalar: Int) { this.subi(scalar) }
inline operator fun NDArray.timesAssign(scalar: Int) { this.muli(scalar) }
inline operator fun NDArray.divAssign(scalar: Int) { this.divi(scalar) }
