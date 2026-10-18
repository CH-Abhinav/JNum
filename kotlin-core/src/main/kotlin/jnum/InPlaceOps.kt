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
 * In-place augmented assignment operators for Kotlin (`+=`, `-=`, `*=`, `/=`).
 *
 * <p>Directly invokes SIMD-accelerated {@code addi}, {@code subi}, {@code muli}, and {@code divi},
 * modifying the underlying native memory buffer in-place with zero heap reallocation.</p>
 */

// =============================================================================
// In-Place Array Assignments
// =============================================================================

/** In-place element-wise addition: `this += other`. Modifies this array's memory directly. */
inline operator fun NDArray.plusAssign(other: NDArray) { this.addi(other) }

/** In-place element-wise subtraction: `this -= other`. Modifies this array's memory directly. */
inline operator fun NDArray.minusAssign(other: NDArray) { this.subi(other) }

/** In-place element-wise multiplication: `this *= other`. Modifies this array's memory directly. */
inline operator fun NDArray.timesAssign(other: NDArray) { this.muli(other) }

/** In-place element-wise division: `this /= other`. Modifies this array's memory directly. */
inline operator fun NDArray.divAssign(other: NDArray) { this.divi(other) }

// =============================================================================
// In-Place Scalar Assignments (Double, Float, Int)
// =============================================================================

/** In-place scalar addition: `this += scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.plusAssign(scalar: Double) { this.addi(scalar) }

/** In-place scalar subtraction: `this -= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.minusAssign(scalar: Double) { this.subi(scalar) }

/** In-place scalar multiplication: `this *= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.timesAssign(scalar: Double) { this.muli(scalar) }

/** In-place scalar division: `this /= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.divAssign(scalar: Double) { this.divi(scalar) }

/** In-place scalar addition: `this += scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.plusAssign(scalar: Float) { this.addi(scalar) }

/** In-place scalar subtraction: `this -= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.minusAssign(scalar: Float) { this.subi(scalar) }

/** In-place scalar multiplication: `this *= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.timesAssign(scalar: Float) { this.muli(scalar) }

/** In-place scalar division: `this /= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.divAssign(scalar: Float) { this.divi(scalar) }

/** In-place scalar addition: `this += scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.plusAssign(scalar: Int) { this.addi(scalar) }

/** In-place scalar subtraction: `this -= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.minusAssign(scalar: Int) { this.subi(scalar) }

/** In-place scalar multiplication: `this *= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.timesAssign(scalar: Int) { this.muli(scalar) }

/** In-place scalar division: `this /= scalar`. Modifies this array's memory directly. */
inline operator fun NDArray.divAssign(scalar: Int) { this.divi(scalar) }

