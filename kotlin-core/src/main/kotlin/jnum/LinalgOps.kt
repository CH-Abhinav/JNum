@file:Suppress("NOTHING_TO_INLINE")

package jnum

/**
 * Infix operations for linear algebra and vector products in Kotlin.
 *
 * <p>Enables natural mathematical notation: {@code val c = a matmul b}, {@code val dotProduct = x dot y}.</p>
 */

/** Performs infix matrix multiplication or batched matrix multiplication (`a matmul b`). */
inline infix fun NDArray.matmul(other: NDArray): NDArray = this.matmul(other)

/** Performs infix 1D vector dot product (`a dot b`). */
inline infix fun NDArray.dot(other: NDArray): Double = this.dot(other)

