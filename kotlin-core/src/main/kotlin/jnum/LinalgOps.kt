@file:Suppress("NOTHING_TO_INLINE")

package jnum

/**
 * Infix functions for linear algebra and dot products.
 * Enables clean notation: a matmul b, a dot b.
 */

inline infix fun NDArray.matmul(other: NDArray): NDArray = this.matmul(other)
inline infix fun NDArray.dot(other: NDArray): Double = this.dot(other)
