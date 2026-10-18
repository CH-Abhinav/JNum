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
 * Infix operations for linear algebra and vector products in Kotlin.
 *
 * <p>Enables natural mathematical notation: {@code val c = a matmul b}, {@code val dotProduct = x dot y}.</p>
 */

/** Performs infix matrix multiplication or batched matrix multiplication (`a matmul b`). */
inline infix fun NDArray.matmul(other: NDArray): NDArray = this.matmul(other)

/** Performs infix 1D vector dot product (`a dot b`). */
inline infix fun NDArray.dot(other: NDArray): Double = this.dot(other)

