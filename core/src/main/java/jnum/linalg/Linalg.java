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
package jnum.linalg;

import java.lang.foreign.Arena;
import jnum.NDArray;
import jnum.internal.ops.LinalgOps;
import jnum.internal.ops.ReduceOps;

/**
 * High-performance linear algebra operations, matrix decompositions, solvers, and norms for JNum.
 *
 * <p>Provides static functional entry points matching NumPy's {@code numpy.linalg} module.
 * Hardware-accelerated using Project Panama SIMD vector kernels, cache-blocked matrix packing,
 * and high-performance algorithms without JNI overhead.</p>
 */
public final class Linalg {

    private Linalg() {
        throw new AssertionError("Linalg facade cannot be instantiated.");
    }

    // =========================================================================
    // MATRIX MULTIPLICATION & VECTOR METRICS
    // =========================================================================

    /**
     * Performs matrix multiplication ({@code a @ b}) or batched matrix multiplication.
     *
     * <p>Hardware-accelerated using tile-based SIMD vector kernels. Adheres to standard NumPy
     * batched matmul rules where leading dimensions are broadcasted.</p>
     *
     * @param a the left-hand matrix or batched tensor operand
     * @param b the right-hand matrix or batched tensor operand
     * @return a newly allocated {@link NDArray} containing the matrix product
     * @throws IllegalArgumentException if inner matrix dimensions do not match or ranks are invalid
     */
    public static NDArray matmul(NDArray a, NDArray b) {
        return LinalgOps.matmul(a, b);
    }

    /**
     * Performs matrix multiplication, writing results into pre-allocated {@code resArray}.
     *
     * <p>Zero-allocation variant for high-throughput linear algebra loops.</p>
     *
     * @param a the left-hand matrix operand
     * @param b the right-hand matrix operand
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the matrix product
     * @throws IllegalArgumentException if dimensions mismatch or buffer is non-contiguous
     */
    public static NDArray matmul(NDArray a, NDArray b, NDArray resArray) {
        return LinalgOps.matmul(a, b, resArray);
    }

    /**
     * Computes the dot product (scalar inner product) of two 1D vectors.
     *
     * @param a first 1D vector operand
     * @param b second 1D vector operand
     * @return the scalar dot product promoted to {@code double}
     * @throws IllegalArgumentException if either tensor is not 1D or their lengths differ
     */
    public static double dot(NDArray a, NDArray b) {
        return ReduceOps.dot(a, b);
    }

    /**
     * Computes the cosine similarity between two 1D vectors:
     * {@code (a . b) / (||a||_2 * ||b||_2)}.
     *
     * @param a first 1D vector operand
     * @param b second 1D vector operand
     * @return scalar cosine similarity value in range {@code [-1.0, 1.0]}
     * @throws IllegalArgumentException if either tensor is not 1D or their lengths differ
     */
    public static double cosineSimilarity(NDArray a, NDArray b) {
        return LinalgOps.cosineSimilarity(a, b);
    }

    // =========================================================================
    // MATRIX PROPERTIES & METRICS
    // =========================================================================

    /**
     * Computes the trace (sum of diagonal elements) of a 2D square matrix.
     *
     * @param a the 2D input matrix
     * @return the sum of elements along the main diagonal
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static double trace(NDArray a) {
        return LinalgOps.trace(a);
    }

    /**
     * Computes the trace along an offset diagonal of a 2D matrix.
     *
     * @param a the 2D input matrix
     * @param offset diagonal offset (0 = main diagonal, &gt;0 = above, &lt;0 = below)
     * @return the sum of elements along the specified diagonal
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static double trace(NDArray a, int offset) {
        return LinalgOps.trace(a, offset);
    }

    /**
     * Computes the Euclidean 2-norm for vectors or Frobenius norm for matrices.
     *
     * @param a the input tensor
     * @return the scalar norm value
     */
    public static double norm(NDArray a) {
        return LinalgOps.norm(a);
    }

    /**
     * Computes the matrix or vector norm of the specified order {@code ord}.
     *
     * @param a the input tensor
     * @param ord norm order (e.g. 1, 2)
     * @return the scalar norm value
     */
    public static double norm(NDArray a, int ord) {
        return LinalgOps.norm(a, ord);
    }

    /**
     * Computes the determinant of a 2D square matrix.
     *
     * @param a the 2D square matrix
     * @return the scalar determinant value
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static double det(NDArray a) {
        return LinalgOps.det(a);
    }

    /**
     * Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
     *
     * @param a the 2D square matrix
     * @return {@link SlogdetResult} containing sign and log-absolute-determinant
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static SlogdetResult slogdet(NDArray a) {
        return LinalgOps.slogdet(a);
    }

    /**
     * Computes the numerical rank of a matrix using SVD with default singular value tolerance.
     *
     * @param a the 2D input matrix
     * @return the numerical rank (count of singular values greater than tolerance)
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static int matrixRank(NDArray a) {
        return LinalgOps.matrixRank(a);
    }

    /**
     * Computes the numerical rank of a matrix using SVD with the specified tolerance {@code tol}.
     *
     * @param a the 2D input matrix
     * @param tol singular value threshold below which values are treated as zero
     * @return the numerical rank
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static int matrixRank(NDArray a, double tol) {
        return LinalgOps.matrixRank(a, tol);
    }

    /**
     * Computes the numerical rank of a matrix using SVD with the specified tolerance in {@code arena}.
     *
     * @param a the 2D input matrix
     * @param tol singular value threshold below which values are treated as zero
     * @param arena memory arena governing temporary allocations
     * @return the numerical rank
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static int matrixRank(NDArray a, double tol, Arena arena) {
        return LinalgOps.matrixRank(a, tol, arena);
    }

    /**
     * Computes the 2-norm condition number of a matrix using SVD.
     *
     * @param a the 2D input matrix
     * @return condition number (ratio of largest to smallest singular value)
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static double cond(NDArray a) {
        return LinalgOps.cond(a);
    }

    /**
     * Computes the 2-norm condition number of a matrix using SVD in {@code arena}.
     *
     * @param a the 2D input matrix
     * @param arena memory arena governing temporary allocations
     * @return condition number
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static double cond(NDArray a, Arena arena) {
        return LinalgOps.cond(a, arena);
    }

    // =========================================================================
    // INVERSES & SOLVERS
    // =========================================================================

    /**
     * Computes the multiplicative inverse of a square non-singular matrix ({@code A^-1}).
     *
     * @param a the 2D square matrix to invert
     * @return a newly allocated {@link NDArray} containing {@code A^-1}
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     * @throws ArithmeticException if the matrix is singular (non-invertible)
     */
    public static NDArray inv(NDArray a) {
        return LinalgOps.inv(a);
    }

    /**
     * Computes the multiplicative inverse of a square non-singular matrix, allocating in {@code arena}.
     *
     * @param a the 2D square matrix to invert
     * @param arena memory arena governing the lifecycle of the result
     * @return a newly allocated {@link NDArray} containing {@code A^-1}
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     * @throws ArithmeticException if the matrix is singular (non-invertible)
     */
    public static NDArray inv(NDArray a, Arena arena) {
        return LinalgOps.inv(a, arena);
    }

    /**
     * Computes the Moore-Penrose pseudo-inverse of a 2D matrix using SVD.
     *
     * @param a the 2D input matrix
     * @return a newly allocated {@link NDArray} containing the pseudo-inverse
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static NDArray pinv(NDArray a) {
        return LinalgOps.pinv(a);
    }

    /**
     * Computes the Moore-Penrose pseudo-inverse of a 2D matrix with cutoff threshold {@code rcond}.
     *
     * @param a the 2D input matrix
     * @param rcond singular value cutoff threshold ratio
     * @return a newly allocated {@link NDArray} containing the pseudo-inverse
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static NDArray pinv(NDArray a, double rcond) {
        return LinalgOps.pinv(a, rcond);
    }

    /**
     * Computes the Moore-Penrose pseudo-inverse of a 2D matrix with cutoff threshold {@code rcond} in {@code arena}.
     *
     * @param a the 2D input matrix
     * @param rcond singular value cutoff threshold ratio
     * @param arena memory arena governing the lifecycle of the result
     * @return a newly allocated {@link NDArray} containing the pseudo-inverse
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static NDArray pinv(NDArray a, double rcond, Arena arena) {
        return LinalgOps.pinv(a, rcond, arena);
    }

    /**
     * Solves the linear system of equations {@code A * x = b} for unknown vector or matrix {@code x}.
     *
     * @param a coefficient square matrix {@code A}
     * @param b right-hand side vector or matrix {@code b}
     * @return solution vector or matrix {@code x}
     * @throws IllegalArgumentException if {@code a} is not square or dimensions mismatch
     * @throws ArithmeticException if {@code a} is singular
     */
    public static NDArray solve(NDArray a, NDArray b) {
        return LinalgOps.solve(a, b);
    }

    /**
     * Solves the linear system {@code A * x = b}, allocating in {@code arena}.
     *
     * @param a coefficient square matrix {@code A}
     * @param b right-hand side vector or matrix {@code b}
     * @param arena memory arena governing the lifecycle of the result
     * @return solution vector or matrix {@code x}
     * @throws IllegalArgumentException if {@code a} is not square or dimensions mismatch
     * @throws ArithmeticException if {@code a} is singular
     */
    public static NDArray solve(NDArray a, NDArray b, Arena arena) {
        return LinalgOps.solve(a, b, arena);
    }

    // =========================================================================
    // CANONICAL MATRIX DECOMPOSITIONS
    // =========================================================================

    /**
     * Computes the Cholesky decomposition of a symmetric positive-definite matrix {@code A = L * L^T}.
     *
     * @param a symmetric positive-definite matrix
     * @return lower-triangular factor {@code L}
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     * @throws ArithmeticException if {@code a} is not positive-definite
     */
    public static NDArray cholesky(NDArray a) {
        return LinalgOps.cholesky(a);
    }

    /**
     * Computes the Cholesky decomposition of a symmetric positive-definite matrix in {@code arena}.
     *
     * @param a symmetric positive-definite matrix
     * @param arena memory arena governing the lifecycle of the result
     * @return lower-triangular factor {@code L}
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     * @throws ArithmeticException if {@code a} is not positive-definite
     */
    public static NDArray cholesky(NDArray a, Arena arena) {
        return LinalgOps.cholesky(a, arena);
    }

    /**
     * Computes the QR decomposition of matrix {@code A = Q * R}.
     *
     * @param a the 2D input matrix
     * @return {@link QRResult} containing orthogonal {@code Q} and upper-triangular {@code R}
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static QRResult qr(NDArray a) {
        return LinalgOps.qr(a);
    }

    /**
     * Computes the QR decomposition of matrix {@code A = Q * R} in {@code arena}.
     *
     * @param a the 2D input matrix
     * @param arena memory arena governing the lifecycle of the results
     * @return {@link QRResult} containing orthogonal {@code Q} and upper-triangular {@code R}
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static QRResult qr(NDArray a, Arena arena) {
        return LinalgOps.qr(a, arena);
    }

    /**
     * Computes the Singular Value Decomposition (SVD) of matrix {@code A = U * S * V^T}.
     *
     * @param a the 2D input matrix
     * @return {@link SVDResult} containing left vectors {@code U}, singular values {@code S}, and right vectors {@code V^T}
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static SVDResult svd(NDArray a) {
        return LinalgOps.svd(a);
    }

    /**
     * Computes the Singular Value Decomposition (SVD) of matrix {@code A = U * S * V^T} in {@code arena}.
     *
     * @param a the 2D input matrix
     * @param arena memory arena governing the lifecycle of the results
     * @return {@link SVDResult} containing left vectors {@code U}, singular values {@code S}, and right vectors {@code V^T}
     * @throws IllegalArgumentException if {@code a} is not a 2D matrix
     */
    public static SVDResult svd(NDArray a, Arena arena) {
        return LinalgOps.svd(a, arena);
    }

    /**
     * Computes eigenvalues and eigenvectors of a symmetric real matrix.
     *
     * @param a 2D symmetric real matrix
     * @return {@link EighResult} containing real eigenvalues and orthogonal eigenvectors
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static EighResult eigh(NDArray a) {
        return LinalgOps.eigh(a);
    }

    /**
     * Computes eigenvalues and eigenvectors of a symmetric real matrix in {@code arena}.
     *
     * @param a 2D symmetric real matrix
     * @param arena memory arena governing the lifecycle of the results
     * @return {@link EighResult} containing real eigenvalues and orthogonal eigenvectors
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static EighResult eigh(NDArray a, Arena arena) {
        return LinalgOps.eigh(a, arena);
    }

    /**
     * Computes eigenvalues and eigenvectors of a general square real matrix.
     *
     * @param a 2D square real matrix
     * @return {@link EigResult} containing real and imaginary eigenvalue components
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static EigResult eig(NDArray a) {
        return LinalgOps.eig(a);
    }

    /**
     * Computes eigenvalues and eigenvectors of a general square real matrix in {@code arena}.
     *
     * @param a 2D square real matrix
     * @param arena memory arena governing the lifecycle of the results
     * @return {@link EigResult} containing real and imaginary eigenvalue components
     * @throws IllegalArgumentException if {@code a} is not a 2D square matrix
     */
    public static EigResult eig(NDArray a, Arena arena) {
        return LinalgOps.eig(a, arena);
    }
}
