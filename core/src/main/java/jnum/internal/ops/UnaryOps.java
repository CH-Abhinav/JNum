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
package jnum.internal.ops;

import java.lang.foreign.Arena;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.unary.*;

/**
 * High-level orchestration facade for unary mathematical operations (sqrt, abs, exp, log, log10, sigmoid).
 *
 * <p>Ensures input arrays are contiguous before delegating to vectorized unary kernels
 * in {@link jnum.internal.kernel.unary}.</p>
 */
public final class UnaryOps {

    private UnaryOps() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /** Computes element-wise square root. */
    public static NDArray sqrt(NDArray a) {
        return sqrt(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sqrt(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sqrt.sqrtFloat(safeThis, res);
            case f64 -> Sqrt.sqrtDouble(safeThis, res);
            case i32 -> Sqrt.sqrtInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sqrt(NDArray a, Arena arena) {
        return sqrt(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray abs(NDArray a) {
        return abs(a, JNum.zeros(a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray abs(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Abs.absFloat(safeThis, res);
            case f64 -> Abs.absDouble(safeThis, res);
            case i32 -> Abs.absInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray abs(NDArray a, Arena arena) {
        return abs(a, JNum.zeros(arena, a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray exp(NDArray a) {
        return exp(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray exp(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Exp.expFloat(safeThis, res);
            case f64 -> Exp.expDouble(safeThis, res);
            case i32 -> Exp.expInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray exp(NDArray a, Arena arena) {
        return exp(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray log(NDArray a) {
        return log(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray log(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Log.logFloat(safeThis, res);
            case f64 -> Log.logDouble(safeThis, res);
            case i32 -> Log.logInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray log(NDArray a, Arena arena) {
        return log(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray log10(NDArray a) {
        return log10(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray log10(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Log10.log10Float(safeThis, res);
            case f64 -> Log10.log10Double(safeThis, res);
            case i32 -> Log10.log10Int(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray log10(NDArray a, Arena arena) {
        return log10(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sigmoid(NDArray a) {
        return sigmoid(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sigmoid(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sigmoid.sigmoidFloat(safeThis, res);
            case f64 -> Sigmoid.sigmoidDouble(safeThis, res);
            case i32 -> Sigmoid.sigmoidInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sigmoid(NDArray a, Arena arena) {
        return sigmoid(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }
}
