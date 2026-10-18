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
import jnum.internal.kernel.trig.*;

/**
 * High-level orchestration facade for trigonometric and hyperbolic functions (sin, cos, tan, sinh, cosh, tanh).
 *
 * <p>Ensures contiguous buffer materialization if necessary before invoking vectorized
 * math kernels in {@link jnum.internal.kernel.trig}.</p>
 */
public final class TrigOps {

    private TrigOps() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    // =========================================================================
    // Trigonometric Operations
    // =========================================================================

    /** Computes element-wise sine, converting integer inputs to float32. */
    public static NDArray sin(NDArray a) {
        return sin(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sin(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sin.sinFloat(safeThis, res);
            case f64 -> Sin.sinDouble(safeThis, res);
            case i32 -> Sin.sinInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sin(NDArray a, Arena arena) {
        return sin(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray cos(NDArray a) {
        return cos(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray cos(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Cos.cosFloat(safeThis, res);
            case f64 -> Cos.cosDouble(safeThis, res);
            case i32 -> Cos.cosInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray cos(NDArray a, Arena arena) {
        return cos(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray tan(NDArray a) {
        return tan(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray tan(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Tan.tanFloat(safeThis, res);
            case f64 -> Tan.tanDouble(safeThis, res);
            case i32 -> Tan.tanInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray tan(NDArray a, Arena arena) {
        return tan(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sinh(NDArray a) {
        return sinh(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray sinh(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sinh.sinhFloat(safeThis, res);
            case f64 -> Sinh.sinhDouble(safeThis, res);
            case i32 -> Sinh.sinhInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sinh(NDArray a, Arena arena) {
        return sinh(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray cosh(NDArray a) {
        return cosh(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray cosh(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Cosh.coshFloat(safeThis, res);
            case f64 -> Cosh.coshDouble(safeThis, res);
            case i32 -> Cosh.coshInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray cosh(NDArray a, Arena arena) {
        return cosh(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray tanh(NDArray a) {
        return tanh(a, JNum.zeros(a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }

    public static NDArray tanh(NDArray a, NDArray res) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Tanh.tanhFloat(safeThis, res);
            case f64 -> Tanh.tanhDouble(safeThis, res);
            case i32 -> Tanh.tanhInt(safeThis, res);
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray tanh(NDArray a, Arena arena) {
        return tanh(a, JNum.zeros(arena, a.getDType() == DType.i32 ? DType.f32 : a.getDType(), a.internalShapeUnsafe()));
    }
}
