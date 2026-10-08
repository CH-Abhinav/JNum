package jnum.internal.ops;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.trig.*;

public class TrigOps {
    private TrigOps() {
        throw new AssertionError();
    }

    // =========================================================================
    // High-Level Trigonometric Operations
    // =========================================================================

    public static NDArray sin(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sin.sinFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Sin.sinDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Sin.sinInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray cos(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Cos.cosFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Cos.cosDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Cos.cosInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray tan(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Tan.tanFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Tan.tanDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Tan.tanInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sinh(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sinh.sinhFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Sinh.sinhDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Sinh.sinhInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray cosh(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Cosh.coshFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Cosh.coshDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Cosh.coshInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray tanh(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Tanh.tanhFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Tanh.tanhDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Tanh.tanhInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }
}
