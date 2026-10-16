package jnum.internal.ops;

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
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sqrt.sqrtFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Sqrt.sqrtDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Sqrt.sqrtInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray abs(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Abs.absFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Abs.absDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Abs.absInt(safeThis, JNum.zeros(DType.i32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray exp(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Exp.expFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Exp.expDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Exp.expInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray log(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Log.logFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Log.logDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Log.logInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray log10(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Log10.log10Float(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Log10.log10Double(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Log10.log10Int(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sigmoid(NDArray a) {
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return switch (a.getDType()) {
            case f32 -> Sigmoid.sigmoidFloat(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            case f64 -> Sigmoid.sigmoidDouble(safeThis, JNum.zeros(DType.f64, a.internalShapeUnsafe()));
            case i32 -> Sigmoid.sigmoidInt(safeThis, JNum.zeros(DType.f32, a.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
        };
    }
}
