package jnum.internal.ops;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.logical.*;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.layout.ValidUtil;

/**
 * High-level orchestration facade for boolean tensor operations (and, or, xor, not, all, any).
 *
 * <p>Enforces {@link DType#bool} operand casting, applies multidimensional broadcasting,
 * and delegates to bitwise SIMD kernels in {@link jnum.internal.kernel.logical}.</p>
 */
public final class BooleanOps {

    private BooleanOps() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    // =========================================================================
    // Logical Binary Operations (and, or, xor)
    // =========================================================================

    /** Element-wise logical AND between boolean tensors with broadcasting. */
    public static NDArray and(NDArray a, NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return And.and(A, B, resArray);
    }

    public static NDArray and(NDArray a, NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return And.and(A, B, targetRes);
    }

    public static NDArray or(NDArray a, NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return Or.or(A, B, resArray);
    }

    public static NDArray or(NDArray a, NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return Or.or(A, B, targetRes);
    }

    public static NDArray xor(NDArray a, NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return Xor.xor(A, B, resArray);
    }

    public static NDArray xor(NDArray a, NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return Xor.xor(A, B, targetRes);
    }

    public static NDArray not(NDArray a) {
        NDArray A = a.getDType() == DType.bool ? a : a.cast(DType.bool);
        NDArray safeThis = A.isContiguous() ? A : A.contiguous();
        NDArray resArray = JNum.zeros(DType.bool, safeThis.internalShapeUnsafe());
        return Not.not(safeThis, resArray);
    }

    public static NDArray not(NDArray a, NDArray resArray) {
        NDArray A = a.getDType() == DType.bool ? a : a.cast(DType.bool);
        NDArray safeThis = A.isContiguous() ? A : A.contiguous();
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, safeThis.internalShapeUnsafe());
        return Not.not(safeThis, targetRes);
    }

    public static boolean any(NDArray a) {
        NDArray A = a.getDType() == DType.bool ? a : a.cast(DType.bool);
        return Any.any(A);
    }

    public static boolean all(NDArray a) {
        NDArray A = a.getDType() == DType.bool ? a : a.cast(DType.bool);
        return All.all(A);
    }
}
