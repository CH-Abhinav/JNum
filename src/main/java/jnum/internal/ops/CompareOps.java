package jnum.internal.ops;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.compare.*;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.layout.TypeUtil;
import jnum.internal.layout.ValidUtil;

public class CompareOps {

    private CompareOps() {
        throw new AssertionError();
    }

    // =========================================================================
    // High-Level Maximum
    // =========================================================================

    public static NDArray maximum(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Maximum.maximumFloat(A, B, resArray);
            case i32 -> Maximum.maximumInt(A, B, resArray);
            case f64 -> Maximum.maximumDouble(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray maximum(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Maximum.maximumFloat(A, b, resArray);
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> Maximum.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray maximum(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Maximum.maximumFloat(A, b, resArray);
            case i32 -> Maximum.maximumInt(A, b, resArray);
            case f64 -> Maximum.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray maximum(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> throw new UnsupportedOperationException();
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> Maximum.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // =========================================================================
    // High-Level Minimum
    // =========================================================================

    public static NDArray minimum(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Minimum.minimumFloat(A, B, resArray);
            case i32 -> Minimum.minimumInt(A, B, resArray);
            case f64 -> Minimum.minimumDouble(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray minimum(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Minimum.minimumFloat(A, b, resArray);
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> Minimum.minimumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray minimum(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Minimum.minimumFloat(A, b, resArray);
            case i32 -> Minimum.minimumInt(A, b, resArray);
            case f64 -> Minimum.minimumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray minimum(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> throw new UnsupportedOperationException();
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> Minimum.minimumDouble(A, b, resArray);
            default ->
                    throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }
}
