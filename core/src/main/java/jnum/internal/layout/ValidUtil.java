package jnum.internal.layout;

import java.util.Arrays;
import jnum.DType;
import jnum.NDArray;

/**
 * Validation and preparation routines for tensor dimensions, shapes, data types, and memory contiguousness.
 */
public final class ValidUtil {

    private ValidUtil() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    /**
     * Broadcasts and casts an operand tensor to a target shape and promoted data type prior to kernel execution.
     */
    public static NDArray prepareBroadcastOperand(NDArray array, long[] targetShape, DType targetType) {
        try {
            return array.broadcastTo(targetShape).cast(targetType);
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException(
                "Cannot broadcast shape " + Arrays.toString(array.internalShapeUnsafe()) +
                " to target shape " + Arrays.toString(targetShape) +
                " for dtype promotion to " + targetType,
                ex
            );
        }
    }

    /**
     * Validates that a user-supplied destination array has the required data type and exact target shape.
     */
    public static NDArray validateResultArray(NDArray resArray, DType targetType, long[] targetShape) {
        if (resArray.getDType() != targetType) {
            throw new IllegalArgumentException("Result dtype must be " + targetType + " but was " + resArray.getDType());
        }
        if (!Arrays.equals(resArray.internalShapeUnsafe(), targetShape)) {
            throw new IllegalArgumentException("Result shape must be " + Arrays.toString(targetShape) + " but was " + Arrays.toString(resArray.internalShapeUnsafe()));
        }
        return resArray;
    }

    /**
     * Validates that both operands for matrix multiplication are rank-2 and have matching inner contraction dimensions (A.cols == B.rows).
     */
    public static void validateMatmulInputs(NDArray a, NDArray b) {
        if (a.dim() != 2 || b.dim() != 2) {
            throw new IllegalArgumentException(
                "Matmul requires 2D arrays, but got shapes " +
                Arrays.toString(a.internalShapeUnsafe()) + " and " + Arrays.toString(b.internalShapeUnsafe())
            );
        }
        if (a.internalShapeUnsafe()[1] != b.internalShapeUnsafe()[0]) {
            throw new IllegalArgumentException(
                "Matmul shape mismatch: left shape " + Arrays.toString(a.internalShapeUnsafe()) +
                " and right shape " + Arrays.toString(b.internalShapeUnsafe()) +
                " are incompatible because " + a.internalShapeUnsafe()[1] + " != " + b.internalShapeUnsafe()[0]
            );
        }
    }

    /**
     * Asserts that two tensors share the exact same data type.
     */
    public static void checkSameDtype(NDArray a, NDArray b) {
        if (a.getDType() != b.getDType()) {
            throw new IllegalArgumentException(
                "DType mismatch: Cannot perform operation between " + a.getDType() + " and " + b.getDType()
            );
        }
    }

    /**
     * Asserts that an output buffer is C-contiguous so SIMD vector stores can write sequentially without stride hops.
     */
    public static void validateOutputBuffer(NDArray res) {
        if (!res.isContiguous()) {
            throw new IllegalArgumentException(
                "Output buffer must be contiguous, but got shape " +
                Arrays.toString(res.internalShapeUnsafe()) + " with strides " + Arrays.toString(res.internalStridesUnsafe())
            );
        }
    }
}
