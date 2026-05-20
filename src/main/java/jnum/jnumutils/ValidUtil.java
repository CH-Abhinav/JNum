package jnum.jnumutils;

import java.util.Arrays;

import jnum.DType;
import jnum.NDArray;

public class ValidUtil {

    public static NDArray prepareBroadcastOperand(NDArray array, int[] targetShape, DType targetType) {
        try {
            return array.broadcastTo(targetShape).cast(targetType);
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException(
                "Cannot broadcast shape " + Arrays.toString(array.shape) +
                " to target shape " + Arrays.toString(targetShape) +
                " for dtype promotion to " + targetType,
                ex
            );
        }
    }

    public static NDArray validateResultArray(NDArray resArray, DType targetType, int[] targetShape) {
        if (resArray.dtype != targetType) {
            throw new IllegalArgumentException("Result dtype must be " + targetType + " but was " + resArray.dtype);
        }
        if (!Arrays.equals(resArray.shape, targetShape)) {
            throw new IllegalArgumentException("Result shape must be " + Arrays.toString(targetShape) + " but was " + Arrays.toString(resArray.shape));
        }
        return resArray;
    }

    public static void validateMatmulInputs(NDArray a, NDArray b) {
        if (a.ndim() != 2 || b.ndim() != 2) {
            throw new IllegalArgumentException(
                "Matmul requires 2D arrays, but got shapes " +
                Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape)
            );
        }
        if (a.shape[1] != b.shape[0]) {
            throw new IllegalArgumentException(
                "Matmul shape mismatch: left shape " + Arrays.toString(a.shape) +
                " and right shape " + Arrays.toString(b.shape) +
                " are incompatible because " + a.shape[1] + " != " + b.shape[0]
            );
        }
    }

    public static void checkSameDtype(NDArray a, NDArray b) {
    if (a.dtype != b.dtype) {
        throw new IllegalArgumentException(
            "DType mismatch: Cannot perform operation between " + a.dtype + " and " + b.dtype
        );
    }
    }

    public static void validateOutputBuffer(NDArray res){
        if (!res.isContiguous()) {
            throw new IllegalArgumentException(
                "Output buffer must be contiguous, but got shape " +
                Arrays.toString(res.shape) + " with strides " + Arrays.toString(res.strides)
            );
        }
    }
}
