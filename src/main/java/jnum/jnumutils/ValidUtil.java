package jnum.jnumutils;

import java.util.Arrays;
import jnum.DType;
import jnum.NDArray;

public class ValidUtil {

    public static NDArray prepareBroadcastOperand(NDArray array, int[] targetShape, DType targetType) {
        return array.broadcastTo(targetShape).cast(targetType);
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
            throw new IllegalArgumentException();
        }
        if (a.shape[1] != b.shape[0]) {
            throw new IllegalArgumentException();
        }
    }
}
