package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.linalg.MatMul;

public final class LinalgOps {

    private LinalgOps() {
        throw new AssertionError("No jnum.internal.ops.LinalgOps instances for you!");
    }

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray res) {
        return MatMul.matmulFloat(a, b, res);
    }

    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray res) {
        return MatMul.matmulDouble(a, b, res);
    }

    public static NDArray matmulInt(NDArray a, NDArray b, NDArray res) {
        return MatMul.matmulInt(a, b, res);
    }
}
