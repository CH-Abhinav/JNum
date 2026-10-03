package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.logical.*;

public class BooleanOps {

    private BooleanOps() {
        throw new AssertionError();
    }

    public static NDArray and(NDArray a, NDArray b, NDArray resArray) {
        return And.and(a, b, resArray);
    }

    public static NDArray or(NDArray a, NDArray b, NDArray resArray) {
        return Or.or(a, b, resArray);
    }

    public static NDArray xor(NDArray a, NDArray b, NDArray resArray) {
        return Xor.xor(a, b, resArray);
    }

    public static NDArray not(NDArray a, NDArray resArray) {
        return Not.not(a, resArray);
    }

    public static boolean any(NDArray a) {
        return Any.any(a);
    }

    public static boolean all(NDArray a) {
        return All.all(a);
    }
}
