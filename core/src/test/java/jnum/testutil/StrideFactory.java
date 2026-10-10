package jnum.testutil;

import jnum.NDArray;
import jnum.Slice;

public final class StrideFactory {

    private StrideFactory() {
        throw new AssertionError("Utility class");
    }

    public static NDArray transposed(NDArray arr) {
        return arr.transpose();
    }

    public static NDArray sliced(NDArray arr, Slice... slices) {
        return arr.slice(slices);
    }

    public static NDArray reversed(NDArray arr, int axis) {
        Slice[] slices = new Slice[arr.dim()];
        for (int i = 0; i < slices.length; i++) {
            if (i == axis) {
                slices[i] = new Slice(Slice.UNBOUNDED_START, Slice.UNBOUNDED_STOP, -1);
            } else {
                slices[i] = Slice.all();
            }
        }
        return arr.slice(slices);
    }

    public static NDArray stridedStep(NDArray arr, int step) {
        Slice[] slices = new Slice[arr.dim()];
        for (int i = 0; i < slices.length; i++) {
            slices[i] = new Slice(0, arr.getShape()[i], step);
        }
        return arr.slice(slices);
    }

    public static NDArray broadcast(NDArray arr, long... targetShape) {
        return arr.broadcastTo(targetShape);
    }
}
