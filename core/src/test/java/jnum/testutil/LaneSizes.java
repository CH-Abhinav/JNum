package jnum.testutil;

import jnum.internal.Constants;

public final class LaneSizes {

    private LaneSizes() {
        throw new AssertionError("Utility class");
    }

    public static final int VL_F32 = Constants.VL_F32;
    public static final int VL_F64 = Constants.VL_F64;
    public static final int VL_I32 = Constants.VL_I32;
    public static final int VL_BOOL = Constants.VL_BOOL;

    public static long[] boundarySizes(int vl) {
        return new long[]{
            0,
            1,
            Math.max(0, vl - 1),
            vl,
            vl + 1,
            Math.max(0, 2 * vl - 1),
            2L * vl,
            2L * vl + 1,
            1024,
            1025
        };
    }

    public static long[] boundarySizesF32() {
        return boundarySizes(VL_F32);
    }

    public static long[] boundarySizesF64() {
        return boundarySizes(VL_F64);
    }

    public static long[] boundarySizesI32() {
        return boundarySizes(VL_I32);
    }

    public static long[] boundarySizesBool() {
        return boundarySizes(VL_BOOL);
    }
}
