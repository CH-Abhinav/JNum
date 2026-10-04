package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class Inv {

    private Inv() {
        throw new AssertionError("Inv kernel cannot be instantiated.");
    }

    public static NDArray inv(NDArray a, Arena arena) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("Matrix inverse requires a square matrix, got shape: " + a.shapeString());
        }
        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return invFloat(a, arena);
        } else {
            return invDouble(a, arena);
        }
    }

    public static NDArray invFloat(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray invArr = JNum.zeros(arena, DType.f32, n, n);
        MemorySegment invSeg = invArr.getData();

        // Initialize Identity in invSeg
        for (int i = 0; i < n; i++) {
            invSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i, 1.0f);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F32, 64);
            NDArray contig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
            MemorySegment.copy(contig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F32);

            for (int i = 0; i < n; i++) {
                // Find pivot
                int maxRow = i;
                float maxVal = Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i));
                for (int k = i + 1; k < n; k++) {
                    float val = Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * n + i));
                    if (val > maxVal) {
                        maxVal = val;
                        maxRow = k;
                    }
                }

                if (maxVal < 1e-7f) {
                    throw new ArithmeticException("Matrix is singular and cannot be inverted.");
                }

                // Swap rows in both A and Inv
                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        float tmpA = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + c);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r1 + c, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + c));
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r2 + c, tmpA);

                        float tmpI = invSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + c);
                        invSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r1 + c, invSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + c));
                        invSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r2 + c, tmpI);
                    }
                }

                // Scale pivot row so A[i, i] == 1.0
                float pivot = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i);
                float invPivot = 1.0f / pivot;
                long rowI = (long) i * n;

                FloatVector vInvPivot = FloatVector.broadcast(SPECIES_F32, invPivot);
                int j = 0;
                int bound = SPECIES_F32.loopBound(n);
                for (; j < bound; j += VL_F32) {
                    FloatVector vA = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);
                    vA.mul(vInvPivot).intoMemorySegment(aSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);

                    FloatVector vI = FloatVector.fromMemorySegment(SPECIES_F32, invSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);
                    vI.mul(vInvPivot).intoMemorySegment(invSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);
                }
                for (; j < n; j++) {
                    aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowI + j, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j) * invPivot);
                    invSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowI + j, invSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j) * invPivot);
                }

                // Eliminate all other rows
                for (int k = 0; k < n; k++) {
                    if (k != i) {
                        long rowK = (long) k * n;
                        float factor = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + i);
                        if (Math.abs(factor) < 1e-12f) continue;

                        FloatVector vFactor = FloatVector.broadcast(SPECIES_F32, factor);
                        int col = 0;
                        for (; col < bound; col += VL_F32) {
                            FloatVector destA = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (rowK + col) * BYTES_F32, NATIVE_ORDER);
                            FloatVector srcA  = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (rowI + col) * BYTES_F32, NATIVE_ORDER);
                            destA.sub(srcA.mul(vFactor)).intoMemorySegment(aSeg, (rowK + col) * BYTES_F32, NATIVE_ORDER);

                            FloatVector destI = FloatVector.fromMemorySegment(SPECIES_F32, invSeg, (rowK + col) * BYTES_F32, NATIVE_ORDER);
                            FloatVector srcI  = FloatVector.fromMemorySegment(SPECIES_F32, invSeg, (rowI + col) * BYTES_F32, NATIVE_ORDER);
                            destI.sub(srcI.mul(vFactor)).intoMemorySegment(invSeg, (rowK + col) * BYTES_F32, NATIVE_ORDER);
                        }
                        for (; col < n; col++) {
                            float curA = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + col);
                            aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowK + col, curA - factor * aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + col));

                            float curI = invSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + col);
                            invSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowK + col, curI - factor * invSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + col));
                        }
                    }
                }
            }
        }
        return invArr;
    }

    public static NDArray invDouble(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray invArr = JNum.zeros(arena, DType.f64, n, n);
        MemorySegment invSeg = invArr.getData();

        for (int i = 0; i < n; i++) {
            invSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i, 1.0);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F64, 64);
            NDArray contig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
            MemorySegment.copy(contig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F64);

            for (int i = 0; i < n; i++) {
                int maxRow = i;
                double maxVal = Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i));
                for (int k = i + 1; k < n; k++) {
                    double val = Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) k * n + i));
                    if (val > maxVal) {
                        maxVal = val;
                        maxRow = k;
                    }
                }

                if (maxVal < 1e-15) {
                    throw new ArithmeticException("Matrix is singular and cannot be inverted.");
                }

                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        double tmpA = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c, aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c));
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c, tmpA);

                        double tmpI = invSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c);
                        invSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c, invSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c));
                        invSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c, tmpI);
                    }
                }

                double pivot = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i);
                double invPivot = 1.0 / pivot;
                long rowI = (long) i * n;

                DoubleVector vInvPivot = DoubleVector.broadcast(SPECIES_F64, invPivot);
                int j = 0;
                int bound = SPECIES_F64.loopBound(n);
                for (; j < bound; j += VL_F64) {
                    DoubleVector vA = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);
                    vA.mul(vInvPivot).intoMemorySegment(aSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);

                    DoubleVector vI = DoubleVector.fromMemorySegment(SPECIES_F64, invSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);
                    vI.mul(vInvPivot).intoMemorySegment(invSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);
                }
                for (; j < n; j++) {
                    aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j, aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j) * invPivot);
                    invSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j, invSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j) * invPivot);
                }

                for (int k = 0; k < n; k++) {
                    if (k != i) {
                        long rowK = (long) k * n;
                        double factor = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + i);
                        if (Math.abs(factor) < 1e-15) continue;

                        DoubleVector vFactor = DoubleVector.broadcast(SPECIES_F64, factor);
                        int col = 0;
                        for (; col < bound; col += VL_F64) {
                            DoubleVector destA = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (rowK + col) * BYTES_F64, NATIVE_ORDER);
                            DoubleVector srcA  = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (rowI + col) * BYTES_F64, NATIVE_ORDER);
                            destA.sub(srcA.mul(vFactor)).intoMemorySegment(aSeg, (rowK + col) * BYTES_F64, NATIVE_ORDER);

                            DoubleVector destI = DoubleVector.fromMemorySegment(SPECIES_F64, invSeg, (rowK + col) * BYTES_F64, NATIVE_ORDER);
                            DoubleVector srcI  = DoubleVector.fromMemorySegment(SPECIES_F64, invSeg, (rowI + col) * BYTES_F64, NATIVE_ORDER);
                            destI.sub(srcI.mul(vFactor)).intoMemorySegment(invSeg, (rowK + col) * BYTES_F64, NATIVE_ORDER);
                        }
                        for (; col < n; col++) {
                            double curA = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + col);
                            aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowK + col, curA - factor * aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + col));

                            double curI = invSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + col);
                            invSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowK + col, curI - factor * invSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + col));
                        }
                    }
                }
            }
        }
        return invArr;
    }
}