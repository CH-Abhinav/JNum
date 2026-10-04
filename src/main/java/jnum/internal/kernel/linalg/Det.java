package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jnum.DType;
import jnum.NDArray;

public final class Det {

    private Det() {
        throw new AssertionError("Det kernel cannot be instantiated.");
    }

    public record SlogdetResult(double sign, double logAbsDet) {}

    public static float detFloat(NDArray a) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("det requires a square 2D matrix, got shape: " + a.shapeString());
        }
        int n = (int) a.internalShapeUnsafe()[0];
        if (n == 0) return 1.0f;
        if (n == 1) {
            return a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        }

        // Allocate scratch off-heap memory
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment workSeg = arena.allocate((long) n * n * BYTES_F32, 64);
            if (a.getDType() == DType.f32 && a.isContiguous()) {
                MemorySegment.copy(a.getData(), 0, workSeg, 0, (long) n * n * BYTES_F32);
            } else {
                NDArray f32Arr = a.cast(DType.f32).contiguous();
                MemorySegment.copy(f32Arr.getData(), 0, workSeg, 0, (long) n * n * BYTES_F32);
            }

            float det = 1.0f;
            for (int i = 0; i < n; i++) {
                int maxRow = i;
                float maxVal = Math.abs(workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i));
                for (int k = i + 1; k < n; k++) {
                    float val = Math.abs(workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * n + i));
                    if (val > maxVal) {
                        maxVal = val;
                        maxRow = k;
                    }
                }

                if (maxVal < 1e-7f) {
                    return 0.0f;
                }

                // Swap rows
                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        float tmp = workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + c);
                        workSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r1 + c, workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + c));
                        workSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r2 + c, tmp);
                    }
                    det = -det;
                }

                float diag = workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i);
                det *= diag;

                // SIMD row updates
                long rowI = (long) i * n;
                for (int k = i + 1; k < n; k++) {
                    long rowK = (long) k * n;
                    float factor = workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + i) / diag;
                    FloatVector vFactor = FloatVector.broadcast(SPECIES_F32, factor);

                    int j = i + 1;
                    int len = n - (i + 1);
                    int bound = i + 1 + SPECIES_F32.loopBound(len);

                    for (; j < bound; j += VL_F32) {
                        FloatVector vDest = FloatVector.fromMemorySegment(SPECIES_F32, workSeg, (rowK + j) * BYTES_F32, NATIVE_ORDER);
                        FloatVector vSrc = FloatVector.fromMemorySegment(SPECIES_F32, workSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);
                        vDest.sub(vSrc.mul(vFactor)).intoMemorySegment(workSeg, (rowK + j) * BYTES_F32, NATIVE_ORDER);
                    }
                    for (; j < n; j++) {
                        float cur = workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + j);
                        float sub = factor * workSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j);
                        workSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowK + j, cur - sub);
                    }
                }
            }
            return det;
        }
    }

    public static double detDouble(NDArray a) {
        SlogdetResult res = slogdet(a);
        if (res.sign() == 0.0) return 0.0;
        return res.sign() * Math.exp(res.logAbsDet());
    }

    public static SlogdetResult slogdet(NDArray a) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("det requires a square 2D matrix, got shape: " + a.shapeString());
        }
        int n = (int) a.internalShapeUnsafe()[0];
        if (n == 0) return new SlogdetResult(1.0, 0.0);
        if (n == 1) {
            double v = a.cast(DType.f64).contiguous().getData().getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            return new SlogdetResult(Math.signum(v), Math.log(Math.abs(v)));
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment workSeg = arena.allocate((long) n * n * BYTES_F64, 64);
            if (a.getDType() == DType.f64 && a.isContiguous()) {
                MemorySegment.copy(a.getData(), 0, workSeg, 0, (long) n * n * BYTES_F64);
            } else {
                NDArray f64Arr = a.cast(DType.f64).contiguous();
                MemorySegment.copy(f64Arr.getData(), 0, workSeg, 0, (long) n * n * BYTES_F64);
            }

            double sign = 1.0;
            double logAbsDet = 0.0;

            for (int i = 0; i < n; i++) {
                int maxRow = i;
                double maxVal = Math.abs(workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i));
                for (int k = i + 1; k < n; k++) {
                    double val = Math.abs(workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) k * n + i));
                    if (val > maxVal) {
                        maxVal = val;
                        maxRow = k;
                    }
                }

                if (maxVal < 1e-15) {
                    return new SlogdetResult(0.0, Double.NEGATIVE_INFINITY);
                }

                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        double tmp = workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c);
                        workSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c, workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c));
                        workSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c, tmp);
                    }
                    sign = -sign;
                }

                double diag = workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i);
                if (diag < 0) {
                    sign = -sign;
                }
                logAbsDet += Math.log(Math.abs(diag));

                long rowI = (long) i * n;
                for (int k = i + 1; k < n; k++) {
                    long rowK = (long) k * n;
                    double factor = workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + i) / diag;
                    DoubleVector vFactor = DoubleVector.broadcast(SPECIES_F64, factor);

                    int j = i + 1;
                    int len = n - (i + 1);
                    int bound = i + 1 + SPECIES_F64.loopBound(len);

                    for (; j < bound; j += VL_F64) {
                        DoubleVector vDest = DoubleVector.fromMemorySegment(SPECIES_F64, workSeg, (rowK + j) * BYTES_F64, NATIVE_ORDER);
                        DoubleVector vSrc = DoubleVector.fromMemorySegment(SPECIES_F64, workSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);
                        vDest.sub(vSrc.mul(vFactor)).intoMemorySegment(workSeg, (rowK + j) * BYTES_F64, NATIVE_ORDER);
                    }
                    for (; j < n; j++) {
                        double cur = workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + j);
                        double sub = factor * workSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j);
                        workSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowK + j, cur - sub);
                    }
                }
            }

            return new SlogdetResult(sign, logAbsDet);
        }
    }
}