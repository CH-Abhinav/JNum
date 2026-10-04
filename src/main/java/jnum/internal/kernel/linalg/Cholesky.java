package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class Cholesky {

    private Cholesky() {
        throw new AssertionError("Cholesky kernel cannot be instantiated.");
    }

    public static NDArray cholesky(NDArray a, Arena arena) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("Cholesky requires a square 2D matrix, got shape: " + a.shapeString());
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return choleskyFloat(a, arena);
        } else {
            return choleskyDouble(a, arena);
        }
    }

    public static NDArray choleskyFloat(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray lArr = JNum.zeros(arena, DType.f32, n, n);
        MemorySegment lSeg = lArr.getData();

        NDArray aContig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
        MemorySegment aSeg = aContig.getData();

        for (int i = 0; i < n; i++) {
            long rowI = (long) i * n;
            for (int j = 0; j <= i; j++) {
                long rowJ = (long) j * n;

                // SIMD dot product: sum_{k=0}^{j-1} L[i, k] * L[j, k]
                FloatVector acc = FloatVector.zero(SPECIES_F32);
                int k = 0;
                int bound = SPECIES_F32.loopBound(j);

                for (; k < bound; k += VL_F32) {
                    FloatVector vI = FloatVector.fromMemorySegment(SPECIES_F32, lSeg, (rowI + k) * BYTES_F32, NATIVE_ORDER);
                    FloatVector vJ = FloatVector.fromMemorySegment(SPECIES_F32, lSeg, (rowJ + k) * BYTES_F32, NATIVE_ORDER);
                    acc = acc.add(vI.mul(vJ));
                }
                float sum = acc.reduceLanes(VectorOperators.ADD);
                for (; k < j; k++) {
                    sum += lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + k) * lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowJ + k);
                }

                float aVal = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j);
                if (i == j) {
                    float diff = aVal - sum;
                    if (diff <= 0.0f || Float.isNaN(diff)) {
                        throw new ArithmeticException("Matrix is not positive-definite for Cholesky decomposition at diagonal index " + i);
                    }
                    lSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowI + j, (float) Math.sqrt(diff));
                } else {
                    float diagJ = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowJ + j);
                    lSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowI + j, (aVal - sum) / diagJ);
                }
            }
        }
        return lArr;
    }

    public static NDArray choleskyDouble(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray lArr = JNum.zeros(arena, DType.f64, n, n);
        MemorySegment lSeg = lArr.getData();

        NDArray aContig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
        MemorySegment aSeg = aContig.getData();

        for (int i = 0; i < n; i++) {
            long rowI = (long) i * n;
            for (int j = 0; j <= i; j++) {
                long rowJ = (long) j * n;

                DoubleVector acc = DoubleVector.zero(SPECIES_F64);
                int k = 0;
                int bound = SPECIES_F64.loopBound(j);

                for (; k < bound; k += VL_F64) {
                    DoubleVector vI = DoubleVector.fromMemorySegment(SPECIES_F64, lSeg, (rowI + k) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector vJ = DoubleVector.fromMemorySegment(SPECIES_F64, lSeg, (rowJ + k) * BYTES_F64, NATIVE_ORDER);
                    acc = acc.add(vI.mul(vJ));
                }
                double sum = acc.reduceLanes(VectorOperators.ADD);
                for (; k < j; k++) {
                    sum += lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + k) * lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowJ + k);
                }

                double aVal = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j);
                if (i == j) {
                    double diff = aVal - sum;
                    if (diff <= 0.0 || Double.isNaN(diff)) {
                        throw new ArithmeticException("Matrix is not positive-definite for Cholesky decomposition at diagonal index " + i);
                    }
                    lSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j, Math.sqrt(diff));
                } else {
                    double diagJ = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowJ + j);
                    lSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j, (aVal - sum) / diagJ);
                }
            }
        }
        return lArr;
    }
}