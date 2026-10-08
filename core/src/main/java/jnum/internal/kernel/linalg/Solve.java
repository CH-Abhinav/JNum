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

public final class Solve {

    private Solve() {
        throw new AssertionError("Solve kernel cannot be instantiated.");
    }

    public static NDArray solve(NDArray a, NDArray b, Arena arena) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("Solve requires a square matrix A, got shape: " + a.shapeString());
        }
        int n = (int) a.internalShapeUnsafe()[0];

        if (b.dim() != 1 && b.dim() != 2) {
            throw new IllegalArgumentException("Right-hand side b must be 1D or 2D, got ndim: " + b.dim());
        }
        if (b.internalShapeUnsafe()[0] != n) {
            throw new IllegalArgumentException(String.format("Dimension mismatch: A is %dx%d, but b has %d rows", n, n, b.internalShapeUnsafe()[0]));
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32 && b.getDType() == DType.f32) {
            return solveFloat(a, b, arena);
        } else {
            return solveDouble(a, b, arena);
        }
    }

    public static NDArray solveFloat(NDArray a, NDArray b, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        boolean is1D = b.dim() == 1;
        int bCols = is1D ? 1 : (int) b.internalShapeUnsafe()[1];

        // Result allocation
        NDArray res = is1D ? JNum.zeros(arena, DType.f32, n) : JNum.zeros(arena, DType.f32, n, bCols);
        MemorySegment xSeg = res.getData();

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F32, 64);
            MemorySegment bSeg = scratch.allocate((long) n * bCols * BYTES_F32, 64);

            // Copy input data
            NDArray aContig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
            NDArray bContig = (b.getDType() == DType.f32 && b.isContiguous()) ? b : b.cast(DType.f32).contiguous();
            MemorySegment.copy(aContig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F32);
            MemorySegment.copy(bContig.getData(), 0, bSeg, 0, (long) n * bCols * BYTES_F32);

            int[] piv = new int[n];
            for (int i = 0; i < n; i++) piv[i] = i;

            // 1. LU Factorization with Partial Pivoting
            for (int i = 0; i < n; i++) {
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
                    throw new ArithmeticException("Matrix A is singular to working precision; cannot solve system.");
                }

                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        float tmp = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + c);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r1 + c, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + c));
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, r2 + c, tmp);
                    }
                    int tmpP = piv[i]; piv[i] = piv[maxRow]; piv[maxRow] = tmpP;
                }

                float diag = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i);
                long rowI = (long) i * n;
                for (int k = i + 1; k < n; k++) {
                    long rowK = (long) k * n;
                    float factor = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + i) / diag;
                    aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowK + i, factor);

                    FloatVector vFactor = FloatVector.broadcast(SPECIES_F32, factor);
                    int j = i + 1;
                    int len = n - (i + 1);
                    int bound = i + 1 + SPECIES_F32.loopBound(len);

                    for (; j < bound; j += VL_F32) {
                        FloatVector vDest = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (rowK + j) * BYTES_F32, NATIVE_ORDER);
                        FloatVector vSrc = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (rowI + j) * BYTES_F32, NATIVE_ORDER);
                        vDest.sub(vSrc.mul(vFactor)).intoMemorySegment(aSeg, (rowK + j) * BYTES_F32, NATIVE_ORDER);
                    }
                    for (; j < n; j++) {
                        float cur = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowK + j);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, rowK + j, cur - factor * aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j));
                    }
                }
            }

            // 2. Forward & Backward Substitution for each column of B
            MemorySegment ySeg = scratch.allocate((long) n * BYTES_F32, 64);
            for (int col = 0; col < bCols; col++) {
                // Permute B into y
                for (int i = 0; i < n; i++) {
                    float bVal = bSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) piv[i] * bCols + col);
                    ySeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, bVal);
                }

                // Forward Substitution: L y = P b
                for (int i = 0; i < n; i++) {
                    long rowI = (long) i * n;
                    float yi = ySeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    for (int j = 0; j < i; j++) {
                        yi -= aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j) * ySeg.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                    }
                    ySeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, yi);
                }

                // Backward Substitution: U x = y
                for (int i = n - 1; i >= 0; i--) {
                    long rowI = (long) i * n;
                    float xi = ySeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    for (int j = i + 1; j < n; j++) {
                        xi -= aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + j) * xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) j * bCols + col);
                    }
                    xi /= aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rowI + i);
                    xSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * bCols + col, xi);
                }
            }
        }
        return res;
    }

    public static NDArray solveDouble(NDArray a, NDArray b, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        boolean is1D = b.dim() == 1;
        int bCols = is1D ? 1 : (int) b.internalShapeUnsafe()[1];

        NDArray res = is1D ? JNum.zeros(arena, DType.f64, n) : JNum.zeros(arena, DType.f64, n, bCols);
        MemorySegment xSeg = res.getData();

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F64, 64);
            MemorySegment bSeg = scratch.allocate((long) n * bCols * BYTES_F64, 64);

            NDArray aContig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
            NDArray bContig = (b.getDType() == DType.f64 && b.isContiguous()) ? b : b.cast(DType.f64).contiguous();
            MemorySegment.copy(aContig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F64);
            MemorySegment.copy(bContig.getData(), 0, bSeg, 0, (long) n * bCols * BYTES_F64);

            int[] piv = new int[n];
            for (int i = 0; i < n; i++) piv[i] = i;

            // 1. LU Factorization
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
                    throw new ArithmeticException("Matrix A is singular to working precision; cannot solve system.");
                }

                if (maxRow != i) {
                    long r1 = (long) i * n;
                    long r2 = (long) maxRow * n;
                    for (int c = 0; c < n; c++) {
                        double tmp = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r1 + c, aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c));
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, r2 + c, tmp);
                    }
                    int tmpP = piv[i]; piv[i] = piv[maxRow]; piv[maxRow] = tmpP;
                }

                double diag = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i);
                long rowI = (long) i * n;
                for (int k = i + 1; k < n; k++) {
                    long rowK = (long) k * n;
                    double factor = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + i) / diag;
                    aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowK + i, factor);

                    DoubleVector vFactor = DoubleVector.broadcast(SPECIES_F64, factor);
                    int j = i + 1;
                    int len = n - (i + 1);
                    int bound = i + 1 + SPECIES_F64.loopBound(len);

                    for (; j < bound; j += VL_F64) {
                        DoubleVector vDest = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (rowK + j) * BYTES_F64, NATIVE_ORDER);
                        DoubleVector vSrc = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (rowI + j) * BYTES_F64, NATIVE_ORDER);
                        vDest.sub(vSrc.mul(vFactor)).intoMemorySegment(aSeg, (rowK + j) * BYTES_F64, NATIVE_ORDER);
                    }
                    for (; j < n; j++) {
                        double cur = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowK + j);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, rowK + j, cur - factor * aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j));
                    }
                }
            }

            // 2. Forward & Backward Substitution
            MemorySegment ySeg = scratch.allocate((long) n * BYTES_F64, 64);
            for (int col = 0; col < bCols; col++) {
                for (int i = 0; i < n; i++) {
                    double bVal = bSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) piv[i] * bCols + col);
                    ySeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, bVal);
                }

                for (int i = 0; i < n; i++) {
                    long rowI = (long) i * n;
                    double yi = ySeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    for (int j = 0; j < i; j++) {
                        yi -= aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j) * ySeg.getAtIndex(ValueLayout.JAVA_DOUBLE, j);
                    }
                    ySeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, yi);
                }

                for (int i = n - 1; i >= 0; i--) {
                    long rowI = (long) i * n;
                    double xi = ySeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    for (int j = i + 1; j < n; j++) {
                        xi -= aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + j) * xSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * bCols + col);
                    }
                    xi /= aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rowI + i);
                    xSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * bCols + col, xi);
                }
            }
        }
        return res;
    }
}