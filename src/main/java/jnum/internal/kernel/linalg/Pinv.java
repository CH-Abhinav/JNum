package jnum.internal.kernel.linalg;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class Pinv {

    private Pinv() {
        throw new AssertionError("Pinv kernel cannot be instantiated.");
    }

    public static NDArray pinv(NDArray a, Arena arena) {
        return pinv(a, -1.0, arena);
    }

    public static NDArray pinv(NDArray a, double rcond, Arena arena) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("pinv requires a 2D matrix, got shape: " + a.shapeString());
        }

        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];
        int k = Math.min(m, n);

        DType dtype = a.getDType();
        NDArray res = JNum.zeros(arena, dtype == DType.f32 ? DType.f32 : DType.f64, n, m);
        MemorySegment resSeg = res.getData();

        SVD.SVDResult svd = SVD.svd(a, arena);
        MemorySegment uSeg = svd.u().getData();
        MemorySegment sSeg = svd.s().getData();
        MemorySegment vtSeg = svd.vt().getData();

        if (dtype == DType.f32) {
            float maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            float cutoff = rcond >= 0.0 ? (float) (rcond * maxSigma) : Math.max(m, n) * 1.1920929e-7f * maxSigma;

            // Invert singular values
            float[] sInv = new float[k];
            for (int i = 0; i < k; i++) {
                float s = sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                sInv[i] = s > cutoff ? (1.0f / s) : 0.0f;
            }

            // Compute A^+ = V * S^+ * U^T directly into resSeg
            // res[i, j] = sum_{r=0}^{k-1} V[i, r] * sInv[r] * U[j, r]
            // Note: vtSeg stores V^T, so V[i, r] is vtSeg[r, i]
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    float sum = 0.0f;
                    for (int r = 0; r < k; r++) {
                        float vVal = vtSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + i);
                        float uVal = uSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) j * m + r);
                        sum += vVal * sInv[r] * uVal;
                    }
                    resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * m + j, sum);
                }
            }
        } else {
            double maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            double cutoff = rcond >= 0.0 ? (rcond * maxSigma) : Math.max(m, n) * 2.220446049250313e-16 * maxSigma;

            double[] sInv = new double[k];
            for (int i = 0; i < k; i++) {
                double s = sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                sInv[i] = s > cutoff ? (1.0 / s) : 0.0;
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    double sum = 0.0;
                    for (int r = 0; r < k; r++) {
                        double vVal = vtSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + i);
                        double uVal = uSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * m + r);
                        sum += vVal * sInv[r] * uVal;
                    }
                    resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * m + j, sum);
                }
            }
        }
        return res;
    }
}