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

public final class QR {

    private QR() {
        throw new AssertionError("QR kernel cannot be instantiated.");
    }

    public record QRResult(NDArray q, NDArray r) {}

    public static QRResult qr(NDArray a, Arena arena) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("QR decomposition requires a 2D matrix, got shape: " + a.shapeString());
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return qrFloat(a, arena);
        } else {
            return qrDouble(a, arena);
        }
    }

    public static QRResult qrFloat(NDArray a, Arena arena) {
        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];

        NDArray qArr = JNum.zeros(arena, DType.f32, m, m);
        NDArray rArr = JNum.zeros(arena, DType.f32, m, n);
        MemorySegment qSeg = qArr.getData();
        MemorySegment rSeg = rArr.getData();

        // Initialize Q = Identity
        for (int i = 0; i < m; i++) {
            qSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * m + i, 1.0f);
        }

        // Copy A into R
        NDArray aContig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
        MemorySegment.copy(aContig.getData(), 0, rSeg, 0, (long) m * n * BYTES_F32);

        int minMN = Math.min(m, n);

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment vSeg = scratch.allocate((long) m * BYTES_F32, 64);

            for (int k = 0; k < minMN; k++) {
                // Compute norm of column k below diagonal
                float normX = 0.0f;
                for (int i = k; i < m; i++) {
                    float val = rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + k);
                    normX += val * val;
                }
                normX = (float) Math.sqrt(normX);
                if (normX < 1e-12f) continue;

                float rkk = rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * n + k);
                float alpha = (rkk >= 0.0f ? -1.0f : 1.0f) * normX;

                // Form Householder vector v
                for (int i = k; i < m; i++) {
                    vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + k));
                }
                vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, k, rkk - alpha);

                float vNormSq = 0.0f;
                for (int i = k; i < m; i++) {
                    float vi = vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    vNormSq += vi * vi;
                }
                if (vNormSq < 1e-12f) continue;

                float tau = 2.0f / vNormSq;

                // Apply Householder reflector to R: R = R - tau * v * (v^T * R)
                for (int j = k; j < n; j++) {
                    float dot = 0.0f;
                    for (int i = k; i < m; i++) {
                        dot += vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j);
                    }
                    float scale = tau * dot;
                    for (int i = k; i < m; i++) {
                        long idx = (long) i * n + j;
                        float cur = rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                        rSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, cur - scale * vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                    }
                }

                // Apply Householder reflector to Q: Q = Q - tau * (Q * v) * v^T
                for (int j = 0; j < m; j++) {
                    float dot = 0.0f;
                    for (int i = k; i < m; i++) {
                        dot += vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) * qSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) j * m + i);
                    }
                    float scale = tau * dot;
                    for (int i = k; i < m; i++) {
                        long idx = (long) j * m + i;
                        float cur = qSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                        qSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, cur - scale * vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                    }
                }
            }
        }
        return new QRResult(qArr, rArr);
    }

    public static QRResult qrDouble(NDArray a, Arena arena) {
        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];

        NDArray qArr = JNum.zeros(arena, DType.f64, m, m);
        NDArray rArr = JNum.zeros(arena, DType.f64, m, n);
        MemorySegment qSeg = qArr.getData();
        MemorySegment rSeg = rArr.getData();

        for (int i = 0; i < m; i++) {
            qSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * m + i, 1.0);
        }

        NDArray aContig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
        MemorySegment.copy(aContig.getData(), 0, rSeg, 0, (long) m * n * BYTES_F64);

        int minMN = Math.min(m, n);

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment vSeg = scratch.allocate((long) m * BYTES_F64, 64);

            for (int k = 0; k < minMN; k++) {
                double normX = 0.0;
                for (int i = k; i < m; i++) {
                    double val = rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + k);
                    normX += val * val;
                }
                normX = Math.sqrt(normX);
                if (normX < 1e-15) continue;

                double rkk = rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) k * n + k);
                double alpha = (rkk >= 0.0 ? -1.0 : 1.0) * normX;

                for (int i = k; i < m; i++) {
                    vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + k));
                }
                vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, k, rkk - alpha);

                double vNormSq = 0.0;
                for (int i = k; i < m; i++) {
                    double vi = vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    vNormSq += vi * vi;
                }
                if (vNormSq < 1e-15) continue;

                double tau = 2.0 / vNormSq;

                for (int j = k; j < n; j++) {
                    double dot = 0.0;
                    for (int i = k; i < m; i++) {
                        dot += vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i) * rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j);
                    }
                    double scale = tau * dot;
                    for (int i = k; i < m; i++) {
                        long idx = (long) i * n + j;
                        double cur = rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx);
                        rSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, cur - scale * vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                    }
                }

                for (int j = 0; j < m; j++) {
                    double dot = 0.0;
                    for (int i = k; i < m; i++) {
                        dot += vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i) * qSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * m + i);
                    }
                    double scale = tau * dot;
                    for (int i = k; i < m; i++) {
                        long idx = (long) j * m + i;
                        double cur = qSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx);
                        qSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, cur - scale * vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                    }
                }
            }
        }
        return new QRResult(qArr, rArr);
    }
}