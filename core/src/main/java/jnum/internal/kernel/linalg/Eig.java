package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class Eig {

    private Eig() {
        throw new AssertionError("Eig kernel cannot be instantiated.");
    }

    public record EigResult(NDArray realEigenvalues, NDArray imagEigenvalues) {}

    public static EigResult eig(NDArray a, Arena arena) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("eig requires a square 2D matrix, got shape: " + a.shapeString());
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return eigFloat(a, arena);
        } else {
            return eigDouble(a, arena);
        }
    }

    public static EigResult eigFloat(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray wrArr = JNum.zeros(arena, DType.f32, n);
        NDArray wiArr = JNum.zeros(arena, DType.f32, n);
        MemorySegment wrSeg = wrArr.getData();
        MemorySegment wiSeg = wiArr.getData();

        if (n == 0) return new EigResult(wrArr, wiArr);
        if (n == 1) {
            float val = a.cast(DType.f32).contiguous().getData().getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, 0, val);
            wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0.0f);
            return new EigResult(wrArr, wiArr);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment hSeg = scratch.allocate((long) n * n * BYTES_F32, 64);
            NDArray contig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
            MemorySegment.copy(contig.getData(), 0, hSeg, 0, (long) n * n * BYTES_F32);

            // Step 1: Reduce to upper Hessenberg form via Householder reflections
            for (int k = 0; k < n - 2; k++) {
                float normSq = 0.0f;
                for (int i = k + 1; i < n; i++) {
                    float val = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + k);
                    normSq += val * val;
                }
                float norm = (float) Math.sqrt(normSq);
                if (norm < 1e-12f) continue;

                float hVal = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (k + 1) * n + k);
                float alpha = (hVal >= 0.0f ? -1.0f : 1.0f) * norm;

                MemorySegment v = scratch.allocate((long) n * BYTES_F32, 64);
                for (int i = k + 1; i < n; i++) {
                    v.setAtIndex(ValueLayout.JAVA_FLOAT, i, hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + k));
                }
                v.setAtIndex(ValueLayout.JAVA_FLOAT, k + 1, hVal - alpha);

                float vNormSq = 0.0f;
                for (int i = k + 1; i < n; i++) {
                    float vi = v.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    vNormSq += vi * vi;
                }
                if (vNormSq < 1e-12f) continue;
                float tau = 2.0f / vNormSq;

                // H = (I - tau * v * v^T) * H
                for (int j = k; j < n; j++) {
                    float dot = 0.0f;
                    for (int i = k + 1; i < n; i++) {
                        dot += v.getAtIndex(ValueLayout.JAVA_FLOAT, i) * hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j);
                    }
                    float scale = tau * dot;
                    for (int i = k + 1; i < n; i++) {
                        long idx = (long) i * n + j;
                        float cur = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, cur - scale * v.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                    }
                }

                // H = H * (I - tau * v * v^T)
                for (int i = 0; i < n; i++) {
                    float dot = 0.0f;
                    for (int j = k + 1; j < n; j++) {
                        dot += v.getAtIndex(ValueLayout.JAVA_FLOAT, j) * hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j);
                    }
                    float scale = tau * dot;
                    for (int j = k + 1; j < n; j++) {
                        long idx = (long) i * n + j;
                        float cur = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, cur - scale * v.getAtIndex(ValueLayout.JAVA_FLOAT, j));
                    }
                }
            }

            // Step 2: Francis QR algorithm on Hessenberg matrix
            int p = n - 1;
            int iter = 0;
            float eps = 1e-7f;

            while (p >= 0) {
                if (p == 0) {
                    wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, 0, hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, 0));
                    wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0.0f);
                    break;
                }

                // Check 1x1 deflation
                float hSub = Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + (p - 1)));
                float hDiag = Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + (p - 1))) +
                        Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p));
                if (hSub <= eps * hDiag || hSub < 1e-12f) {
                    wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p));
                    wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, 0.0f);
                    p--;
                    iter = 0;
                    continue;
                }

                // Check 2x2 deflation
                float hSubPrev = p > 1 ? Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + (p - 2))) : 0.0f;
                float hDiagPrev = p > 1 ? Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 2) * n + (p - 2))) +
                        Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + (p - 1))) : 0.0f;
                if (p == 1 || hSubPrev <= eps * hDiagPrev || hSubPrev < 1e-12f) {
                    float a11 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + (p - 1));
                    float a12 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + p);
                    float a21 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + (p - 1));
                    float a22 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p);

                    float tr = a11 + a22;
                    float det = a11 * a22 - a12 * a21;
                    float disc = tr * tr - 4.0f * det;

                    if (disc >= 0.0f) {
                        float sqrtD = (float) Math.sqrt(disc);
                        wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p - 1, (tr + sqrtD) * 0.5f);
                        wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p - 1, 0.0f);
                        wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, (tr - sqrtD) * 0.5f);
                        wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, 0.0f);
                    } else {
                        float sqrtD = (float) Math.sqrt(-disc);
                        wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p - 1, tr * 0.5f);
                        wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p - 1, sqrtD * 0.5f);
                        wrSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, tr * 0.5f);
                        wiSeg.setAtIndex(ValueLayout.JAVA_FLOAT, p, -sqrtD * 0.5f);
                    }
                    p -= 2;
                    iter = 0;
                    continue;
                }

                // Francis QR step with Wilkinson shift
                float s = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p);
                float t = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (p - 1) * n + (p - 1));
                if (++iter % 10 == 0) {
                    s += hSub;
                    t += hSub;
                }

                for (int i = 0; i <= p; i++) {
                    long idx = (long) i * n + i;
                    hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx) - s);
                }

                // QR decomposition of shifted Hessenberg and RQ accumulation
                for (int i = 0; i < p; i++) {
                    float a1 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i);
                    float b1 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (i + 1) * n + i);
                    float r = (float) Math.hypot(a1, b1);
                    if (r < 1e-12f) continue;
                    float c = a1 / r;
                    float sn = b1 / r;

                    for (int j = i; j <= p; j++) {
                        float rowI = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j);
                        float rowI1 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) (i + 1) * n + j);
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j, c * rowI + sn * rowI1);
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) (i + 1) * n + j, -sn * rowI + c * rowI1);
                    }
                    for (int j = 0; j <= Math.min(i + 2, p); j++) {
                        float colI = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) j * n + i);
                        float colI1 = hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) j * n + (i + 1));
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * n + i, c * colI + sn * colI1);
                        hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * n + (i + 1), -sn * colI + c * colI1);
                    }
                }

                for (int i = 0; i <= p; i++) {
                    long idx = (long) i * n + i;
                    hSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, hSeg.getAtIndex(ValueLayout.JAVA_FLOAT, idx) + s);
                }
            }
        }
        return new EigResult(wrArr, wiArr);
    }

    public static EigResult eigDouble(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray wrArr = JNum.zeros(arena, DType.f64, n);
        NDArray wiArr = JNum.zeros(arena, DType.f64, n);
        MemorySegment wrSeg = wrArr.getData();
        MemorySegment wiSeg = wiArr.getData();

        if (n == 0) return new EigResult(wrArr, wiArr);
        if (n == 1) {
            double val = a.cast(DType.f64).contiguous().getData().getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, val);
            wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, 0.0);
            return new EigResult(wrArr, wiArr);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment hSeg = scratch.allocate((long) n * n * BYTES_F64, 64);
            NDArray contig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
            MemorySegment.copy(contig.getData(), 0, hSeg, 0, (long) n * n * BYTES_F64);

            // Step 1: Reduce to Hessenberg form
            for (int k = 0; k < n - 2; k++) {
                double normSq = 0.0;
                for (int i = k + 1; i < n; i++) {
                    double val = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + k);
                    normSq += val * val;
                }
                double norm = Math.sqrt(normSq);
                if (norm < 1e-15) continue;

                double hVal = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (k + 1) * n + k);
                double alpha = (hVal >= 0.0 ? -1.0 : 1.0) * norm;

                MemorySegment v = scratch.allocate((long) n * BYTES_F64, 64);
                for (int i = k + 1; i < n; i++) {
                    v.setAtIndex(ValueLayout.JAVA_DOUBLE, i, hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + k));
                }
                v.setAtIndex(ValueLayout.JAVA_DOUBLE, k + 1, hVal - alpha);

                double vNormSq = 0.0;
                for (int i = k + 1; i < n; i++) {
                    double vi = v.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    vNormSq += vi * vi;
                }
                if (vNormSq < 1e-15) continue;
                double tau = 2.0 / vNormSq;

                for (int j = k; j < n; j++) {
                    double dot = 0.0;
                    for (int i = k + 1; i < n; i++) {
                        dot += v.getAtIndex(ValueLayout.JAVA_DOUBLE, i) * hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j);
                    }
                    double scale = tau * dot;
                    for (int i = k + 1; i < n; i++) {
                        long idx = (long) i * n + j;
                        double cur = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx);
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, cur - scale * v.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                    }
                }

                for (int i = 0; i < n; i++) {
                    double dot = 0.0;
                    for (int j = k + 1; j < n; j++) {
                        dot += v.getAtIndex(ValueLayout.JAVA_DOUBLE, j) * hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j);
                    }
                    double scale = tau * dot;
                    for (int j = k + 1; j < n; j++) {
                        long idx = (long) i * n + j;
                        double cur = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx);
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, cur - scale * v.getAtIndex(ValueLayout.JAVA_DOUBLE, j));
                    }
                }
            }

            // Step 2: Francis QR algorithm
            int p = n - 1;
            int iter = 0;
            double eps = 1e-15;

            while (p >= 0) {
                if (p == 0) {
                    wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, 0));
                    wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, 0.0);
                    break;
                }

                double hSub = Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + (p - 1)));
                double hDiag = Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + (p - 1))) +
                        Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p));
                if (hSub <= eps * hDiag || hSub < 1e-15) {
                    wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p));
                    wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, 0.0);
                    p--;
                    iter = 0;
                    continue;
                }

                double hSubPrev = p > 1 ? Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + (p - 2))) : 0.0;
                double hDiagPrev = p > 1 ? Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 2) * n + (p - 2))) +
                        Math.abs(hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + (p - 1))) : 0.0;
                if (p == 1 || hSubPrev <= eps * hDiagPrev || hSubPrev < 1e-15) {
                    double a11 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + (p - 1));
                    double a12 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + p);
                    double a21 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + (p - 1));
                    double a22 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p);

                    double tr = a11 + a22;
                    double det = a11 * a22 - a12 * a21;
                    double disc = tr * tr - 4.0 * det;

                    if (disc >= 0.0) {
                        double sqrtD = Math.sqrt(disc);
                        wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p - 1, (tr + sqrtD) * 0.5);
                        wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p - 1, 0.0);
                        wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, (tr - sqrtD) * 0.5);
                        wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, 0.0);
                    } else {
                        double sqrtD = Math.sqrt(-disc);
                        wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p - 1, tr * 0.5);
                        wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p - 1, sqrtD * 0.5);
                        wrSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, tr * 0.5);
                        wiSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, p, -sqrtD * 0.5);
                    }
                    p -= 2;
                    iter = 0;
                    continue;
                }

                double s = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p);
                double t = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (p - 1) * n + (p - 1));
                if (++iter % 10 == 0) {
                    s += hSub;
                    t += hSub;
                }

                for (int i = 0; i <= p; i++) {
                    long idx = (long) i * n + i;
                    hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx) - s);
                }

                for (int i = 0; i < p; i++) {
                    double a1 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i);
                    double b1 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (i + 1) * n + i);
                    double r = Math.hypot(a1, b1);
                    if (r < 1e-15) continue;
                    double c = a1 / r;
                    double sn = b1 / r;

                    for (int j = i; j <= p; j++) {
                        double rowI = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j);
                        double rowI1 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) (i + 1) * n + j);
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j, c * rowI + sn * rowI1);
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) (i + 1) * n + j, -sn * rowI + c * rowI1);
                    }
                    for (int j = 0; j <= Math.min(i + 2, p); j++) {
                        double colI = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + i);
                        double colI1 = hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + (i + 1));
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + i, c * colI + sn * colI1);
                        hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + (i + 1), -sn * colI + c * colI1);
                    }
                }

                for (int i = 0; i <= p; i++) {
                    long idx = (long) i * n + i;
                    hSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, hSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, idx) + s);
                }
            }
        }
        return new EigResult(wrArr, wiArr);
    }
}