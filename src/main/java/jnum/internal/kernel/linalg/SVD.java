package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class SVD {

    private SVD() {
        throw new AssertionError("SVD kernel cannot be instantiated.");
    }

    public record SVDResult(NDArray u, NDArray s, NDArray vt) {}

    public static SVDResult svd(NDArray a, Arena arena) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("SVD requires a 2D matrix, got shape: " + a.shapeString());
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return svdFloat(a, arena);
        } else {
            return svdDouble(a, arena);
        }
    }

    public static SVDResult svdFloat(NDArray a, Arena arena) {
        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];

        // If m < n, compute SVD of A^T and swap U and V
        if (m < n) {
            NDArray aT = a.transpose();
            SVDResult resT = svdFloat(aT, arena);
            return new SVDResult(resT.vt().transpose(), resT.s(), resT.u().transpose());
        }

        int k = Math.min(m, n);
        NDArray uArr = JNum.zeros(arena, DType.f32, m, m);
        NDArray sArr = JNum.zeros(arena, DType.f32, k);
        NDArray vtArr = JNum.zeros(arena, DType.f32, n, n);

        MemorySegment uSeg = uArr.getData();
        MemorySegment sSeg = sArr.getData();
        MemorySegment vtSeg = vtArr.getData();

        // Initialize V = Identity in scratch
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aWork = scratch.allocate((long) m * n * BYTES_F32, 64);
            MemorySegment vWork = scratch.allocate((long) n * n * BYTES_F32, 64);

            NDArray contig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
            MemorySegment.copy(contig.getData(), 0, aWork, 0, (long) m * n * BYTES_F32);

            for (int i = 0; i < n; i++) {
                vWork.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i, 1.0f);
            }

            int maxSweeps = 30;
            float eps = 1e-7f;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                boolean converged = true;

                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        float alpha = 0.0f;
                        float beta = 0.0f;
                        float gamma = 0.0f;

                        for (int i = 0; i < m; i++) {
                            float ap = aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + p);
                            float aq = aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + q);
                            alpha += ap * ap;
                            beta  += aq * aq;
                            gamma += ap * aq;
                        }

                        if (Math.abs(gamma) > eps * Math.sqrt(alpha * beta)) {
                            converged = false;

                            float zeta = (beta - alpha) / (2.0f * gamma);
                            float t = (float) (Math.signum(zeta) / (Math.abs(zeta) + Math.sqrt(1.0f + zeta * zeta)));
                            if (Float.isNaN(t)) t = 0.0f;

                            float c = (float) (1.0 / Math.sqrt(1.0f + t * t));
                            float s = t * c;

                            // Rotate columns p and q of A
                            for (int i = 0; i < m; i++) {
                                float ap = aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + p);
                                float aq = aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + q);
                                aWork.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + p, c * ap - s * aq);
                                aWork.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + q, s * ap + c * aq);
                            }

                            // Rotate columns p and q of V
                            for (int i = 0; i < n; i++) {
                                float vp = vWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + p);
                                float vq = vWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + q);
                                vWork.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + p, c * vp - s * vq);
                                vWork.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + q, s * vp + c * vq);
                            }
                        }
                    }
                }
                if (converged) break;
            }

            // Compute singular values and normalize columns to form U
            float[] sVals = new float[n];
            for (int j = 0; j < n; j++) {
                float normSq = 0.0f;
                for (int i = 0; i < m; i++) {
                    float v = aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + j);
                    normSq += v * v;
                }
                sVals[j] = (float) Math.sqrt(normSq);
            }

            // Sort singular values descending
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (sVals[order[j]] > sVals[order[i]]) {
                        int tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                }
            }

            for (int j = 0; j < k; j++) {
                int col = order[j];
                float sigma = sVals[col];
                sSeg.setAtIndex(ValueLayout.JAVA_FLOAT, j, sigma);

                float invSigma = sigma > 1e-12f ? (1.0f / sigma) : 0.0f;
                for (int i = 0; i < m; i++) {
                    uSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * m + j, aWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + col) * invSigma);
                }
            }

            // Write V^T
            for (int j = 0; j < n; j++) {
                int col = order[j];
                for (int i = 0; i < n; i++) {
                    vtSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * n + i, vWork.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + col));
                }
            }
        }
        return new SVDResult(uArr, sArr, vtArr);
    }

    public static SVDResult svdDouble(NDArray a, Arena arena) {
        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];

        if (m < n) {
            NDArray aT = a.transpose();
            SVDResult resT = svdDouble(aT, arena);
            return new SVDResult(resT.vt().transpose(), resT.s(), resT.u().transpose());
        }

        int k = Math.min(m, n);
        NDArray uArr = JNum.zeros(arena, DType.f64, m, m);
        NDArray sArr = JNum.zeros(arena, DType.f64, k);
        NDArray vtArr = JNum.zeros(arena, DType.f64, n, n);

        MemorySegment uSeg = uArr.getData();
        MemorySegment sSeg = sArr.getData();
        MemorySegment vtSeg = vtArr.getData();

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aWork = scratch.allocate((long) m * n * BYTES_F64, 64);
            MemorySegment vWork = scratch.allocate((long) n * n * BYTES_F64, 64);

            NDArray contig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
            MemorySegment.copy(contig.getData(), 0, aWork, 0, (long) m * n * BYTES_F64);

            for (int i = 0; i < n; i++) {
                vWork.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i, 1.0);
            }

            int maxSweeps = 30;
            double eps = 1e-15;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                boolean converged = true;

                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        double alpha = 0.0;
                        double beta = 0.0;
                        double gamma = 0.0;

                        for (int i = 0; i < m; i++) {
                            double ap = aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + p);
                            double aq = aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + q);
                            alpha += ap * ap;
                            beta  += aq * aq;
                            gamma += ap * aq;
                        }

                        if (Math.abs(gamma) > eps * Math.sqrt(alpha * beta)) {
                            converged = false;

                            double zeta = (beta - alpha) / (2.0 * gamma);
                            double t = Math.signum(zeta) / (Math.abs(zeta) + Math.sqrt(1.0 + zeta * zeta));
                            if (Double.isNaN(t)) t = 0.0;

                            double c = 1.0 / Math.sqrt(1.0 + t * t);
                            double s = t * c;

                            for (int i = 0; i < m; i++) {
                                double ap = aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + p);
                                double aq = aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + q);
                                aWork.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + p, c * ap - s * aq);
                                aWork.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + q, s * ap + c * aq);
                            }

                            for (int i = 0; i < n; i++) {
                                double vp = vWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + p);
                                double vq = vWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + q);
                                vWork.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + p, c * vp - s * vq);
                                vWork.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + q, s * vp + c * vq);
                            }
                        }
                    }
                }
                if (converged) break;
            }

            double[] sVals = new double[n];
            for (int j = 0; j < n; j++) {
                double normSq = 0.0;
                for (int i = 0; i < m; i++) {
                    double v = aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + j);
                    normSq += v * v;
                }
                sVals[j] = Math.sqrt(normSq);
            }

            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (sVals[order[j]] > sVals[order[i]]) {
                        int tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                }
            }

            for (int j = 0; j < k; j++) {
                int col = order[j];
                double sigma = sVals[col];
                sSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, j, sigma);

                double invSigma = sigma > 1e-15 ? (1.0 / sigma) : 0.0;
                for (int i = 0; i < m; i++) {
                    uSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * m + j, aWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + col) * invSigma);
                }
            }

            for (int j = 0; j < n; j++) {
                int col = order[j];
                for (int i = 0; i < n; i++) {
                    vtSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * n + i, vWork.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + col));
                }
            }
        }
        return new SVDResult(uArr, sArr, vtArr);
    }
}