package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public final class Eigh {

    private Eigh() {
        throw new AssertionError("Eigh kernel cannot be instantiated.");
    }

    public record EighResult(NDArray eigenvalues, NDArray eigenvectors) {}

    public static EighResult eigh(NDArray a, Arena arena) {
        if (a.dim() != 2 || a.internalShapeUnsafe()[0] != a.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("eigh requires a square 2D symmetric matrix, got shape: " + a.shapeString());
        }

        DType dtype = a.getDType();
        if (dtype == DType.f32) {
            return eighFloat(a, arena);
        } else {
            return eighDouble(a, arena);
        }
    }

    public static EighResult eighFloat(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray wArr = JNum.zeros(arena, DType.f32, n);
        NDArray vArr = JNum.zeros(arena, DType.f32, n, n);
        MemorySegment wSeg = wArr.getData();
        MemorySegment vSeg = vArr.getData();

        // Initialize V = Identity
        for (int i = 0; i < n; i++) {
            vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i, 1.0f);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F32, 64);
            NDArray contig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
            MemorySegment.copy(contig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F32);

            int maxSweeps = 50;
            float eps = 1e-7f;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                float offDiagSum = 0.0f;
                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        offDiagSum += Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + q));
                    }
                }

                if (offDiagSum < eps) {
                    break;
                }

                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        float apq = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + q);
                        if (Math.abs(apq) < 1e-12f) continue;

                        float app = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p);
                        float aqq = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) q * n + q);

                        float theta = (aqq - app) / (2.0f * apq);
                        float t = (float) (Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0f)));
                        if (Float.isNaN(t)) t = 0.0f;

                        float c = (float) (1.0 / Math.sqrt(t * t + 1.0f));
                        float s = t * c;
                        float tau = s / (1.0f + c);

                        // Update diagonal
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + p, app - t * apq);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) q * n + q, aqq + t * apq);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + q, 0.0f);
                        aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) q * n + p, 0.0f);

                        // Rotate other elements in A
                        for (int r = 0; r < n; r++) {
                            if (r != p && r != q) {
                                float arp = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + p);
                                float arq = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + q);

                                float newArp = arp - s * (arq + tau * arp);
                                float newArq = arq + s * (arp - tau * arq);

                                aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + p, newArp);
                                aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) p * n + r, newArp);
                                aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + q, newArq);
                                aSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) q * n + r, newArq);
                            }
                        }

                        // Accumulate into eigenvector matrix V
                        for (int r = 0; r < n; r++) {
                            float vrp = vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + p);
                            float vrq = vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + q);

                            vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + p, vrp - s * (vrq + tau * vrp));
                            vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + q, vrq + s * (vrp - tau * vrq));
                        }
                    }
                }
            }

            // Extract eigenvalues from diagonal
            for (int i = 0; i < n; i++) {
                wSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * n + i));
            }

            // Sort eigenvalues and corresponding eigenvector columns in ascending order
            for (int i = 0; i < n - 1; i++) {
                int minIdx = i;
                float minVal = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                for (int j = i + 1; j < n; j++) {
                    float val = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                    if (val < minVal) {
                        minVal = val;
                        minIdx = j;
                    }
                }
                if (minIdx != i) {
                    wSeg.setAtIndex(ValueLayout.JAVA_FLOAT, minIdx, wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                    wSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, minVal);

                    for (int r = 0; r < n; r++) {
                        float tmp = vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + i);
                        vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + i, vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + minIdx));
                        vSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (long) r * n + minIdx, tmp);
                    }
                }
            }
        }
        return new EighResult(wArr, vArr);
    }

    public static EighResult eighDouble(NDArray a, Arena arena) {
        int n = (int) a.internalShapeUnsafe()[0];
        NDArray wArr = JNum.zeros(arena, DType.f64, n);
        NDArray vArr = JNum.zeros(arena, DType.f64, n, n);
        MemorySegment wSeg = wArr.getData();
        MemorySegment vSeg = vArr.getData();

        for (int i = 0; i < n; i++) {
            vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i, 1.0);
        }

        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment aSeg = scratch.allocate((long) n * n * BYTES_F64, 64);
            NDArray contig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
            MemorySegment.copy(contig.getData(), 0, aSeg, 0, (long) n * n * BYTES_F64);

            int maxSweeps = 50;
            double eps = 1e-15;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                double offDiagSum = 0.0;
                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        offDiagSum += Math.abs(aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + q));
                    }
                }

                if (offDiagSum < eps) {
                    break;
                }

                for (int p = 0; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        double apq = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + q);
                        if (Math.abs(apq) < 1e-20) continue;

                        double app = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p);
                        double aqq = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) q * n + q);

                        double theta = (aqq - app) / (2.0 * apq);
                        double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                        if (Double.isNaN(t)) t = 0.0;

                        double c = 1.0 / Math.sqrt(t * t + 1.0);
                        double s = t * c;
                        double tau = s / (1.0 + c);

                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + p, app - t * apq);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) q * n + q, aqq + t * apq);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + q, 0.0);
                        aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) q * n + p, 0.0);

                        for (int r = 0; r < n; r++) {
                            if (r != p && r != q) {
                                double arp = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + p);
                                double arq = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + q);

                                double newArp = arp - s * (arq + tau * arp);
                                double newArq = arq + s * (arp - tau * arq);

                                aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + p, newArp);
                                aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) p * n + r, newArp);
                                aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + q, newArq);
                                aSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) q * n + r, newArq);
                            }
                        }

                        for (int r = 0; r < n; r++) {
                            double vrp = vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + p);
                            double vrq = vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + q);

                            vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + p, vrp - s * (vrq + tau * vrp));
                            vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + q, vrq + s * (vrp - tau * vrq));
                        }
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                wSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * n + i));
            }

            for (int i = 0; i < n - 1; i++) {
                int minIdx = i;
                double minVal = wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                for (int j = i + 1; j < n; j++) {
                    double val = wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, j);
                    if (val < minVal) {
                        minVal = val;
                        minIdx = j;
                    }
                }
                if (minIdx != i) {
                    wSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, minIdx, wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                    wSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, minVal);

                    for (int r = 0; r < n; r++) {
                        double tmp = vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + i);
                        vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + i, vSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + minIdx));
                        vSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) r * n + minIdx, tmp);
                    }
                }
            }
        }
        return new EighResult(wArr, vArr);
    }
}