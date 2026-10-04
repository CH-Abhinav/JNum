package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

public final class Norm {

    private Norm() {
        throw new AssertionError("Norm kernel cannot be instantiated.");
    }

    public static double norm(NDArray a, int ord) {
        long n = a.getSize();
        if (n == 0) return 0.0;

        return switch (a.getDType()) {
            case f32 -> normFloat(a, ord);
            case f64 -> normDouble(a, ord);
            default -> normDouble(a.cast(jnum.DType.f64), ord);
        };
    }

    public static float normFloat(NDArray a, int ord) {
        long n = a.getSize();
        MemorySegment seg = a.getData();

        if (a.isContiguous()) {
            if (ord == 2) {
                // 4x unrolled Euclidean norm
                FloatVector acc0 = FloatVector.zero(SPECIES_F32);
                FloatVector acc1 = FloatVector.zero(SPECIES_F32);
                FloatVector acc2 = FloatVector.zero(SPECIES_F32);
                FloatVector acc3 = FloatVector.zero(SPECIES_F32);

                long i = 0;
                long unrollBound = n - (n % (VL_F32 * 4));
                for (; i < unrollBound; i += VL_F32 * 4) {
                    FloatVector v0 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 0L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v1 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 1L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v2 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 2L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v3 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 3L * VL_F32) * BYTES_F32, NATIVE_ORDER);

                    acc0 = acc0.add(v0.mul(v0));
                    acc1 = acc1.add(v1.mul(v1));
                    acc2 = acc2.add(v2.mul(v2));
                    acc3 = acc3.add(v3.mul(v3));
                }

                FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
                long loopBound = SPECIES_F32.loopBound(n);
                for (; i < loopBound; i += VL_F32) {
                    FloatVector v = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER);
                    acc = acc.add(v.mul(v));
                }

                float sumSq = acc.reduceLanes(VectorOperators.ADD);
                for (; i < n; i++) {
                    float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    sumSq += val * val;
                }
                return (float) Math.sqrt(sumSq);

            } else if (ord == 1) {
                // 4x unrolled L1 norm
                FloatVector acc0 = FloatVector.zero(SPECIES_F32);
                FloatVector acc1 = FloatVector.zero(SPECIES_F32);
                FloatVector acc2 = FloatVector.zero(SPECIES_F32);
                FloatVector acc3 = FloatVector.zero(SPECIES_F32);

                long i = 0;
                long unrollBound = n - (n % (VL_F32 * 4));
                for (; i < unrollBound; i += VL_F32 * 4) {
                    FloatVector v0 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 0L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v1 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 1L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v2 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 2L * VL_F32) * BYTES_F32, NATIVE_ORDER);
                    FloatVector v3 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + 3L * VL_F32) * BYTES_F32, NATIVE_ORDER);

                    acc0 = acc0.add(v0.abs());
                    acc1 = acc1.add(v1.abs());
                    acc2 = acc2.add(v2.abs());
                    acc3 = acc3.add(v3.abs());
                }

                FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);
                long loopBound = SPECIES_F32.loopBound(n);
                for (; i < loopBound; i += VL_F32) {
                    FloatVector v = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER);
                    acc = acc.add(v.abs());
                }

                float sum = acc.reduceLanes(VectorOperators.ADD);
                for (; i < n; i++) {
                    sum += Math.abs(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                }
                return sum;

            } else if (ord == Integer.MAX_VALUE) {
                float max = 0.0f;
                for (long i = 0; i < n; i++) {
                    max = Math.max(max, Math.abs(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
                }
                return max;
            }
        } else {
            // Non-contiguous fallback using NDIter
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            if (ord == 2) {
                double sumSq = 0.0;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    float v = seg.get(ValueLayout.JAVA_FLOAT, byteOffset);
                    sumSq += (double) v * v;
                    iter.next();
                }
                return (float) Math.sqrt(sumSq);
            } else if (ord == 1) {
                float sum = 0.0f;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    sum += Math.abs(seg.get(ValueLayout.JAVA_FLOAT, byteOffset));
                    iter.next();
                }
                return sum;
            } else if (ord == Integer.MAX_VALUE) {
                float max = 0.0f;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    max = Math.max(max, Math.abs(seg.get(ValueLayout.JAVA_FLOAT, byteOffset)));
                    iter.next();
                }
                return max;
            }
        }
        throw new UnsupportedOperationException("Unsupported norm order: " + ord);
    }

    public static double normDouble(NDArray a, int ord) {
        long n = a.getSize();
        MemorySegment seg = a.getData();

        if (a.isContiguous()) {
            if (ord == 2) {
                // 4x unrolled DoubleVector Euclidean norm
                DoubleVector acc0 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc1 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc2 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc3 = DoubleVector.zero(SPECIES_F64);

                long i = 0;
                long unrollBound = n - (n % (VL_F64 * 4));
                for (; i < unrollBound; i += VL_F64 * 4) {
                    DoubleVector v0 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 0L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v1 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 1L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v2 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 2L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v3 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 3L * VL_F64) * BYTES_F64, NATIVE_ORDER);

                    acc0 = acc0.add(v0.mul(v0));
                    acc1 = acc1.add(v1.mul(v1));
                    acc2 = acc2.add(v2.mul(v2));
                    acc3 = acc3.add(v3.mul(v3));
                }

                DoubleVector acc = acc0.add(acc1).add(acc2).add(acc3);
                long loopBound = SPECIES_F64.loopBound(n);
                for (; i < loopBound; i += VL_F64) {
                    DoubleVector v = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER);
                    acc = acc.add(v.mul(v));
                }

                double sumSq = acc.reduceLanes(VectorOperators.ADD);
                for (; i < n; i++) {
                    double val = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    sumSq += val * val;
                }
                return Math.sqrt(sumSq);

            } else if (ord == 1) {
                // 4x unrolled L1 norm
                DoubleVector acc0 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc1 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc2 = DoubleVector.zero(SPECIES_F64);
                DoubleVector acc3 = DoubleVector.zero(SPECIES_F64);

                long i = 0;
                long unrollBound = n - (n % (VL_F64 * 4));
                for (; i < unrollBound; i += VL_F64 * 4) {
                    DoubleVector v0 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 0L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v1 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 1L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v2 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 2L * VL_F64) * BYTES_F64, NATIVE_ORDER);
                    DoubleVector v3 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + 3L * VL_F64) * BYTES_F64, NATIVE_ORDER);

                    acc0 = acc0.add(v0.abs());
                    acc1 = acc1.add(v1.abs());
                    acc2 = acc2.add(v2.abs());
                    acc3 = acc3.add(v3.abs());
                }

                DoubleVector acc = acc0.add(acc1).add(acc2).add(acc3);
                long loopBound = SPECIES_F64.loopBound(n);
                for (; i < loopBound; i += VL_F64) {
                    DoubleVector v = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER);
                    acc = acc.add(v.abs());
                }

                double sum = acc.reduceLanes(VectorOperators.ADD);
                for (; i < n; i++) {
                    sum += Math.abs(seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                }
                return sum;

            } else if (ord == Integer.MAX_VALUE) {
                double max = 0.0;
                for (long i = 0; i < n; i++) {
                    max = Math.max(max, Math.abs(seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i)));
                }
                return max;
            }
        } else {
            // Non-contiguous fallback using NDIter
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            if (ord == 2) {
                double sumSq = 0.0;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    double v = seg.get(ValueLayout.JAVA_DOUBLE, byteOffset);
                    sumSq += v * v;
                    iter.next();
                }
                return Math.sqrt(sumSq);
            } else if (ord == 1) {
                double sum = 0.0;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    sum += Math.abs(seg.get(ValueLayout.JAVA_DOUBLE, byteOffset));
                    iter.next();
                }
                return sum;
            } else if (ord == Integer.MAX_VALUE) {
                double max = 0.0;
                while (iter.hasNext) {
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                    max = Math.max(max, Math.abs(seg.get(ValueLayout.JAVA_DOUBLE, byteOffset)));
                    iter.next();
                }
                return max;
            }
        }
        throw new UnsupportedOperationException("Unsupported norm order: " + ord);
    }
}