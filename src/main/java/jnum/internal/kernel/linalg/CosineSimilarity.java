package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.DType;
import jnum.NDArray;

public final class CosineSimilarity {

    private CosineSimilarity() {
        throw new AssertionError("CosineSimilarity kernel cannot be instantiated.");
    }

    public static double compute(NDArray a, NDArray b) {
        if (a.dim() != 1 || b.dim() != 1) {
            throw new IllegalArgumentException("Vector cosine similarity requires 1D arrays, got: " +
                    a.shapeString() + " and " + b.shapeString());
        }
        if (a.getSize() != b.getSize()) {
            throw new IllegalArgumentException("Vectors must have the same length: " +
                    a.getSize() + " vs " + b.getSize());
        }

        long n = a.getSize();
        if (n == 0) return 0.0;

        DType dtype = a.getDType() == DType.f32 && b.getDType() == DType.f32 ? DType.f32 : DType.f64;
        if (dtype == DType.f32) {
            return computeFloat(a, b, n);
        } else {
            return computeDouble(a, b, n);
        }
    }

    private static float computeFloat(NDArray a, NDArray b, long n) {
        NDArray aContig = (a.getDType() == DType.f32 && a.isContiguous()) ? a : a.cast(DType.f32).contiguous();
        NDArray bContig = (b.getDType() == DType.f32 && b.isContiguous()) ? b : b.cast(DType.f32).contiguous();

        MemorySegment segA = aContig.getData();
        MemorySegment segB = bContig.getData();

        FloatVector vDot = FloatVector.zero(SPECIES_F32);
        FloatVector vNormA = FloatVector.zero(SPECIES_F32);
        FloatVector vNormB = FloatVector.zero(SPECIES_F32);

        long i = 0;
        long bound = SPECIES_F32.loopBound(n);

        // Single-pass SIMD: computes dot, normA^2, and normB^2 in one go
        for (; i < bound; i += VL_F32) {
            FloatVector va = FloatVector.fromMemorySegment(SPECIES_F32, segA, i * BYTES_F32, NATIVE_ORDER);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES_F32, segB, i * BYTES_F32, NATIVE_ORDER);

            vDot   = vDot.add(va.mul(vb));
            vNormA = vNormA.add(va.mul(va));
            vNormB = vNormB.add(vb.mul(vb));
        }

        float dot   = vDot.reduceLanes(VectorOperators.ADD);
        float normA = vNormA.reduceLanes(VectorOperators.ADD);
        float normB = vNormB.reduceLanes(VectorOperators.ADD);

        for (; i < n; i++) {
            float va = segA.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = segB.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dot   += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (denom < 1e-12f) return 0.0f;
        return dot / denom;
    }

    private static double computeDouble(NDArray a, NDArray b, long n) {
        NDArray aContig = (a.getDType() == DType.f64 && a.isContiguous()) ? a : a.cast(DType.f64).contiguous();
        NDArray bContig = (b.getDType() == DType.f64 && b.isContiguous()) ? b : b.cast(DType.f64).contiguous();

        MemorySegment segA = aContig.getData();
        MemorySegment segB = bContig.getData();

        DoubleVector vDot = DoubleVector.zero(SPECIES_F64);
        DoubleVector vNormA = DoubleVector.zero(SPECIES_F64);
        DoubleVector vNormB = DoubleVector.zero(SPECIES_F64);

        long i = 0;
        long bound = SPECIES_F64.loopBound(n);

        for (; i < bound; i += VL_F64) {
            DoubleVector va = DoubleVector.fromMemorySegment(SPECIES_F64, segA, i * BYTES_F64, NATIVE_ORDER);
            DoubleVector vb = DoubleVector.fromMemorySegment(SPECIES_F64, segB, i * BYTES_F64, NATIVE_ORDER);

            vDot   = vDot.add(va.mul(vb));
            vNormA = vNormA.add(va.mul(va));
            vNormB = vNormB.add(vb.mul(vb));
        }

        double dot   = vDot.reduceLanes(VectorOperators.ADD);
        double normA = vNormA.reduceLanes(VectorOperators.ADD);
        double normB = vNormB.reduceLanes(VectorOperators.ADD);

        for (; i < n; i++) {
            double va = segA.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = segB.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            dot   += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        if (denom < 1e-15) return 0.0;
        return dot / denom;
    }
}