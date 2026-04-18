package jnum.jnumops;

import jnum.NDArray;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class LinalgOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VL = SPECIES.length();
    private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    // Non-instantiable utility class
    private LinalgOps() {
        throw new AssertionError();
    }

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        long M = a.shape[a.shape.length - 2];
        long K = a.shape[a.shape.length - 1];
        long N = b.shape[b.shape.length - 1];
        long aRowStride = a.strides[a.strides.length - 2];
        long aColStride = a.strides[a.strides.length - 1];
        long bRowStride = b.strides[b.strides.length - 2];
        long resRowStride = resArray.strides[resArray.strides.length - 2];

        for (long i = 0; i < M; i++) {
            long vaRowOffset = i * aRowStride;
            long vresRowOffset = i * resRowStride;
            for (long k = 0; k < K; k++) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, vaRowOffset + (k * aColStride));
                var vA = FloatVector.broadcast(SPECIES, valA);
                
                long vbRowOffset = k * bRowStride;
                long j = 0;
                long loopbound = N - (N % (VL * 2));
                for (; j < loopbound; j += VL * 2) {
                    var vB1 = FloatVector.fromMemorySegment(SPECIES, b.data, (vbRowOffset + j) * FLOAT_BYTES, ORDER);
                    var vB2 = FloatVector.fromMemorySegment(SPECIES, b.data, (vbRowOffset + j + VL) * FLOAT_BYTES, ORDER);
                    var vC1 = FloatVector.fromMemorySegment(SPECIES, resArray.data, (vresRowOffset + j) * FLOAT_BYTES, ORDER);
                    var vC2 = FloatVector.fromMemorySegment(SPECIES, resArray.data, (vresRowOffset + j + VL) * FLOAT_BYTES, ORDER);
                    vC1 = vA.fma(vB1, vC1);
                    vC2 = vA.fma(vB2, vC2);
                    vC1.intoMemorySegment(resArray.data, (vresRowOffset + j) * FLOAT_BYTES, ORDER);
                    vC2.intoMemorySegment(resArray.data, (vresRowOffset + j + VL) * FLOAT_BYTES, ORDER);
                }

                loopbound = SPECIES.loopBound(N);
                for (; j < loopbound; j += VL) {
                    var vB = FloatVector.fromMemorySegment(SPECIES, b.data, (vbRowOffset + j) * FLOAT_BYTES, ORDER);
                    var vC = FloatVector.fromMemorySegment(SPECIES, resArray.data, (vresRowOffset + j) * FLOAT_BYTES, ORDER);
                    vC = vA.fma(vB, vC);
                    vC.intoMemorySegment(resArray.data, (vresRowOffset + j) * FLOAT_BYTES, ORDER);
                }

                for (; j < N; j++) {
                    float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, vbRowOffset + j);
                    float valC = resArray.data.getAtIndex(ValueLayout.JAVA_FLOAT, vresRowOffset + j);
                    resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, vresRowOffset + j, (valA * valB) + valC);
                }
            }
        }
        return resArray;
    }
}