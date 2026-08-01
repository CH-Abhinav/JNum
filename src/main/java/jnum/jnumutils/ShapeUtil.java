package jnum.jnumutils;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import jnum.DType;

public class ShapeUtil {

    public static long[] calculateDefaultStrides(long[] shape){
        long[] strides=new long[shape.length];
        long currentstride=1;
        for(int i=shape.length-1;i>=0;i--){
            strides[i]=currentstride;
            currentstride*=shape[i];
        }
        return strides;
    }

    public static long[] calculateBroadcastShape(long[] shapeA, long[] shapeB){
        int maxDims=Math.max(shapeA.length, shapeB.length);
        long[] result=new long[maxDims];
        for (int i = 1; i <= maxDims; i++) {
            long dimA = (shapeA.length - i >= 0) ? shapeA[shapeA.length - i] : 1;
            long dimB = (shapeB.length - i >= 0) ? shapeB[shapeB.length - i] : 1;

            if (dimA == dimB) {
                result[maxDims - i] = dimA;
            } else if (dimA == 1) {
                result[maxDims - i] = dimB;
            } else if (dimB == 1) {
                result[maxDims - i] = dimA;
            } else {
                throw new IllegalArgumentException("Shapes " + Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB) + " are not broadcastable.");
            }
        }
        return result;
    }

    public static long[] calculateReductionShape(long[] shape, int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for shape " + Arrays.toString(shape));
        }
        long[] reducedShape = new long[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reducedShape[j++] = shape[i];
            }
        }
        return reducedShape;
    }

    public static long getByteOffset(long[] coords, long[] strides, DType dtype) {
        long elementOffset = 0;
        for (int i = 0; i < coords.length; i++) {
            elementOffset += coords[i] * strides[i];
        }
        return elementOffset * dtype.layout.byteSize();
    }


}
