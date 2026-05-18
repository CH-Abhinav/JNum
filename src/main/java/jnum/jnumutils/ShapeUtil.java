package jnum.jnumutils;

import java.util.Arrays;

public class ShapeUtil {

    public static int[] calculateDefaultStrides(int[] shape){
        int[] strides=new int[shape.length];
        int currentstride=1;
        for(int i=shape.length-1;i>=0;i--){
            strides[i]=currentstride;
            currentstride*=shape[i];
        }
        return strides;
    }

    public static int[] calculateBroadcastShape(int[] shapeA, int[] shapeB){
        int maxDims=Math.max(shapeA.length, shapeB.length);
        int[] result=new int[maxDims];
        for (int i = 1; i <= maxDims; i++) {
            int dimA = (shapeA.length - i >= 0) ? shapeA[shapeA.length - i] : 1;
            int dimB = (shapeB.length - i >= 0) ? shapeB[shapeB.length - i] : 1;

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

    public static int[] calculateReductionShape(int[] shape, int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for shape " + Arrays.toString(shape));
        }
        int[] reducedShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reducedShape[j++] = shape[i];
            }
        }
        return reducedShape;
    }
}
