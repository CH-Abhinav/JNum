package jnum.internal.ops;

import java.lang.foreign.ValueLayout;
import jnum.NDArray;

public final class GetOps {
    private GetOps() { throw new AssertionError(); }

    public static long getPhysicalOffset(long logicalIndex, long[] shape, long[] strides) {
        long remaining = logicalIndex;
        long offset = 0;
        for (int i = shape.length - 1; i >= 0; i--) {
            long coord = (remaining % shape[i]);
            remaining /= shape[i];
            offset += coord * strides[i];
        }
        return offset;
    }

    private static void validateFlatIndex(NDArray a, long index) {
        if (index < 0 || index >= a.getSize()) {
            throw new IndexOutOfBoundsException("Flat index " + index + " is out of bounds for size " + a.getSize());
        }
    }

    public static double getFlat(NDArray a, long index){
        validateFlatIndex(a, index);
        long physicalOffset = a.isContiguous() ? index : getPhysicalOffset(index, a.internalShapeUnsafe(), a.internalStridesUnsafe());
        return switch(a.getDType()){
            case f32 -> a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset);
            case i32 -> a.getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset);
            case f64 -> a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset);
            case bool -> a.getData().getAtIndex(ValueLayout.JAVA_BYTE, physicalOffset) != 0 ? 1.0 : 0.0;
            default -> throw new UnsupportedOperationException("This dtype "+a.getDType()+" doesn't support this method");
        };
    }

    public static float getFlatFloat(NDArray a, long index) {
        validateFlatIndex(a, index);
        long physicalOffset = a.isContiguous() ? index : getPhysicalOffset(index, a.internalShapeUnsafe(), a.internalStridesUnsafe());
        return a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset);
    }

    public static int getFlatInt(NDArray a, long index){
        validateFlatIndex(a, index);
        long physicalOffset = a.isContiguous() ? index : getPhysicalOffset(index, a.internalShapeUnsafe(), a.internalStridesUnsafe());
        return a.getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset);
    }

    public static double getFlatDouble(NDArray a, long index){
        validateFlatIndex(a, index);
        long physicalOffset = a.isContiguous() ? index : getPhysicalOffset(index, a.internalShapeUnsafe(), a.internalStridesUnsafe());
        return a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset);
    }

    public static boolean getFlatBoolean(NDArray a, long index) {
        validateFlatIndex(a, index);
        long physicalOffset = a.isContiguous() ? index : getPhysicalOffset(index, a.internalShapeUnsafe(), a.internalStridesUnsafe());
        return a.getData().getAtIndex(ValueLayout.JAVA_BYTE, physicalOffset) != 0;
    }

    public static double get(NDArray a, long... indices){
        if(indices.length!= a.internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ a.internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= a.internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + a.internalShapeUnsafe()[i]);
            }
            flatIndex+=indices[i]* a.internalStridesUnsafe()[i];
        }
        return switch(a.getDType()){
            case f32 -> a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex);
            case i32 -> a.getData().getAtIndex(ValueLayout.JAVA_INT, flatIndex);
            case f64 -> a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex);
            case bool -> a.getData().getAtIndex(ValueLayout.JAVA_BYTE, flatIndex) != 0 ? 1.0 : 0.0;
            default -> throw new UnsupportedOperationException("This dtype "+a.getDType()+" doesn't support this method");
        };
    }

    // --- FLOAT ---
    public static float getFloat(NDArray a, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_FLOAT, x * a.internalStridesUnsafe()[0] * Float.BYTES);
    }
    public static float getFloat(NDArray a, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_FLOAT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Float.BYTES);
    }
    public static float getFloat(NDArray a, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_FLOAT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Float.BYTES);
    }
    public static float getFloat(NDArray a, long... indices){
        if(indices.length!= a.internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ a.internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= a.internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + a.internalShapeUnsafe()[i]);
            }
            flatIndex+=indices[i]* a.internalStridesUnsafe()[i];
        }
        return a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex);
    }

    // --- DOUBLE ---
    public static double getDouble(NDArray a, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_DOUBLE, x * a.internalStridesUnsafe()[0] * Double.BYTES);
    }
    public static double getDouble(NDArray a, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_DOUBLE, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Double.BYTES);
    }
    public static double getDouble(NDArray a, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_DOUBLE, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Double.BYTES);
    }
    public static double getDouble(NDArray a, long... indices){
        if(indices.length!= a.internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ a.internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= a.internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + a.internalShapeUnsafe()[i]);
            }
            flatIndex+=indices[i]* a.internalStridesUnsafe()[i];
        }
        return a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex);
    }

    // --- INT ---
    public static int getInt(NDArray a, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_INT, x * a.internalStridesUnsafe()[0] * Integer.BYTES);
    }
    public static int getInt(NDArray a, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_INT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Integer.BYTES);
    }
    public static int getInt(NDArray a, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_INT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Integer.BYTES);
    }

    // --- BOOLEAN ---
    public static boolean getBoolean(NDArray a, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0]) != 0;
    }
    public static boolean getBoolean(NDArray a, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) != 0;
    }
    public static boolean getBoolean(NDArray a, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        return a.getData().get(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) != 0;
    }
}
