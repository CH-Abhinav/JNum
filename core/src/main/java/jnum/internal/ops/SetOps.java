package jnum.internal.ops;

import java.lang.foreign.ValueLayout;
import jnum.NDArray;

public final class SetOps {
    private SetOps() { throw new AssertionError(); }

    public static void set(NDArray a, double val, long... indices){
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
        switch(a.getDType()){
            case f32 -> a.getData().setAtIndex(ValueLayout.JAVA_FLOAT, flatIndex, (float) val);
            case i32 -> a.getData().setAtIndex(ValueLayout.JAVA_INT, flatIndex, (int) val);
            case f64 -> a.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex,val);
            case bool -> a.getData().setAtIndex(ValueLayout.JAVA_BYTE, flatIndex, (byte) (val!=0?1:0));
            default -> throw new UnsupportedOperationException("This dtype "+a.getDType()+" doesn't support this method");
        }
    }

    public static void setFloat(NDArray a, float val, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_FLOAT, x * a.internalStridesUnsafe()[0] * Float.BYTES, val);
    }
    public static void setFloat(NDArray a, float val, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_FLOAT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Float.BYTES, val);
    }
    public static void setFloat(NDArray a, float val, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_FLOAT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Float.BYTES, val);
    }
    public static void setFloat(NDArray a, float val,long... indices){
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
        a.getData().setAtIndex(ValueLayout.JAVA_FLOAT, flatIndex, val);
    }

    public static void setDouble(NDArray a, double val, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_DOUBLE, x * a.internalStridesUnsafe()[0] * Double.BYTES, val);
    }
    public static void setDouble(NDArray a, double val, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_DOUBLE, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Double.BYTES, val);
    }
    public static void setDouble(NDArray a, double val, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_DOUBLE, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Double.BYTES, val);
    }
    public static void setDouble(NDArray a, double val,long... indices){
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
        a.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex, val);
    }

    public static void setInt(NDArray a, int val, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_INT, x * a.internalStridesUnsafe()[0] * Integer.BYTES, val);
    }
    public static void setInt(NDArray a, int val, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_INT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1]) * Integer.BYTES, val);
    }
    public static void setInt(NDArray a, int val, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_INT, (x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2]) * Integer.BYTES, val);
    }
    public static void setInt(NDArray a, int val,long... indices){
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
        a.getData().setAtIndex(ValueLayout.JAVA_INT, flatIndex, val);
    }

    public static void setBoolean(NDArray a, boolean val, long x) {
        long dim0 = a.internalShapeUnsafe()[0];
        if (x < 0) x += dim0;
        if (x < 0 || x >= dim0) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0], (byte) (val ? 1 : 0));
    }
    public static void setBoolean(NDArray a, boolean val, long x, long y) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1];
        if (x < 0) x += dim0; if (y < 0) y += dim1;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1], (byte) (val ? 1 : 0));
    }
    public static void setBoolean(NDArray a, boolean val, long x, long y, long z) {
        long dim0 = a.internalShapeUnsafe()[0], dim1 = a.internalShapeUnsafe()[1], dim2 = a.internalShapeUnsafe()[2];
        if (x < 0) x += dim0; if (y < 0) y += dim1; if (z < 0) z += dim2;
        if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2) throw new IndexOutOfBoundsException("Index out of bounds");
        a.getData().set(ValueLayout.JAVA_BYTE, x * a.internalStridesUnsafe()[0] + y * a.internalStridesUnsafe()[1] + z * a.internalStridesUnsafe()[2], (byte) (val ? 1 : 0));
    }
    public static void setBoolean(NDArray a, boolean val,long... indices){
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
        a.getData().setAtIndex(ValueLayout.JAVA_BYTE, flatIndex, (byte) (val ? 1 : 0));
    }
}
