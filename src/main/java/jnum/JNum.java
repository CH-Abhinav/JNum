package jnum;

import jnum.jnumutils.ShapeUtil;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class JNum {

    private JNum(){
        throw new AssertionError("JNum facade cannot be instantiated.");
    }

    public static NDArray zeros(long... shape){
        return zeros(Arena.ofAuto(),DType.f32,shape);
    }

    public static NDArray zeros(Arena arena,long... shape){
        return zeros(arena,DType.f32, shape);
    }

    public static NDArray zeros(DType dType,long... shape){
        return zeros(Arena.ofAuto(),dType,shape);
    }

    public static NDArray zeros(Arena arena,DType dType,long... shape){
        long Size=1;
        for(long dim:shape) Size*=dim;
        long byteSize=Size*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        return NDArray.ofRaw(segment,shape, ShapeUtil.calculateDefaultStrides(shape),dType);
    }

    //TODO : ones method is using naive loops. need to optimise later
    public static NDArray ones(long... shape){
        return ones(Arena.ofAuto(),DType.f32,shape);
    }

    public static NDArray ones(DType dType,long... shape){
        return ones(Arena.ofAuto(),dType,shape);
    }

    public static NDArray ones(Arena arena,long...shape){
        return ones(arena,DType.f32, shape);
    }

    public static NDArray ones(Arena arena,DType dType,long... shape){
        long Size=1;
        for(long dim:shape) Size*=dim;
        long byteSize=Size*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        switch (dType) {
            case i32 -> {
                for(long i=0;i<Size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, 1);
                }
            }
            case f32 ->{
                for(long i=0;i<Size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
                }
            }
            case f64 -> {
                for(long i=0;i<Size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.0);
                }
            }
            case bool -> {
                for(long i=0;i<Size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) 1);
                }
            }
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape),dType);
    }

    public static NDArray from(float[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, float[] data, long... shape) {
        long CalcSize = 1;
        for (long dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                    "Requested shape " + Arrays.toString(shape) +
                            " requires size " + CalcSize +
                            ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.f32;
        long byteSize=CalcSize*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    //int array from methods
    public static NDArray from(int[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, int[] data, long... shape) {
        long CalcSize = 1;
        for (long dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                    "Requested shape " + Arrays.toString(shape) +
                            " requires size " + CalcSize +
                            ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.i32;
        long byteSize=CalcSize*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    // DOUBLE array from method
    public static NDArray from(double[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, double[] data, long... shape) {
        long CalcSize = 1;
        for (long dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                    "Requested shape " + Arrays.toString(shape) +
                            " requires size " + CalcSize +
                            ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.f64;
        long byteSize=CalcSize*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    // boolean array from methods
    public static NDArray from(boolean[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, boolean[] data, long... shape) {
        long CalcSize = 1;
        for (long dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                    "Requested shape " + Arrays.toString(shape) +
                            " requires size " + CalcSize +
                            ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.bool;
        long byteSize=CalcSize*dType.layout.byteSize();
        MemorySegment segment=arena.allocate(byteSize,64);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (data[i] ? 1 : 0));
        }
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    //TODO: should we add more rand methods?
    //TODO: no randi() for in place random values
    //TODO: should we move rand methods to a different file? (NO for now)
    public static NDArray rand(long... shape){
        NDArray resArray=JNum.zeros(shape);
        for(long i = 0; i< resArray.getSize(); i++){
            resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
        }
        return resArray;
    }

    public static NDArray rand(DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
            case bool ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (ThreadLocalRandom.current().nextBoolean() ? 1 : 0));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(Arena arena,DType dType,long... shape){
        NDArray resArray=JNum.zeros(arena,dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
            case bool ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (ThreadLocalRandom.current().nextBoolean() ? 1 : 0));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(int max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(max));
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(float max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                        "rand(float max, DType, shape) cannot generate FLOAT random values into dtype " +
                                dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(double max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                        "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                                dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                        "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                                dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(int min,int max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case i32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(min, max));
            }}
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(float min,float max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case i32 ->throw new IllegalArgumentException(
                    "rand(float min, float max, DType, shape) cannot generate FLOAT random values into dtype " +
                            dType + " for shape " + Arrays.toString(shape)
            );
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray rand(double min,double max,DType dType,long... shape){
        NDArray resArray=JNum.zeros(dType, shape);
        switch(dType){
            case f32 ->throw new IllegalArgumentException(
                    "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                            dType + " for shape " + Arrays.toString(shape)
            );
            case i32 ->throw new IllegalArgumentException(
                    "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                            dType + " for shape " + Arrays.toString(shape)
            );
            case f64 ->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
            default -> throw new UnsupportedOperationException("This dtype "+dType+" doesn't support this method");
        }
        return resArray;
    }

    public static NDArray arange(double stop) {
        return arange(0.0, stop, 1.0, DType.f32, Arena.ofAuto());
    }

    public static NDArray arange(double start, double stop) {
        return arange(start, stop, 1.0, DType.f32, Arena.ofAuto());
    }

    public static NDArray arange(double start, double stop, double step) {
        return arange(start, stop, step, DType.f32, Arena.ofAuto());
    }

    public static NDArray arange(double start, double stop, double step, DType dType, Arena arena) {
        if (step == 0) {
            throw new IllegalArgumentException("Step cannot be zero.");
        }

        long size = (long) Math.ceil((stop - start) / step);
        if (size <= 0) {
            // Return an empty 1D array if bounds are invalid
            return NDArray.ofRaw(arena.allocate(0), new long[]{0}, new long[]{1}, dType);
        }

        long byteSize = size * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64); // 64-byte aligned for SIMD

        switch (dType) {
            case f32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (start + i * step));
                }
            }
            case f64 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, start + i * step);
                }
            }
            case i32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, (int) (start + i * step));
                }
            }
            case bool -> throw new IllegalArgumentException("arange is not supported for boolean type.");
            default -> throw new UnsupportedOperationException("Unsupported dtype: " + dType);
        }

        return NDArray.ofRaw(segment, new long[]{size}, new long[]{1}, dType);
    }
}
