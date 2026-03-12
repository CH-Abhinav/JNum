package jnum;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class NDArray{
    private final MemorySegment data;
    private final int[] shape;
    private final int[] strider;
    private final long size;

    private NDArray(MemorySegment data,int[] shape,int[] strider){
        this.data=data;
        this.shape=shape.clone();
        this.strider=strider.clone();
        long CalcSize=1;
        for (int dim : shape) CalcSize *= dim;
        this.size = CalcSize;
    }

    public static NDArray zeros(int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(ValueLayout.JAVA_FLOAT,Size);
        return new NDArray(segment,shape,calculateDefaultStrides(shape));
    }

    public static NDArray ones(int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(ValueLayout.JAVA_FLOAT,Size);
        for(long i=0;i<Size;i++){
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        return new NDArray(segment, shape, calculateDefaultStrides(shape));
    }

    private static int[] calculateDefaultStrides(int[] shape){
        int[] strider=new int[shape.length];
        int currentstride=1;
        for(int i=shape.length-1;i>=0;i--){
            strider[i]=currentstride;
            currentstride*=shape[i];
        }
        return strider;
    }

    public String shapeString() {
        return Arrays.toString(shape).replace("[", "(").replace("]", ")");
    }

    public float getFlat(long index) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

}