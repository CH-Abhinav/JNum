package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.nio.ByteOrder;


public class NDArray{
    private final MemorySegment data;
    private final int[] shape;
    private final int[] strides;
    private final long size;

    private NDArray(MemorySegment data,int[] shape,int[] strides){
        this.data=data;
        this.shape=shape.clone();
        this.strides=strides.clone();
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
        int[] strides=new int[shape.length];
        int currentstride=1;
        for(int i=shape.length-1;i>=0;i--){
            strides[i]=currentstride;
            currentstride*=shape[i];
        }
        return strides;
    }

    public String shapeString() {
        return Arrays.toString(shape).replace("[", "(").replace("]", ")");
    }

    public float getFlat(long index) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder("NDArray" + shapeString() + " [");
        for (long i = 0; i < size; i++) {
            sb.append(data.getAtIndex(ValueLayout.JAVA_FLOAT, i));
            if (i < size - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    public NDArray add(NDArray b){
        if(this.size!=b.size) throw new IllegalArgumentException("Shape mismatch detected:cannot add NDArrays with diffrent shapes");
        Arena arena=Arena.ofAuto();
        MemorySegment res_seg= arena.allocate(ValueLayout.JAVA_FLOAT,this.size);

        VectorSpecies<Float> SPECIES= FloatVector.SPECIES_PREFERRED;
        long i=0;
        long loopbound=SPECIES.loopBound(this.size);

        for(;i<loopbound;i+=SPECIES.length()){
            var v1=FloatVector.fromMemorySegment(SPECIES,this.data,i*4,ByteOrder.nativeOrder());
            var v2=FloatVector.fromMemorySegment(SPECIES,b.data,i*4,ByteOrder.nativeOrder());
            var Vres=v1.add(v2);
            Vres.intoMemorySegment(res_seg,i*4,ByteOrder.nativeOrder());
        }

        for (; i < this.size; i++) {
            var val1 = this.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            res_seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 + val2);
        }
        return new NDArray(res_seg, this.shape, this.strides);
    }
}