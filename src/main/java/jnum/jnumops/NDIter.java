package jnum.jnumops;

import jnum.jnumutils.ShapeUtil;

public class NDIter {
    public final long[] shape;
    public final long[] strides;
    public final int rank;
    public long[] coords;
    public final long[] backstrides;
    public long offset;
    public boolean hasNext;

    public NDIter(long[] shape){
        this(shape, ShapeUtil.calculateDefaultStrides(shape));
    }
    
    public NDIter(long[] shape,long[] strides){
        this.shape = shape;
        this.strides = strides;
        this.rank = shape.length;
        this.coords = new long[rank];
        this.backstrides = new long[rank];
        this.offset = 0;
        this.hasNext = true;
        for (int i = 0; i < rank; i++) {
            this.backstrides[i] = (shape[i] - 1) * strides[i];
        }
    }

    public void next(){
        int last=rank-1;
        while(last>=0){
            coords[last]++;
            if(coords[last]==shape[last]){
                coords[last]=0;
                offset -= backstrides[last];
                last--;
            }
            else{
                offset += strides[last];
                break;
            }
        }
        if(last<0){
            hasNext=false;
        }
    }

    public int nextVector(long[] indexMap,int vl){
        int count=0;
        while(this.hasNext && count<vl){
            indexMap[count++]=this.offset;
            this.next();
        }
        return count;
    }
}
