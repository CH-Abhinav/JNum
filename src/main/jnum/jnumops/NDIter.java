package jnum.jnumops;

public class NDIter {
    public final int[] shape;
    public final int[] strides;
    public final int rank;
    public int[] coords;
    public final int[] backstrides;
    public int offset;
    public boolean hasNext;
    
    public NDIter(int[] shape,int[] strides){
        this.shape = shape;
        this.strides = strides;
        this.rank = shape.length;
        this.coords = new int[rank];
        this.backstrides = new int[rank];
        this.offset = 0;
        this.hasNext = true;
        for (int i = 0; i < rank; i++) {
            this.backstrides[i] = (shape[i] - 1) * strides[i];
        }
    }

    void next(){
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
}
