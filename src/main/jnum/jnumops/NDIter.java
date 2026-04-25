package jnum.jnumops;

public class NDIter {
    public final int[] shape;
    public final int[] strides;
    public final int rank;
    public int[] coords;
    public int offset;
    public boolean hasNext;
    
    public NDIter(int[] shape,int[] strides){
        this.shape = shape;
        this.strides = strides;
        this.rank = shape.length;
        this.coords = new int[rank];
        this.offset = 0;
        this.hasNext = true;
    }

    void next(){
        int last=rank-1;
        while(last>=0){
            coords[last]++;
            if(coords[last]==shape[last]){
                coords[last]=0;
                last--;
            }
            else{
                break;
            }
        }
        if(last<0){
            hasNext=false;
        }
        else{
            offset=0;
            for(int i=0;i<rank;i++){
                offset+=coords[i]*strides[i];
            }
        }
    }

}
