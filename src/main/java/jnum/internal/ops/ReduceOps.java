package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.reduce.*;

public class ReduceOps {

    //non instatitable utility class
    private ReduceOps(){
        throw new AssertionError();
    }

    public static double sumFloat(NDArray a){
        return Sum.sumFloat(a);
    }

    public static double sumInt(NDArray a){
        return Sum.sumInt(a);
    }

    public static double sumDouble(NDArray a){
        return Sum.sumDouble(a);
    }

    public static NDArray sumFloatAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumFloatAxis(a, axis, resArray);
    }

    public static NDArray sumIntAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumIntAxis(a, axis, resArray);
    }

    public static NDArray sumDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumDoubleAxis(a, axis, resArray);
    }

    public static double maxFloat(NDArray a){
        return Max.maxFloat(a);
    }

    public static NDArray maxFloatAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxFloatAxis(a, axis, resArray);
    }

    public static double maxInt(NDArray a) {
        return Max.maxInt(a);
    }

    public static NDArray maxIntAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxIntAxis(a, axis, resArray);
    }

    public static double maxDouble(NDArray a) {
        return Max.maxDouble(a);
    }

    public static NDArray maxDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxDoubleAxis(a, axis, resArray);
    }

    public static double minFloat(NDArray a){
        return Min.minFloat(a);
    }

    public static NDArray minFloatAxis(NDArray a,int axis,NDArray resArray){
        return Min.minFloatAxis(a, axis, resArray);
    }

    public static double minInt(NDArray a) {
        return Min.minInt(a);
    }

    public static NDArray minIntAxis(NDArray a,int axis,NDArray resArray){
        return Min.minIntAxis(a, axis, resArray);
    }

    public static double minDouble(NDArray a) {
        return Min.minDouble(a);
    }

    public static NDArray minDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Min.minDoubleAxis(a, axis, resArray);
    }

    public static double dotFloat(NDArray a,NDArray b){
        return Dot.dotFloat(a, b);
    }

    public static double dotInt(NDArray a,NDArray b){
        return Dot.dotInt(a, b);
    }

    public static double dotDouble(NDArray a,NDArray b){
        return Dot.dotDouble(a, b);
    }

}
