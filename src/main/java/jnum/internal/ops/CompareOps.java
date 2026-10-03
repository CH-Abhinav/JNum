package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.compare.*;

public class CompareOps {

    private CompareOps() {
        throw new AssertionError();
    }

    public static NDArray maximumFloat(NDArray a,NDArray b,NDArray resArray){
        return Maximum.maximumFloat(a, b, resArray);
    }

    public static NDArray maximumFloat(NDArray a,float b,NDArray resArray){
        return Maximum.maximumFloat(a, b, resArray);
    }

    public static NDArray maximumDouble(NDArray a,NDArray b,NDArray resArray){
        return Maximum.maximumDouble(a, b, resArray);
    }

    public static NDArray maximumDouble(NDArray a,double b,NDArray resArray){
        return Maximum.maximumDouble(a, b, resArray);
    }

    public static NDArray maximumInt(NDArray a,NDArray b,NDArray resArray){
        return Maximum.maximumInt(a, b, resArray);
    }

    public static NDArray maximumInt(NDArray a,int b,NDArray resArray){
        return Maximum.maximumInt(a, b, resArray);
    }

    public static NDArray minimumFloat(NDArray a,NDArray b,NDArray resArray){
        return Minimum.minimumFloat(a, b, resArray);
    }

    public static NDArray minimumFloat(NDArray a,float b,NDArray resArray){
        return Minimum.minimumFloat(a, b, resArray);
    }

    public static NDArray minimumDouble(NDArray a,NDArray b,NDArray resArray){
        return Minimum.minimumDouble(a, b, resArray);
    }

    public static NDArray minimumDouble(NDArray a,double b,NDArray resArray){
        return Minimum.minimumDouble(a, b, resArray);
    }

    public static NDArray minimumInt(NDArray a,NDArray b,NDArray resArray){
        return Minimum.minimumInt(a, b, resArray);
    }

    public static NDArray minimumInt(NDArray a,int b,NDArray resArray){
        return Minimum.minimumInt(a, b, resArray);
    }
}
