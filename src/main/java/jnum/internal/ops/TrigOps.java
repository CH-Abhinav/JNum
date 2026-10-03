package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.trig.*;

public class TrigOps {
    //non instatitable utility class
    private TrigOps(){
        throw new AssertionError();
    }

    public static NDArray sinFloat(NDArray a, NDArray resArray){
        return Sin.sinFloat(a, resArray);
    }

    public static NDArray sinDouble(NDArray a, NDArray resArray){
        return Sin.sinDouble(a, resArray);
    }

    public static NDArray sinInt(NDArray a, NDArray resArray){
        return Sin.sinInt(a, resArray);
    }

    public static NDArray cosFloat(NDArray a, NDArray resArray){
        return Cos.cosFloat(a, resArray);
    }

    public static NDArray cosDouble(NDArray a, NDArray resArray){
        return Cos.cosDouble(a, resArray);
    }

    public static NDArray cosInt(NDArray a, NDArray resArray){
        return Cos.cosInt(a, resArray);
    }

    public static NDArray tanFloat(NDArray a, NDArray resArray){
        return Tan.tanFloat(a, resArray);
    }

    public static NDArray tanDouble(NDArray a, NDArray resArray){
        return Tan.tanDouble(a, resArray);
    }

    public static NDArray tanInt(NDArray a, NDArray resArray){
        return Tan.tanInt(a, resArray);
    }

    public static NDArray cotFloat(NDArray a, NDArray resArray){
        return Cot.cotFloat(a, resArray);
    }

    public static NDArray cotDouble(NDArray a, NDArray resArray){
        return Cot.cotDouble(a, resArray);
    }

    public static NDArray cotInt(NDArray a, NDArray resArray){
        return Cot.cotInt(a, resArray);
    }

    public static NDArray sinhFloat(NDArray a, NDArray resArray){
        return Sinh.sinhFloat(a, resArray);
    }

    public static NDArray sinhDouble(NDArray a, NDArray resArray){
        return Sinh.sinhDouble(a, resArray);
    }

    public static NDArray sinhInt(NDArray a, NDArray resArray){
        return Sinh.sinhInt(a, resArray);
    }

    public static NDArray coshFloat(NDArray a, NDArray resArray){
        return Cosh.coshFloat(a, resArray);
    }

    public static NDArray coshDouble(NDArray a, NDArray resArray){
        return Cosh.coshDouble(a, resArray);
    }

    public static NDArray coshInt(NDArray a, NDArray resArray){
        return Cosh.coshInt(a, resArray);
    }

    public static NDArray tanhFloat(NDArray a, NDArray resArray){
        return Tanh.tanhFloat(a, resArray);
    }

    public static NDArray tanhDouble(NDArray a, NDArray resArray){
        return Tanh.tanhDouble(a, resArray);
    }

    public static NDArray tanhInt(NDArray a, NDArray resArray){
        return Tanh.tanhInt(a, resArray);
    }

    public static NDArray cothFloat(NDArray a, NDArray resArray){
        return Coth.cothFloat(a, resArray);
    }

    public static NDArray cothDouble(NDArray a, NDArray resArray){
        return Coth.cothDouble(a, resArray);
    }

    public static NDArray cothInt(NDArray a, NDArray resArray){
        return Coth.cothInt(a, resArray);
    }
}
