package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.unary.*;

public final class UnaryOps {
    
    private UnaryOps() {
        throw new AssertionError();
    }
    
    public static NDArray sqrtFloat(NDArray a, NDArray resArray) {
        return Sqrt.sqrtFloat(a, resArray);
    }
    
    public static NDArray sqrtDouble(NDArray a, NDArray resArray) {
        return Sqrt.sqrtDouble(a, resArray);
    }
    
    public static NDArray sqrtInt(NDArray a, NDArray resArray) {
        return Sqrt.sqrtInt(a, resArray);
    }
    
    public static NDArray absFloat(NDArray a, NDArray resArray) {
        return Abs.absFloat(a, resArray);
    }
    
    public static NDArray absDouble(NDArray a, NDArray resArray) {
        return Abs.absDouble(a, resArray);
    }
    
    public static NDArray absInt(NDArray a, NDArray resArray) {
        return Abs.absInt(a, resArray);
    }
    
    public static NDArray expFloat(NDArray a, NDArray resArray) {
        return Exp.expFloat(a, resArray);
    }
    
    public static NDArray expDouble(NDArray a, NDArray resArray) {
        return Exp.expDouble(a, resArray);
    }
    
    public static NDArray expInt(NDArray a, NDArray resArray) {
        return Exp.expInt(a, resArray);
    }
    
    public static NDArray logFloat(NDArray a, NDArray resArray) {
        return Log.logFloat(a, resArray);
    }
    
    public static NDArray logDouble(NDArray a, NDArray resArray) {
        return Log.logDouble(a, resArray);
    }
    
    public static NDArray logInt(NDArray a, NDArray resArray) {
        return Log.logInt(a, resArray);
    }
    
    public static NDArray log10Float(NDArray a, NDArray resArray) {
        return Log10.log10Float(a, resArray);
    }
    
    public static NDArray log10Double(NDArray a, NDArray resArray) {
        return Log10.log10Double(a, resArray);
    }
    
    public static NDArray log10Int(NDArray a, NDArray resArray) {
        return Log10.log10Int(a, resArray);
    }
    
    public static NDArray sigmoidFloat(NDArray a, NDArray resArray) {
        return Sigmoid.sigmoidFloat(a, resArray);
    }
    
    public static NDArray sigmoidDouble(NDArray a, NDArray resArray) {
        return Sigmoid.sigmoidDouble(a, resArray);
    }
    
    public static NDArray sigmoidInt(NDArray a, NDArray resArray) {
        return Sigmoid.sigmoidInt(a, resArray);
    }
}
