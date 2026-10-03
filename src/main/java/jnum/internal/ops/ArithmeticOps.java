package jnum.internal.ops;

import jnum.NDArray;
import jnum.internal.kernel.arithmetic.*;

public class ArithmeticOps {
    private ArithmeticOps() {
        throw new AssertionError();
    }

    public static NDArray addFloat(NDArray a, NDArray b, NDArray resArray) {
        return Add.addFloat(a, b, resArray);
    }

    public static NDArray addFloat(NDArray a, float b, NDArray resArray) {
        return Add.addFloat(a, b, resArray);
    }

    public static NDArray addDouble(NDArray a, NDArray b, NDArray resArray) {
        return Add.addDouble(a, b, resArray);
    }

    public static NDArray addDouble(NDArray a, double b, NDArray resArray) {
        return Add.addDouble(a, b, resArray);
    }

    public static NDArray addInt(NDArray a, NDArray b, NDArray resArray) {
        return Add.addInt(a, b, resArray);
    }

    public static NDArray addInt(NDArray a, int b, NDArray resArray) {
        return Add.addInt(a, b, resArray);
    }

    public static NDArray subFloat(NDArray a, NDArray b, NDArray resArray) {
        return Sub.subFloat(a, b, resArray);
    }

    public static NDArray subFloat(NDArray a, float b, NDArray resArray) {
        return Sub.subFloat(a, b, resArray);
    }

    public static NDArray subDouble(NDArray a, NDArray b, NDArray resArray) {
        return Sub.subDouble(a, b, resArray);
    }

    public static NDArray subDouble(NDArray a, double b, NDArray resArray) {
        return Sub.subDouble(a, b, resArray);
    }

    public static NDArray subInt(NDArray a, NDArray b, NDArray resArray) {
        return Sub.subInt(a, b, resArray);
    }

    public static NDArray subInt(NDArray a, int b, NDArray resArray) {
        return Sub.subInt(a, b, resArray);
    }

    public static NDArray mulFloat(NDArray a, NDArray b, NDArray resArray) {
        return Mul.mulFloat(a, b, resArray);
    }

    public static NDArray mulFloat(NDArray a, float b, NDArray resArray) {
        return Mul.mulFloat(a, b, resArray);
    }

    public static NDArray mulDouble(NDArray a, NDArray b, NDArray resArray) {
        return Mul.mulDouble(a, b, resArray);
    }

    public static NDArray mulDouble(NDArray a, double b, NDArray resArray) {
        return Mul.mulDouble(a, b, resArray);
    }

    public static NDArray mulInt(NDArray a, NDArray b, NDArray resArray) {
        return Mul.mulInt(a, b, resArray);
    }

    public static NDArray mulInt(NDArray a, int b, NDArray resArray) {
        return Mul.mulInt(a, b, resArray);
    }

    public static NDArray divFloat(NDArray a, NDArray b, NDArray resArray) {
        return Div.divFloat(a, b, resArray);
    }

    public static NDArray divFloat(NDArray a, float b, NDArray resArray) {
        return Div.divFloat(a, b, resArray);
    }

    public static NDArray divDouble(NDArray a, NDArray b, NDArray resArray) {
        return Div.divDouble(a, b, resArray);
    }

    public static NDArray divDouble(NDArray a, double b, NDArray resArray) {
        return Div.divDouble(a, b, resArray);
    }

    public static NDArray divInt(NDArray a, NDArray b, NDArray resArray) {
        return Div.divInt(a, b, resArray);
    }

    public static NDArray divInt(NDArray a, int b, NDArray resArray) {
        return Div.divInt(a, b, resArray);
    }

}
