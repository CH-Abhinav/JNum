package jnum.jnumutils;

import jnum.DType;

public class TypeUtil {

    public static DType promoteTypes(DType a, DType b) {
        if (a == DType.f64 || b == DType.f64) return DType.f64;
        if (a == DType.f32 || b == DType.f32) return DType.f32;
        if (a == DType.i32 || b == DType.i32) return DType.i32;
        return DType.bool;
    }

    public static DType scalarType(int value) {
        return DType.i32;
    }

    public static DType scalarType(float value) {
        return DType.f32;
    }

    public static DType scalarType(double value) {
        return DType.f64;
    }

    public static DType scalarType(boolean value) {
        return DType.bool;
    }
}
