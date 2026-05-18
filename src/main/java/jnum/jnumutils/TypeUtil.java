package jnum.jnumutils;

import jnum.DType;

public class TypeUtil {

    public static DType promoteTypes(DType a, DType b) {
        if (a == DType.DOUBLE || b == DType.DOUBLE) return DType.DOUBLE;
        if (a == DType.FLOAT || b == DType.FLOAT) return DType.FLOAT;
        return DType.INTEGER;
    }

    public static DType scalarType(int value) {
        return DType.INTEGER;
    }

    public static DType scalarType(float value) {
        return DType.FLOAT;
    }

    public static DType scalarType(double value) {
        return DType.DOUBLE;
    }
}
