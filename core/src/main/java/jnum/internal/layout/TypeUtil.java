package jnum.internal.layout;

import jnum.DType;

/**
 * Type promotion rules and primitive type resolution utilities following NumPy type ladder semantics.
 */
public final class TypeUtil {

    private TypeUtil() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    /**
     * Resolves the common promoted data type between two operand types.
     * Follows the promotion order: bool &lt; i32 &lt; f32 &lt; f64.
     */
    public static DType promoteTypes(DType a, DType b) {
        if (a == DType.f64 || b == DType.f64) return DType.f64;
        if (a == DType.f32 || b == DType.f32) return DType.f32;
        if (a == DType.i32 || b == DType.i32) return DType.i32;
        return DType.bool;
    }

    /** Maps an integer primitive scalar to {@link DType#i32}. */
    public static DType scalarType(int value) {
        return DType.i32;
    }

    /** Maps a float primitive scalar to {@link DType#f32}. */
    public static DType scalarType(float value) {
        return DType.f32;
    }

    /** Maps a double primitive scalar to {@link DType#f64}. */
    public static DType scalarType(double value) {
        return DType.f64;
    }

    /** Maps a boolean primitive scalar to {@link DType#bool}. */
    public static DType scalarType(boolean value) {
        return DType.bool;
    }
}
