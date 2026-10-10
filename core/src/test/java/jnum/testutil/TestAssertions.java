package jnum.testutil;

import jnum.DType;
import jnum.NDArray;
import org.junit.jupiter.api.Assertions;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public final class TestAssertions {

    private TestAssertions() {
        throw new AssertionError("Utility class");
    }

    public static void assertClose(float expected, float actual, float absTol, float relTol) {
        if (Float.isNaN(expected)) {
            assertTrue(Float.isNaN(actual), () -> "Expected NaN but got " + actual);
            return;
        }
        if (Float.isInfinite(expected)) {
            assertEquals(expected, actual, () -> "Expected infinity " + expected + " but got " + actual);
            return;
        }
        float diff = Math.abs(expected - actual);
        float tol = Math.max(absTol, relTol * Math.abs(expected));
        assertTrue(diff <= tol,
            () -> String.format("Expected %f but got %f (diff: %e > tol: %e, absTol: %e, relTol: %e)",
                expected, actual, diff, tol, absTol, relTol));
    }

    public static void assertClose(double expected, double actual, double absTol, double relTol) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), () -> "Expected NaN but got " + actual);
            return;
        }
        if (Double.isInfinite(expected)) {
            assertEquals(expected, actual, () -> "Expected infinity " + expected + " but got " + actual);
            return;
        }
        double diff = Math.abs(expected - actual);
        double tol = Math.max(absTol, relTol * Math.abs(expected));
        assertTrue(diff <= tol,
            () -> String.format("Expected %f but got %f (diff: %e > tol: %e, absTol: %e, relTol: %e)",
                expected, actual, diff, tol, absTol, relTol));
    }

    public static void assertArrayClose(float[] expected, float[] actual, float absTol, float relTol) {
        assertEquals(expected.length, actual.length, "Array lengths differ");
        for (int i = 0; i < expected.length; i++) {
            final int idx = i;
            assertClose(expected[i], actual[i], absTol, relTol);
        }
    }

    public static void assertArrayClose(double[] expected, double[] actual, double absTol, double relTol) {
        assertEquals(expected.length, actual.length, "Array lengths differ");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], absTol, relTol);
        }
    }

    public static void assertNDArrayClose(NDArray expected, NDArray actual, double absTol, double relTol) {
        assertArrayEquals(expected.getShape(), actual.getShape(),
            () -> "Shapes mismatch: expected " + Arrays.toString(expected.getShape()) +
                  " but got " + Arrays.toString(actual.getShape()));
        assertEquals(expected.getDType(), actual.getDType(), "DTypes mismatch");

        long total = expected.getSize();
        DType dtype = expected.getDType();

        for (long i = 0; i < total; i++) {
            final long idx = i;
            switch (dtype) {
                case f32 -> assertClose(expected.getFlatFloat(idx), actual.getFlatFloat(idx), (float) absTol, (float) relTol);
                case f64 -> assertClose(expected.getFlatDouble(idx), actual.getFlatDouble(idx), absTol, relTol);
                case i32 -> assertEquals(expected.getFlatInt(idx), actual.getFlatInt(idx),
                    () -> "Mismatch at flat index " + idx);
                case bool -> assertEquals(expected.getFlatBoolean(idx), actual.getFlatBoolean(idx),
                    () -> "Mismatch at flat index " + idx);
            }
        }
    }

    public static void assertSignedZero(float val, boolean expectPositiveZero) {
        int bits = Float.floatToRawIntBits(val);
        if (expectPositiveZero) {
            assertEquals(0, bits, "Expected +0.0f but got " + val);
        } else {
            assertEquals(0x80000000, bits, "Expected -0.0f but got " + val);
        }
    }

    public static void assertSignedZero(double val, boolean expectPositiveZero) {
        long bits = Double.doubleToRawLongBits(val);
        if (expectPositiveZero) {
            assertEquals(0L, bits, "Expected +0.0 but got " + val);
        } else {
            assertEquals(0x8000000000000000L, bits, "Expected -0.0 but got " + val);
        }
    }
}
