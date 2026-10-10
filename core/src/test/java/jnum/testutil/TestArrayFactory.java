package jnum.testutil;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

import java.util.Random;

public final class TestArrayFactory {

    private TestArrayFactory() {
        throw new AssertionError("Utility class");
    }

    public static NDArray random(long seed, DType dtype, long... shape) {
        Random rng = new Random(seed);
        long size = 1;
        for (long s : shape) size *= s;

        switch (dtype) {
            case f32 -> {
                float[] data = new float[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = rng.nextFloat() * 10.0f - 5.0f;
                }
                return JNum.from(data, shape);
            }
            case f64 -> {
                double[] data = new double[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = rng.nextDouble() * 10.0 - 5.0;
                }
                return JNum.from(data, shape);
            }
            case i32 -> {
                int[] data = new int[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = rng.nextInt(200) - 100;
                }
                return JNum.from(data, shape);
            }
            case bool -> {
                boolean[] data = new boolean[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = rng.nextBoolean();
                }
                return JNum.from(data, shape);
            }
            default -> throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
    }

    public static NDArray hilbert(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i * n + j] = 1.0 / (i + j + 1.0);
            }
        }
        return JNum.from(data, (long) n, (long) n);
    }

    public static NDArray singular(int n) {
        if (n < 2) {
            return JNum.zeros(DType.f64, n, n);
        }
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i * n + j] = (i + 1) * (j + 1);
            }
        }
        // Duplicate row 0 into row 1 to guarantee rank deficiency / zero determinant
        System.arraycopy(data, 0, data, n, n);
        return JNum.from(data, (long) n, (long) n);
    }

    public static NDArray spd(int n) {
        Random rng = new Random(42L);
        double[] b = new double[n * n];
        for (int i = 0; i < b.length; i++) {
            b[i] = rng.nextDouble() * 2.0 - 1.0;
        }
        // Compute B * B^T + n * I to guarantee symmetric positive definite
        double[] a = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += b[i * n + k] * b[j * n + k];
                }
                if (i == j) {
                    sum += n;
                }
                a[i * n + j] = sum;
            }
        }
        return JNum.from(a, (long) n, (long) n);
    }

    public static NDArray boundaryArray(DType dtype, long... shape) {
        long size = 1;
        for (long s : shape) size *= s;

        switch (dtype) {
            case f32 -> {
                float[] special = {
                    0.0f, -0.0f, 1.0f, -1.0f,
                    Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                    Float.MIN_VALUE, Float.MAX_VALUE, Float.MIN_NORMAL
                };
                float[] data = new float[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = special[i % special.length];
                }
                return JNum.from(data, shape);
            }
            case f64 -> {
                double[] special = {
                    0.0, -0.0, 1.0, -1.0,
                    Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                    Double.MIN_VALUE, Double.MAX_VALUE, Double.MIN_NORMAL
                };
                double[] data = new double[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = special[i % special.length];
                }
                return JNum.from(data, shape);
            }
            case i32 -> {
                int[] special = {
                    0, 1, -1,
                    Integer.MIN_VALUE, Integer.MAX_VALUE,
                    42, -42, 1000, -1000
                };
                int[] data = new int[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = special[i % special.length];
                }
                return JNum.from(data, shape);
            }
            case bool -> {
                boolean[] data = new boolean[(int) size];
                for (int i = 0; i < data.length; i++) {
                    data[i] = (i % 2 == 0);
                }
                return JNum.from(data, shape);
            }
            default -> throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
    }

    public static NDArray eye(int n, DType dtype) {
        NDArray res = JNum.zeros(dtype, n, n);
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case f32 -> res.setFloat(1.0f, i, i);
                case f64 -> res.setDouble(1.0, i, i);
                case i32 -> res.setInt(1, i, i);
                case bool -> res.setBoolean(true, i, i);
            }
        }
        return res;
    }

    public static NDArray matrix(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, flat, r * cols, cols);
        }
        return JNum.from(flat, rows, cols);
    }

    public static NDArray matrix(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, flat, r * cols, cols);
        }
        return JNum.from(flat, rows, cols);
    }

    public static NDArray matrix(int[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        int[] flat = new int[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, flat, r * cols, cols);
        }
        return JNum.from(flat, rows, cols);
    }

    public static NDArray matrix(boolean[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        boolean[] flat = new boolean[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, flat, r * cols, cols);
        }
        return JNum.from(flat, rows, cols);
    }

    public static float[] toFloatArray(NDArray arr) {
        float[] flat = new float[(int) arr.getSize()];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = arr.getFlatFloat(i);
        }
        return flat;
    }

    public static double[] toDoubleArray(NDArray arr) {
        double[] flat = new double[(int) arr.getSize()];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = arr.getFlatDouble(i);
        }
        return flat;
    }
}
