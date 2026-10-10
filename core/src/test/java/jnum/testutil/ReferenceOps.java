package jnum.testutil;

public final class ReferenceOps {

    private ReferenceOps() {
        throw new AssertionError("Utility class");
    }

    public static float[] add(float[] a, float[] b) {
        float[] res = new float[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    public static float[] sub(float[] a, float[] b) {
        float[] res = new float[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    public static float[] mul(float[] a, float[] b) {
        float[] res = new float[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] * b[i];
        return res;
    }

    public static float[] div(float[] a, float[] b) {
        float[] res = new float[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] / b[i];
        return res;
    }

    public static double[] add(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    public static double[] sub(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    public static double[] mul(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] * b[i];
        return res;
    }

    public static double[] div(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] / b[i];
        return res;
    }

    public static int[] add(int[] a, int[] b) {
        int[] res = new int[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    public static int[] sub(int[] a, int[] b) {
        int[] res = new int[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    public static int[] mul(int[] a, int[] b) {
        int[] res = new int[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] * b[i];
        return res;
    }

    public static int[] div(int[] a, int[] b) {
        int[] res = new int[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] / b[i];
        return res;
    }

    public static double dot(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) sum += (double) a[i] * (double) b[i];
        return sum;
    }

    public static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    public static float[] matmul2D(float[] a, int m, int k, float[] b, int n) {
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                float aVal = a[i * k + p];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += aVal * b[p * n + j];
                }
            }
        }
        return c;
    }

    public static double[] matmul2D(double[] a, int m, int k, double[] b, int n) {
        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                double aVal = a[i * k + p];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += aVal * b[p * n + j];
                }
            }
        }
        return c;
    }

    public static double det2x2(double a, double b, double c, double d) {
        return a * d - b * c;
    }

    public static float[] matmulFloat(float[] a, float[] b, int m, int k, int n) {
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                float aVal = a[i * k + p];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += aVal * b[p * n + j];
                }
            }
        }
        return c;
    }

    public static double[] matmulDouble(double[] a, double[] b, int m, int k, int n) {
        return matmul2D(a, m, k, b, n);
    }
}
