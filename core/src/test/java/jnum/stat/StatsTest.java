package jnum.stat;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.stat.Average;
import jnum.internal.kernel.stat.Mean;
import jnum.internal.kernel.stat.Median;
import jnum.internal.kernel.stat.Ptp;
import jnum.internal.kernel.stat.Quantile;
import jnum.internal.kernel.stat.Std;
import jnum.internal.kernel.stat.Var;
import jnum.internal.ops.StatOps;
import jnum.testutil.TestArrayFactory;

public class StatsTest {

    @Test
    @DisplayName("Private constructors throw AssertionError")
    void testPrivateConstructors() throws Exception {
        assertPrivateConstructor(Stats.class);
        assertPrivateConstructor(StatOps.class);
        assertPrivateConstructor(Mean.class);
        assertPrivateConstructor(Average.class);
        assertPrivateConstructor(Var.class);
        assertPrivateConstructor(Std.class);
        assertPrivateConstructor(Ptp.class);
        assertPrivateConstructor(Median.class);
        assertPrivateConstructor(Quantile.class);
    }

    private void assertPrivateConstructor(Class<?> clazz) throws Exception {
        Constructor<?> constructor = clazz.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    // =========================================================================
    // MEAN & AVERAGE TESTS
    // =========================================================================

    @Test
    @DisplayName("Global mean for f32, f64, i32, and bool")
    void testGlobalMean() {
        NDArray fArr = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
        assertEquals(2.5, Stats.mean(fArr), 1e-6);

        NDArray dArr = JNum.from(new double[]{10.0, 20.0, 30.0}, 3);
        assertEquals(20.0, Stats.mean(dArr), 1e-12);

        NDArray iArr = JNum.from(new int[]{1, 2, 3, 4, 5}, 5);
        assertEquals(3.0, Stats.mean(iArr), 1e-6);

        NDArray bArr = JNum.from(new boolean[]{true, false, true, true}, 4);
        assertEquals(0.75, Stats.mean(bArr), 1e-6);
    }

    @Test
    @DisplayName("Global mean on empty array returns NaN")
    void testEmptyArrayMean() {
        NDArray empty = JNum.zeros(DType.f32, 0);
        assertTrue(Double.isNaN(Stats.mean(empty)));
    }

    @Test
    @DisplayName("Global mean on large array triggering SIMD and parallelism")
    void testLargeArrayMean() {
        int n = 100_000;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = 2.0f;
        }
        NDArray a = JNum.from(data, n);
        assertEquals(2.0, Stats.mean(a), 1e-5);

        double[] dataD = new double[n];
        for (int i = 0; i < n; i++) {
            dataD[i] = (double) i;
        }
        NDArray aD = JNum.from(dataD, n);
        double expected = (n - 1) / 2.0;
        assertEquals(expected, Stats.mean(aD), 1e-6);
    }

    @Test
    @DisplayName("Mean on non-contiguous array")
    void testNonContiguousMean() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });
        NDArray transposed = m.transpose();
        assertFalse(transposed.isContiguous());
        assertEquals(3.5, Stats.mean(transposed), 1e-6);
    }

    @Test
    @DisplayName("Axis mean on 2D matrix (axis 0, 1, and negative axes)")
    void testAxisMean2D() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });

        // Axis 0 (down columns): [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        NDArray mean0 = Stats.mean(m, 0);
        assertArrayEquals(new long[]{3}, mean0.getShape());
        assertEquals(2.5f, mean0.getFloat(0), 1e-6f);
        assertEquals(3.5f, mean0.getFloat(1), 1e-6f);
        assertEquals(4.5f, mean0.getFloat(2), 1e-6f);

        // Negative axis: axis -2 is axis 0
        NDArray meanNeg2 = Stats.mean(m, -2);
        assertArrayEquals(new long[]{3}, meanNeg2.getShape());
        assertEquals(2.5f, meanNeg2.getFloat(0), 1e-6f);

        // Axis 1 (across rows): [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        NDArray mean1 = Stats.mean(m, 1);
        assertArrayEquals(new long[]{2}, mean1.getShape());
        assertEquals(2.0f, mean1.getFloat(0), 1e-6f);
        assertEquals(5.0f, mean1.getFloat(1), 1e-6f);

        // Negative axis: axis -1 is axis 1
        NDArray meanNeg1 = Stats.mean(m, -1);
        assertArrayEquals(new long[]{2}, meanNeg1.getShape());
        assertEquals(2.0f, meanNeg1.getFloat(0), 1e-6f);
        assertEquals(5.0f, meanNeg1.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Axis mean with keepDims=true")
    void testAxisMeanKeepDims() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });

        NDArray mean0Keep = Stats.mean(m, 0, true);
        assertArrayEquals(new long[]{1, 3}, mean0Keep.getShape());
        assertEquals(2.5f, mean0Keep.getFloat(0, 0), 1e-6f);

        NDArray mean1Keep = Stats.mean(m, 1, true);
        assertArrayEquals(new long[]{2, 1}, mean1Keep.getShape());
        assertEquals(2.0f, mean1Keep.getFloat(0, 0), 1e-6f);
        assertEquals(5.0f, mean1Keep.getFloat(1, 0), 1e-6f);
    }

    @Test
    @DisplayName("Axis mean for double and integer matrices")
    void testAxisMeanTypes() {
        NDArray mD = JNum.from(new double[]{1.0, 3.0, 5.0, 7.0}, 2, 2);
        NDArray meanD = Stats.mean(mD, 0);
        assertEquals(DType.f64, meanD.getDType());
        assertEquals(3.0, meanD.getDouble(0), 1e-12);
        assertEquals(5.0, meanD.getDouble(1), 1e-12);

        NDArray mI = JNum.from(new int[]{1, 2, 3, 4}, 2, 2);
        NDArray meanI = Stats.mean(mI, 1);
        assertEquals(DType.f64, meanI.getDType());
        assertEquals(1.5, meanI.getDouble(0), 1e-6);
        assertEquals(3.5, meanI.getDouble(1), 1e-6);
    }

    @Test
    @DisplayName("Axis mean on 3D tensor")
    void testAxisMean3D() {
        NDArray t = JNum.from(new float[]{
            1f, 2f,
            3f, 4f,

            5f, 6f,
            7f, 8f
        }, 2, 2, 2);

        NDArray mean0 = Stats.mean(t, 0);
        assertArrayEquals(new long[]{2, 2}, mean0.getShape());
        assertEquals(3.0f, mean0.getFloat(0, 0), 1e-6f);
        assertEquals(4.0f, mean0.getFloat(0, 1), 1e-6f);
        assertEquals(5.0f, mean0.getFloat(1, 0), 1e-6f);
        assertEquals(6.0f, mean0.getFloat(1, 1), 1e-6f);
    }

    @Test
    @DisplayName("Invalid axis throws IllegalArgumentException")
    void testInvalidAxis() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        assertThrows(IllegalArgumentException.class, () -> Stats.mean(a, 1));
        assertThrows(IllegalArgumentException.class, () -> Stats.mean(a, -2));
    }

    @Test
    @DisplayName("Weighted average (scalar)")
    void testWeightedAverageScalar() {
        NDArray a = JNum.from(new float[]{10f, 20f, 30f, 40f}, 4);
        assertEquals(25.0, Stats.average(a, null), 1e-6);
        assertEquals(25.0, Stats.average(a), 1e-6);

        NDArray uniform = JNum.from(new float[]{1f, 1f, 1f, 1f}, 4);
        assertEquals(25.0, Stats.average(a, uniform), 1e-6);

        NDArray w = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
        assertEquals(30.0, Stats.average(a, w), 1e-6);

        NDArray wrongW = JNum.from(new float[]{1f, 2f}, 2);
        assertThrows(IllegalArgumentException.class, () -> Stats.average(a, wrongW));
    }

    @Test
    @DisplayName("Weighted average along axis with 1D weights")
    void testWeightedAverageAxis1DWeights() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {10f, 20f},
            {30f, 40f}
        });
        NDArray w0 = JNum.from(new float[]{1f, 3f}, 2);
        NDArray avg0 = Stats.average(m, 0, w0);
        assertArrayEquals(new long[]{2}, avg0.getShape());
        assertEquals(25f, avg0.getFloat(0), 1e-6f);
        assertEquals(35f, avg0.getFloat(1), 1e-6f);

        NDArray w1 = JNum.from(new float[]{1f, 3f}, 2);
        NDArray avg1 = Stats.average(m, 1, w1, true);
        assertArrayEquals(new long[]{2, 1}, avg1.getShape());
        assertEquals(17.5f, avg1.getFloat(0, 0), 1e-6f);
        assertEquals(37.5f, avg1.getFloat(1, 0), 1e-6f);
    }

    // =========================================================================
    // VARIANCE & STD TESTS
    // =========================================================================

    @Test
    @DisplayName("Global variance and std for f32, f64, and i32")
    void testGlobalVarStd() {
        // [1, 2, 3, 4] -> mean = 2.5
        // diffs: -1.5, -0.5, 0.5, 1.5 -> sq: 2.25, 0.25, 0.25, 2.25 -> sumSq = 5.0
        // var (ddof=0) = 5.0 / 4 = 1.25; std = sqrt(1.25) approx 1.1180339887
        // var (ddof=1) = 5.0 / 3 = 1.6666666667; std = sqrt(5/3) approx 1.2909944
        NDArray fArr = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
        assertEquals(1.25, Stats.var(fArr), 1e-6);
        assertEquals(5.0 / 3.0, Stats.sampleVar(fArr), 1e-6);
        assertEquals(5.0 / 3.0, Stats.varScalar(fArr, 1), 1e-6);
        assertEquals(Math.sqrt(1.25), Stats.std(fArr), 1e-6);
        assertEquals(Math.sqrt(5.0 / 3.0), Stats.sampleStd(fArr), 1e-6);
        assertEquals(Math.sqrt(5.0 / 3.0), Stats.stdScalar(fArr, 1), 1e-6);

        NDArray dArr = JNum.from(new double[]{10.0, 20.0, 30.0}, 3);
        // mean = 20 -> sumSq = 100 + 0 + 100 = 200
        // var(ddof=0) = 200/3 = 66.6666667; var(ddof=1) = 200/2 = 100.0
        assertEquals(200.0 / 3.0, Stats.var(dArr), 1e-12);
        assertEquals(100.0, Stats.sampleVar(dArr), 1e-12);
        assertEquals(100.0, Stats.varScalar(dArr, 1), 1e-12);
        assertEquals(Math.sqrt(200.0 / 3.0), Stats.std(dArr), 1e-12);
        assertEquals(10.0, Stats.sampleStd(dArr), 1e-12);

        NDArray iArr = JNum.from(new int[]{2, 4, 4, 4, 5, 5, 7, 9}, 8);
        // mean = 5.0 -> sq diffs: 9, 1, 1, 1, 0, 0, 4, 16 = 32 -> var(ddof=0) = 4.0, std = 2.0
        assertEquals(4.0, Stats.var(iArr), 1e-6);
        assertEquals(2.0, Stats.std(iArr), 1e-6);
        assertEquals(32.0 / 7.0, Stats.sampleVar(iArr), 1e-6);
    }

    @Test
    @DisplayName("Numerical stability of variance with large constant offset")
    void testVarStdShiftedNumericalStability() {
        // [10^9 + 1, 10^9 + 2, 10^9 + 3] has exact same variance as [1, 2, 3]!
        // var(ddof=0) = 2/3, var(ddof=1) = 1.0
        double offset = 1e9;
        NDArray shifted = JNum.from(new double[]{offset + 1.0, offset + 2.0, offset + 3.0}, 3);
        assertEquals(2.0 / 3.0, Stats.var(shifted), 1e-8);
        assertEquals(1.0, Stats.sampleVar(shifted), 1e-8);
        assertEquals(1.0, Stats.sampleStd(shifted), 1e-8);
    }

    @Test
    @DisplayName("Large array variance with SIMD and parallelism")
    void testLargeArrayVarStd() {
        int n = 100_000;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = 5.0f; // constant array -> variance must be exactly 0
        }
        NDArray a = JNum.from(data, n);
        assertEquals(0.0, Stats.var(a), 1e-6);
        assertEquals(0.0, Stats.std(a), 1e-6);
    }

    @Test
    @DisplayName("Variance on non-contiguous array")
    void testNonContiguousVar() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f},
            {3f, 4f}
        });
        NDArray trans = m.transpose();
        assertFalse(trans.isContiguous());
        // [1, 3, 2, 4] -> same 4 elements, var(ddof=0) = 1.25
        assertEquals(1.25, Stats.var(trans), 1e-6);
    }

    @Test
    @DisplayName("Axis variance and std on 2D matrix")
    void testAxisVarStd2D() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 4f},
            {3f, 8f}
        });

        // Axis 0: down columns
        // col 0: [1, 3] -> mean=2 -> sq: 1, 1 -> var=1.0, std=1.0
        // col 1: [4, 8] -> mean=6 -> sq: 4, 4 -> var=4.0, std=2.0
        NDArray var0 = Stats.var(m, 0);
        assertArrayEquals(new long[]{2}, var0.getShape());
        assertEquals(1.0f, var0.getFloat(0), 1e-6f);
        assertEquals(4.0f, var0.getFloat(1), 1e-6f);

        NDArray std0 = Stats.std(m, 0);
        assertEquals(1.0f, std0.getFloat(0), 1e-6f);
        assertEquals(2.0f, std0.getFloat(1), 1e-6f);

        // Axis 1: across rows with keepDims
        // row 0: [1, 4] -> mean=2.5 -> diff=1.5 -> var=2.25, std=1.5
        // row 1: [3, 8] -> mean=5.5 -> diff=2.5 -> var=6.25, std=2.5
        NDArray var1Keep = Stats.var(m, 1, 0, true);
        assertArrayEquals(new long[]{2, 1}, var1Keep.getShape());
        assertEquals(2.25f, var1Keep.getFloat(0, 0), 1e-6f);
        assertEquals(6.25f, var1Keep.getFloat(1, 0), 1e-6f);

        NDArray std1Keep = Stats.std(m, 1, 0, true);
        assertEquals(1.5f, std1Keep.getFloat(0, 0), 1e-6f);
        assertEquals(2.5f, std1Keep.getFloat(1, 0), 1e-6f);
    }

    @Test
    @DisplayName("Variance edge cases: size 1, ddof >= size, empty")
    void testVarEdgeCases() {
        NDArray single = JNum.from(new float[]{42f}, 1);
        assertEquals(0.0, Stats.var(single));
        assertTrue(Double.isNaN(Stats.sampleVar(single))); // divisor <= 0 -> NaN

        NDArray empty = JNum.zeros(DType.f32, 0);
        assertTrue(Double.isNaN(Stats.var(empty)));
        assertTrue(Double.isNaN(Stats.std(empty)));
    }

    // =========================================================================
    // PEAK-TO-PEAK (PTP: MAX - MIN) TESTS
    // =========================================================================

    @Test
    @DisplayName("Global ptp for f32, f64, and i32")
    void testGlobalPtp() {
        NDArray fArr = JNum.from(new float[]{1f, 5f, 2f, 9f, -3f}, 5);
        assertEquals(12.0, Stats.ptp(fArr), 1e-6); // 9 - (-3) = 12

        NDArray dArr = JNum.from(new double[]{10.5, -2.5, 3.0}, 3);
        assertEquals(13.0, Stats.ptp(dArr), 1e-12); // 10.5 - (-2.5) = 13

        NDArray iArr = JNum.from(new int[]{7, 100, -50}, 3);
        assertEquals(150.0, Stats.ptp(iArr), 1e-6);

        NDArray empty = JNum.zeros(DType.f32, 0);
        assertTrue(Double.isNaN(Stats.ptp(empty)));
    }

    @Test
    @DisplayName("Axis ptp on 2D matrix")
    void testAxisPtp2D() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 10f, 3f},
            {5f, 2f, 8f}
        });

        // Axis 0 (cols): [5-1, 10-2, 8-3] = [4, 8, 5]
        NDArray ptp0 = Stats.ptp(m, 0);
        assertArrayEquals(new long[]{3}, ptp0.getShape());
        assertEquals(4f, ptp0.getFloat(0), 1e-6f);
        assertEquals(8f, ptp0.getFloat(1), 1e-6f);
        assertEquals(5f, ptp0.getFloat(2), 1e-6f);

        // Axis 1 with keepDims: [10-1, 8-2] = [9, 6]
        NDArray ptp1Keep = Stats.ptp(m, 1, true);
        assertArrayEquals(new long[]{2, 1}, ptp1Keep.getShape());
        assertEquals(9f, ptp1Keep.getFloat(0, 0), 1e-6f);
        assertEquals(6f, ptp1Keep.getFloat(1, 0), 1e-6f);
    }

    // =========================================================================
    // MEDIAN, QUANTILE & IQR TESTS
    // =========================================================================

    @Test
    @DisplayName("Global median for odd and even length arrays")
    void testGlobalMedian() {
        // Odd length: [5, 1, 9, 3, 7] -> sorted [1, 3, 5, 7, 9] -> median = 5.0
        NDArray odd = JNum.from(new float[]{5f, 1f, 9f, 3f, 7f}, 5);
        assertEquals(5.0, Stats.median(odd), 1e-6);

        // Even length: [1, 2, 3, 4] -> median = (2 + 3) / 2 = 2.5
        NDArray even = JNum.from(new double[]{4.0, 1.0, 3.0, 2.0}, 4);
        assertEquals(2.5, Stats.median(even), 1e-12);

        // Integer array: [10, 2, 5, 8] -> sorted [2, 5, 8, 10] -> median = (5 + 8) / 2 = 6.5
        NDArray iArr = JNum.from(new int[]{10, 2, 5, 8}, 4);
        assertEquals(6.5, Stats.median(iArr), 1e-6);

        // Single element
        NDArray single = JNum.from(new float[]{42f}, 1);
        assertEquals(42.0, Stats.median(single), 1e-6);

        // Empty array
        NDArray empty = JNum.zeros(DType.f32, 0);
        assertTrue(Double.isNaN(Stats.median(empty)));
    }

    @Test
    @DisplayName("Axis median on 2D matrix")
    void testAxisMedian2D() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 5f, 9f},
            {4f, 2f, 6f},
            {7f, 8f, 3f}
        });

        // Axis 0: columns
        // col 0: [1, 4, 7] -> median = 4.0
        // col 1: [5, 2, 8] -> median = 5.0
        // col 2: [9, 6, 3] -> median = 6.0
        NDArray med0 = Stats.median(m, 0);
        assertArrayEquals(new long[]{3}, med0.getShape());
        assertEquals(4.0f, med0.getFloat(0), 1e-6f);
        assertEquals(5.0f, med0.getFloat(1), 1e-6f);
        assertEquals(6.0f, med0.getFloat(2), 1e-6f);

        // Axis 1: rows with keepDims
        // row 0: [1, 5, 9] -> median = 5.0
        // row 1: [4, 2, 6] -> median = 4.0
        // row 2: [7, 8, 3] -> median = 7.0
        NDArray med1Keep = Stats.median(m, 1, true);
        assertArrayEquals(new long[]{3, 1}, med1Keep.getShape());
        assertEquals(5.0f, med1Keep.getFloat(0, 0), 1e-6f);
        assertEquals(4.0f, med1Keep.getFloat(1, 0), 1e-6f);
        assertEquals(7.0f, med1Keep.getFloat(2, 0), 1e-6f);
    }

    @Test
    @DisplayName("Global quantile and percentile")
    void testGlobalQuantileAndPercentile() {
        NDArray a = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, 5);
        // q=0 is min, q=0.5 is median, q=1 is max
        assertEquals(1.0, Stats.quantile(a, 0.0), 1e-8);
        assertEquals(3.0, Stats.quantile(a, 0.5), 1e-8);
        assertEquals(5.0, Stats.quantile(a, 1.0), 1e-8);

        // Percentiles: 0%, 50%, 100%
        assertEquals(1.0, Stats.percentile(a, 0.0), 1e-8);
        assertEquals(3.0, Stats.percentile(a, 50.0), 1e-8);
        assertEquals(5.0, Stats.percentile(a, 100.0), 1e-8);

        // q=0.25 (p=25): index = 0.25 * 4 = 1.0 -> exactly element at 1 -> 2.0
        assertEquals(2.0, Stats.quantile(a, 0.25), 1e-8);
        assertEquals(2.0, Stats.percentile(a, 25.0), 1e-8);

        // q=0.75 (p=75): index = 0.75 * 4 = 3.0 -> exactly element at 3 -> 4.0
        assertEquals(4.0, Stats.quantile(a, 0.75), 1e-8);
        assertEquals(4.0, Stats.percentile(a, 75.0), 1e-8);

        // Out-of-bounds quantile throws
        assertThrows(IllegalArgumentException.class, () -> Stats.quantile(a, -0.1));
        assertThrows(IllegalArgumentException.class, () -> Stats.quantile(a, 1.1));
    }

    @Test
    @DisplayName("Quantile interpolation methods (linear, lower, higher, midpoint, nearest)")
    void testQuantileMethodsInterpolation() {
        // [10, 20, 30, 40], n = 4
        // q = 0.25 -> v = 0.25 * 3 = 0.75 -> i = 0 (10), j = 1 (20), f = 0.75
        NDArray a = JNum.from(new float[]{10f, 20f, 30f, 40f}, 4);

        // LINEAR: 10 + 0.75 * (20 - 10) = 17.5
        assertEquals(17.5, Stats.quantile(a, 0.25, QuantileMethod.LINEAR), 1e-6);

        // LOWER: 10.0
        assertEquals(10.0, Stats.quantile(a, 0.25, QuantileMethod.LOWER), 1e-6);

        // HIGHER: 20.0
        assertEquals(20.0, Stats.quantile(a, 0.25, QuantileMethod.HIGHER), 1e-6);

        // MIDPOINT: (10 + 20) / 2 = 15.0
        assertEquals(15.0, Stats.quantile(a, 0.25, QuantileMethod.MIDPOINT), 1e-6);

        // NEAREST: f = 0.75 > 0.5 -> 20.0
        assertEquals(20.0, Stats.quantile(a, 0.25, QuantileMethod.NEAREST), 1e-6);
    }

    @Test
    @DisplayName("Axis quantile and percentile on 2D matrix")
    void testAxisQuantile2D() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {10f, 20f, 30f, 40f},
            {1f,  2f,  3f,  4f}
        });

        // Along axis 1 (across cols), q = 0.5 (median)
        NDArray qMed1 = Stats.quantile(m, 0.5, 1);
        assertArrayEquals(new long[]{2}, qMed1.getShape());
        assertEquals(25.0f, qMed1.getFloat(0), 1e-6f);
        assertEquals(2.5f, qMed1.getFloat(1), 1e-6f);

        // With percentile and keepDims
        NDArray p75Keep = Stats.percentile(m, 75.0, 1, true);
        assertArrayEquals(new long[]{2, 1}, p75Keep.getShape());
        assertEquals(32.5f, p75Keep.getFloat(0, 0), 1e-6f);
        assertEquals(3.25f, p75Keep.getFloat(1, 0), 1e-6f);
    }

    @Test
    @DisplayName("Interquartile range (IQR)")
    void testIQR() {
        // [1, 2, 3, 4, 5] -> q75 = 4.0, q25 = 2.0 -> iqr = 2.0
        NDArray a = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, 5);
        assertEquals(2.0, Stats.iqr(a), 1e-6);

        // 2D matrix
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f, 4f, 5f},
            {10f, 20f, 30f, 40f, 50f}
        });
        NDArray iqr1 = Stats.iqr(m, 1);
        assertArrayEquals(new long[]{2}, iqr1.getShape());
        assertEquals(2.0f, iqr1.getFloat(0), 1e-6f);
        assertEquals(20.0f, iqr1.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Large array Introselect median (100,000 elements)")
    void testLargeArrayMedian() {
        int n = 100_001; // odd length
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = (float) i;
        }
        // Shuffle slightly by reversing
        for (int i = 0; i < n / 2; i++) {
            float tmp = data[i];
            data[i] = data[n - 1 - i];
            data[n - 1 - i] = tmp;
        }

        NDArray a = JNum.from(data, n);
        assertEquals(50000.0, Stats.median(a), 1e-4);
    }

    // =========================================================================
    // MODE TESTS
    // =========================================================================

    @Test
    @DisplayName("Scalar mode: unimodal, ties, empty")
    void testModeScalar() {
        // Unimodal float
        NDArray fArr = JNum.from(new float[]{1f, 2f, 2f, 3f}, 4);
        assertEquals(2.0, Stats.mode(fArr), 1e-6);

        // Unimodal double
        NDArray dArr = JNum.from(new double[]{10.0, 20.0, 20.0, 30.0}, 4);
        assertEquals(20.0, Stats.mode(dArr), 1e-12);

        // Unimodal int
        NDArray iArr = JNum.from(new int[]{5, 5, 5, 2, 1}, 5);
        assertEquals(5.0, Stats.mode(iArr), 1e-6);

        // Ties: smallest value chosen
        NDArray tieFloat = JNum.from(new float[]{3f, 1f, 3f, 1f}, 4);
        assertEquals(1.0, Stats.mode(tieFloat), 1e-6);

        NDArray tieInt = JNum.from(new int[]{4, 2, 4, 2, 6}, 5);
        assertEquals(2.0, Stats.mode(tieInt), 1e-6);

        // Single element
        NDArray single = JNum.from(new float[]{42f}, 1);
        assertEquals(42.0, Stats.mode(single), 1e-6);

        // Empty
        NDArray empty = JNum.zeros(DType.f32, 0);
        assertTrue(Double.isNaN(Stats.mode(empty)));
    }

    @Test
    @DisplayName("Axis mode on 2D matrix")
    void testModeAxis() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 2f},
            {3f, 3f, 4f}
        });

        // Axis 1: across rows
        // row 0: [1, 2, 2] -> mode = 2
        // row 1: [3, 3, 4] -> mode = 3
        NDArray mode1 = Stats.mode(m, 1);
        assertArrayEquals(new long[]{2}, mode1.getShape());
        assertEquals(2f, mode1.getFloat(0), 1e-6f);
        assertEquals(3f, mode1.getFloat(1), 1e-6f);

        // Axis 0: across columns
        // col 0: [1, 3] -> tie -> 1
        // col 1: [2, 3] -> tie -> 2
        // col 2: [2, 4] -> tie -> 2
        NDArray mode0 = Stats.mode(m, 0);
        assertArrayEquals(new long[]{3}, mode0.getShape());
        assertEquals(1f, mode0.getFloat(0), 1e-6f);
        assertEquals(2f, mode0.getFloat(1), 1e-6f);
        assertEquals(2f, mode0.getFloat(2), 1e-6f);
    }

    // =========================================================================
    // COVARIANCE TESTS
    // =========================================================================

    @Test
    @DisplayName("Pairwise 1D covariance: float, double, int, ddof")
    void testCovPairwise() {
        // x = [1, 2, 3, 4, 5], y = [2, 4, 6, 8, 10]
        // mean(x) = 3, mean(y) = 6
        // sum = (-2)(-4) + (-1)(-2) + 0 + (1)(2) + (2)(4) = 8 + 2 + 0 + 2 + 8 = 20
        // ddof = 1 -> 20 / 4 = 5.0
        // ddof = 0 -> 20 / 5 = 4.0
        NDArray x = JNum.from(new float[]{1f, 2f, 3f, 4f, 5f}, 5);
        NDArray y = JNum.from(new float[]{2f, 4f, 6f, 8f, 10f}, 5);

        assertEquals(5.0, Stats.cov(x, y), 1e-5);
        assertEquals(4.0, Stats.cov(x, y, 0), 1e-5);

        // cov(x, x) should equal sampleVar(x)
        assertEquals(Stats.sampleVar(x), Stats.cov(x, x), 1e-5);

        // Double array
        NDArray xd = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, 5);
        NDArray yd = JNum.from(new double[]{2.0, 4.0, 6.0, 8.0, 10.0}, 5);
        assertEquals(5.0, Stats.cov(xd, yd), 1e-10);
        assertEquals(4.0, Stats.cov(xd, yd, 0), 1e-10);

        // Int array
        NDArray xi = JNum.from(new int[]{1, 2, 3, 4, 5}, 5);
        NDArray yi = JNum.from(new int[]{2, 4, 6, 8, 10}, 5);
        assertEquals(5.0, Stats.cov(xi, yi), 1e-5);
    }

    @Test
    @DisplayName("2D covariance matrix: rowvar true and false")
    void testCovMatrix2D() {
        // 2 variables, 5 observations each
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f, 4f, 5f},
            {2f, 4f, 6f, 8f, 10f}
        });

        // rowvar = true
        NDArray cov = Stats.cov(m);
        assertArrayEquals(new long[]{2, 2}, cov.getShape());
        // Var(x) = 2.5, Cov(x, y) = 5.0, Var(y) = 10.0
        assertEquals(2.5f, cov.getFloat(0, 0), 1e-5f);
        assertEquals(5.0f, cov.getFloat(0, 1), 1e-5f);
        assertEquals(5.0f, cov.getFloat(1, 0), 1e-5f);
        assertEquals(10.0f, cov.getFloat(1, 1), 1e-5f);

        // rowvar = false with transposed matrix (5 observations, 2 variables)
        NDArray covCol = Stats.cov(m.transpose(), false);
        assertArrayEquals(new long[]{2, 2}, covCol.getShape());
        assertEquals(2.5f, covCol.getFloat(0, 0), 1e-5f);
        assertEquals(5.0f, covCol.getFloat(0, 1), 1e-5f);
        assertEquals(5.0f, covCol.getFloat(1, 0), 1e-5f);
        assertEquals(10.0f, covCol.getFloat(1, 1), 1e-5f);
    }

    // =========================================================================
    // CUMSUM TESTS
    // =========================================================================

    @Test
    @DisplayName("Flat and axis cumsum")
    void testCumsum() {
        // Flat 1D
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
        NDArray cs = Stats.cumsum(a);
        assertArrayEquals(new long[]{4}, cs.getShape());
        assertEquals(1f, cs.getFloat(0), 1e-6f);
        assertEquals(3f, cs.getFloat(1), 1e-6f);
        assertEquals(6f, cs.getFloat(2), 1e-6f);
        assertEquals(10f, cs.getFloat(3), 1e-6f);

        // 2D matrix
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f},
            {3f, 4f}
        });

        // Axis 0 (down columns): [[1, 2], [4, 6]]
        NDArray cs0 = Stats.cumsum(m, 0);
        assertArrayEquals(new long[]{2, 2}, cs0.getShape());
        assertEquals(1f, cs0.getFloat(0, 0), 1e-6f);
        assertEquals(2f, cs0.getFloat(0, 1), 1e-6f);
        assertEquals(4f, cs0.getFloat(1, 0), 1e-6f);
        assertEquals(6f, cs0.getFloat(1, 1), 1e-6f);

        // Axis 1 (across rows): [[1, 3], [3, 7]]
        NDArray cs1 = Stats.cumsum(m, 1);
        assertArrayEquals(new long[]{2, 2}, cs1.getShape());
        assertEquals(1f, cs1.getFloat(0, 0), 1e-6f);
        assertEquals(3f, cs1.getFloat(0, 1), 1e-6f);
        assertEquals(3f, cs1.getFloat(1, 0), 1e-6f);
        assertEquals(7f, cs1.getFloat(1, 1), 1e-6f);
    }

    // =========================================================================
    // CUMPROD TESTS
    // =========================================================================

    @Test
    @DisplayName("Flat and axis cumprod")
    void testCumprod() {
        // Flat 1D
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
        NDArray cp = Stats.cumprod(a);
        assertArrayEquals(new long[]{4}, cp.getShape());
        assertEquals(1f, cp.getFloat(0), 1e-6f);
        assertEquals(2f, cp.getFloat(1), 1e-6f);
        assertEquals(6f, cp.getFloat(2), 1e-6f);
        assertEquals(24f, cp.getFloat(3), 1e-6f);

        // 2D matrix
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f},
            {3f, 4f}
        });

        // Axis 0 (down columns): [[1, 2], [3, 8]]
        NDArray cp0 = Stats.cumprod(m, 0);
        assertArrayEquals(new long[]{2, 2}, cp0.getShape());
        assertEquals(1f, cp0.getFloat(0, 0), 1e-6f);
        assertEquals(2f, cp0.getFloat(0, 1), 1e-6f);
        assertEquals(3f, cp0.getFloat(1, 0), 1e-6f);
        assertEquals(8f, cp0.getFloat(1, 1), 1e-6f);

        // Axis 1 (across rows): [[1, 2], [3, 12]]
        NDArray cp1 = Stats.cumprod(m, 1);
        assertArrayEquals(new long[]{2, 2}, cp1.getShape());
        assertEquals(1f, cp1.getFloat(0, 0), 1e-6f);
        assertEquals(2f, cp1.getFloat(0, 1), 1e-6f);
        assertEquals(3f, cp1.getFloat(1, 0), 1e-6f);
        assertEquals(12f, cp1.getFloat(1, 1), 1e-6f);
    }

    // =========================================================================
    // DIFF TESTS
    // =========================================================================

    @Test
    @DisplayName("Discrete difference: 1st, 2nd order, and axis")
    void testDiff() {
        // 1st difference: [1, 3, 6, 10] -> [2, 3, 4]
        NDArray a = JNum.from(new float[]{1f, 3f, 6f, 10f}, 4);
        NDArray d1 = Stats.diff(a);
        assertArrayEquals(new long[]{3}, d1.getShape());
        assertEquals(2f, d1.getFloat(0), 1e-6f);
        assertEquals(3f, d1.getFloat(1), 1e-6f);
        assertEquals(4f, d1.getFloat(2), 1e-6f);

        // 2nd difference: [2, 3, 4] -> [1, 1]
        NDArray d2 = Stats.diff(a, 2);
        assertArrayEquals(new long[]{2}, d2.getShape());
        assertEquals(1f, d2.getFloat(0), 1e-6f);
        assertEquals(1f, d2.getFloat(1), 1e-6f);

        // 2D axis diff
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 4f, 9f},
            {1f, 2f, 3f}
        });

        // diff along axis 1 (across cols): [[3, 5], [1, 1]]
        NDArray dAxis1 = Stats.diff(m, 1, 1);
        assertArrayEquals(new long[]{2, 2}, dAxis1.getShape());
        assertEquals(3f, dAxis1.getFloat(0, 0), 1e-6f);
        assertEquals(5f, dAxis1.getFloat(0, 1), 1e-6f);
        assertEquals(1f, dAxis1.getFloat(1, 0), 1e-6f);
        assertEquals(1f, dAxis1.getFloat(1, 1), 1e-6f);

        // diff along axis 0 (down rows): [[0, -2, -6]]
        NDArray dAxis0 = Stats.diff(m, 1, 0);
        assertArrayEquals(new long[]{1, 3}, dAxis0.getShape());
        assertEquals(0f, dAxis0.getFloat(0, 0), 1e-6f);
        assertEquals(-2f, dAxis0.getFloat(0, 1), 1e-6f);
        assertEquals(-6f, dAxis0.getFloat(0, 2), 1e-6f);

        // Order n >= dimension length: empty result
        NDArray emptyDiff = Stats.diff(m, 5, 1);
        assertArrayEquals(new long[]{2, 0}, emptyDiff.getShape());
    }
}

