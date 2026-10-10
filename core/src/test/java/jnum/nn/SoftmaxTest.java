package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class SoftmaxTest {

    @Test
    void testSoftmax1DDefaultAxis() {
        Softmax softmax = new Softmax();
        NDArray x = JNum.from(new double[]{1.0, 2.0, 3.0}, 3);
        NDArray out = softmax.forward(x);

        double sum = 0.0;
        for (long i = 0; i < 3; i++) {
            double v = out.getDouble(i);
            assertTrue(v >= 0.0 && v <= 1.0, "Probability must be in [0, 1]");
            sum += v;
        }
        assertEquals(1.0, sum, 1e-6, "Probabilities must sum to 1.0");

        // Verify known values: exp(1-3), exp(2-3), exp(3-3)
        double d = Math.exp(-2) + Math.exp(-1) + 1.0;
        assertEquals(Math.exp(-2) / d, out.getDouble(0), 1e-6);
        assertEquals(Math.exp(-1) / d, out.getDouble(1), 1e-6);
        assertEquals(1.0 / d, out.getDouble(2), 1e-6);
    }

    @Test
    void testSoftmax2DRows() {
        Softmax softmax = new Softmax(-1); // along rows (last axis)
        NDArray x = JNum.from(new float[]{
                0.0f, 0.0f,
                10.0f, 10.0f
        }, 2, 2);
        NDArray out = softmax.forward(x);

        // row 0: identical elements -> 0.5, 0.5
        assertEquals(0.5f, out.getFloat(0, 0), 1e-5f);
        assertEquals(0.5f, out.getFloat(0, 1), 1e-5f);

        // row 1: identical elements -> 0.5, 0.5
        assertEquals(0.5f, out.getFloat(1, 0), 1e-5f);
        assertEquals(0.5f, out.getFloat(1, 1), 1e-5f);
    }

    @Test
    void testSoftmax2DColumns() {
        Softmax softmax = new Softmax(0); // along columns
        NDArray x = JNum.from(new double[]{
                1.0, 5.0,
                2.0, 5.0
        }, 2, 2);
        NDArray out = softmax.forward(x);

        // col 1 has identical elements -> 0.5, 0.5
        assertEquals(0.5, out.getDouble(0, 1), 1e-6);
        assertEquals(0.5, out.getDouble(1, 1), 1e-6);

        // col 0 sums to 1.0
        assertEquals(1.0, out.getDouble(0, 0) + out.getDouble(1, 0), 1e-6);
    }

    @Test
    void testNumericalStabilityLargeValues() {
        Softmax softmax = new Softmax();
        NDArray large = JNum.from(new double[]{1000.0, 1001.0, 1002.0}, 3);
        NDArray out = softmax.forward(large);

        assertFalse(Double.isNaN(out.getDouble(0)));
        assertFalse(Double.isInfinite(out.getDouble(0)));
        assertEquals(1.0, out.getDouble(0) + out.getDouble(1) + out.getDouble(2), 1e-6);
    }
}
