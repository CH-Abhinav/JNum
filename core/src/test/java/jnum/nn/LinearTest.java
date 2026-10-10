package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class LinearTest {

    @Test
    void testNon2DWeightsThrowsException() {
        NDArray weights1D = JNum.from(new float[]{1.0f, 2.0f}, 2);
        assertThrows(IllegalArgumentException.class, () -> new Linear(weights1D, null));

        NDArray weights3D = JNum.zeros(DType.f32, 2, 2, 2);
        assertThrows(IllegalArgumentException.class, () -> new Linear(weights3D, null));
    }

    @Test
    void testFeatureMismatchThrowsException() {
        // weights: [out=2, in=3]
        NDArray weights = JNum.zeros(DType.f32, 2, 3);
        Linear linear = new Linear(weights, null);

        // input: [batch=4, in=2] -> mismatch
        NDArray input = JNum.zeros(DType.f32, 4, 2);
        assertThrows(IllegalArgumentException.class, () -> linear.forward(input));
    }

    @Test
    void testForwardPassWithoutBias() {
        // weights: [out=2, in=3]
        // [ [1, 2, 3],
        //   [4, 5, 6] ]
        NDArray weights = JNum.from(new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        }, 2, 3);
        Linear linear = new Linear(weights, null);

        // input: [batch=2, in=3]
        // [ [1, 1, 1],
        //   [2, 0, 1] ]
        NDArray input = JNum.from(new float[]{
                1.0f, 1.0f, 1.0f,
                2.0f, 0.0f, 1.0f
        }, 2, 3);

        NDArray output = linear.forward(input);
        assertArrayEquals(new long[]{2, 2}, output.getShape());

        // row 0: 1*1 + 1*2 + 1*3 = 6; 1*4 + 1*5 + 1*6 = 15
        assertEquals(6.0f, output.getFloat(0, 0), 1e-5f);
        assertEquals(15.0f, output.getFloat(0, 1), 1e-5f);

        // row 1: 2*1 + 0*2 + 1*3 = 5; 2*4 + 0*5 + 1*6 = 14
        assertEquals(5.0f, output.getFloat(1, 0), 1e-5f);
        assertEquals(14.0f, output.getFloat(1, 1), 1e-5f);
    }

    @Test
    void testForwardPassWithBias() {
        // weights: [out=2, in=2]
        NDArray weights = JNum.from(new double[]{
                1.0, 0.0,
                0.0, 1.0
        }, 2, 2);
        NDArray bias = JNum.from(new double[]{0.5, -0.5}, 1, 2);
        Linear linear = new Linear(weights, bias);

        NDArray input = JNum.from(new double[]{
                2.0, 3.0
        }, 1, 2);

        NDArray output = linear.forward(input);
        assertArrayEquals(new long[]{1, 2}, output.getShape());
        assertEquals(2.5, output.getDouble(0, 0), 1e-9);
        assertEquals(2.5, output.getDouble(0, 1), 1e-9);
    }

    @Test
    void testBatched3DForwardPass() {
        // weights: [out=2, in=3]
        NDArray weights = JNum.from(new float[]{
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f
        }, 2, 3);
        Linear linear = new Linear(weights, null);

        // input: [batch=2, seq=2, in=3]
        NDArray input = JNum.ones(DType.f32, 2, 2, 3);
        NDArray output = linear.forward(input);

        assertArrayEquals(new long[]{2, 2, 2}, output.getShape());
        for (long b = 0; b < 2; b++) {
            for (long s = 0; s < 2; s++) {
                assertEquals(1.0f, output.getFloat(b, s, 0), 1e-5f);
                assertEquals(1.0f, output.getFloat(b, s, 1), 1e-5f);
            }
        }
    }
}
