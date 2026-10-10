package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class SigmoidTest {

    @Test
    void testSigmoidFloat32() {
        Sigmoid sigmoid = new Sigmoid();
        NDArray x = JNum.from(new float[]{0.0f, 100.0f, -100.0f}, 3);
        NDArray out = sigmoid.forward(x);

        assertEquals(0.5f, out.getFloat(0), 1e-5f);
        assertEquals(1.0f, out.getFloat(1), 1e-4f);
        assertEquals(0.0f, out.getFloat(2), 1e-4f);
    }

    @Test
    void testSigmoidFloat64() {
        Sigmoid sigmoid = new Sigmoid();
        NDArray x = JNum.from(new double[]{0.0, 2.0, -2.0}, 3);
        NDArray out = sigmoid.forward(x);

        assertEquals(0.5, out.getDouble(0), 1e-7);
        assertEquals(1.0 / (1.0 + Math.exp(-2.0)), out.getDouble(1), 1e-7);
        assertEquals(1.0 / (1.0 + Math.exp(2.0)), out.getDouble(2), 1e-7);
    }
}
