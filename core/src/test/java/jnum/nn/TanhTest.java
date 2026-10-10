package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class TanhTest {

    @Test
    void testTanhFloat32() {
        Tanh tanh = new Tanh();
        NDArray x = JNum.from(new float[]{0.0f, 50.0f, -50.0f}, 3);
        NDArray out = tanh.forward(x);

        assertEquals(0.0f, out.getFloat(0), 1e-6f);
        assertEquals(1.0f, out.getFloat(1), 1e-4f);
        assertEquals(-1.0f, out.getFloat(2), 1e-4f);
    }

    @Test
    void testTanhFloat64() {
        Tanh tanh = new Tanh();
        NDArray x = JNum.from(new double[]{0.0, 1.0, -1.0}, 3);
        NDArray out = tanh.forward(x);

        assertEquals(0.0, out.getDouble(0), 1e-9);
        assertEquals(Math.tanh(1.0), out.getDouble(1), 1e-9);
        assertEquals(Math.tanh(-1.0), out.getDouble(2), 1e-9);
    }
}
