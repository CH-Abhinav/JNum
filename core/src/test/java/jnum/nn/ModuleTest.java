package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class ModuleTest {

    @Test
    void testModuleLambdaImplementation() {
        Module doubleModule = input -> input.mul(2.0);
        NDArray x = JNum.from(new double[]{1.0, 2.0, 3.0}, 3);
        NDArray out = doubleModule.forward(x);

        assertEquals(2.0, out.getDouble(0), 1e-9);
        assertEquals(4.0, out.getDouble(1), 1e-9);
        assertEquals(6.0, out.getDouble(2), 1e-9);
    }

    @Test
    void testIdentityModule() {
        Module identity = input -> input;
        NDArray x = JNum.from(new float[]{1.5f, 2.5f}, 2);
        NDArray out = identity.forward(x);

        assertSame(x, out);
    }
}
