/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.nn;

import static org.junit.jupiter.api.Assertions.*;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.Test;

class ReLUTest {

    @Test
    void testReLUFloat32() {
        ReLU relu = new ReLU();
        NDArray x = JNum.from(new float[]{-3.0f, 0.0f, 5.5f, -0.001f}, 4);
        NDArray out = relu.forward(x);

        assertEquals(0.0f, out.getFloat(0), 1e-6f);
        assertEquals(0.0f, out.getFloat(1), 1e-6f);
        assertEquals(5.5f, out.getFloat(2), 1e-6f);
        assertEquals(0.0f, out.getFloat(3), 1e-6f);
    }

    @Test
    void testReLUFloat64() {
        ReLU relu = new ReLU();
        NDArray x = JNum.from(new double[]{-10.5, 0.0, 20.2}, 3);
        NDArray out = relu.forward(x);

        assertEquals(0.0, out.getDouble(0), 1e-9);
        assertEquals(0.0, out.getDouble(1), 1e-9);
        assertEquals(20.2, out.getDouble(2), 1e-9);
    }

    @Test
    void testReLUInt32() {
        ReLU relu = new ReLU();
        NDArray x = JNum.from(new int[]{-5, 0, 7}, 3);
        NDArray out = relu.forward(x);

        assertEquals(0, out.getInt(0));
        assertEquals(0, out.getInt(1));
        assertEquals(7, out.getInt(2));
    }

    @Test
    void testUnsupportedDTypeThrowsException() {
        ReLU relu = new ReLU();
        NDArray boolArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(IllegalArgumentException.class, () -> relu.forward(boolArr));
    }
}
