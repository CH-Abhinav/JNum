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

class SequentialTest {

    @Test
    void testEmptySequentialReturnsInput() {
        Sequential seq = new Sequential();
        NDArray x = JNum.from(new float[]{1.0f, 2.0f}, 2);
        NDArray out = seq.forward(x);
        assertSame(x, out);
    }

    @Test
    void testMultiLayerSequential() {
        // Layer 1: weights [2, 2], bias [1, 2]
        NDArray w1 = JNum.from(new float[]{
                1.0f, -1.0f,
                -1.0f, 1.0f
        }, 2, 2);
        NDArray b1 = JNum.from(new float[]{0.0f, 0.0f}, 1, 2);
        Linear l1 = new Linear(w1, b1);

        // Layer 2: ReLU
        ReLU relu = new ReLU();

        Sequential model = new Sequential(l1, relu);

        // Input: [[2, 1]]
        // l1 output: [[2*1 + 1*(-1), 2*(-1) + 1*1]] = [[1, -1]]
        // relu output: [[1, 0]]
        NDArray input = JNum.from(new float[]{2.0f, 1.0f}, 1, 2);
        NDArray output = model.forward(input);

        assertArrayEquals(new long[]{1, 2}, output.getShape());
        assertEquals(1.0f, output.getFloat(0, 0), 1e-5f);
        assertEquals(0.0f, output.getFloat(0, 1), 1e-5f);
    }
}
