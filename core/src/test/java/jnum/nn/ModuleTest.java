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
