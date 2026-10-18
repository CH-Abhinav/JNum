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
package jnum

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LinalgOpsTest {

    @Test
    fun testInfixMatMul() {
        val a = JNum.from(doubleArrayOf(
            1.0, 2.0,
            3.0, 4.0
        ), 2, 2)

        val b = JNum.from(doubleArrayOf(
            2.0, 0.0,
            1.0, 2.0
        ), 2, 2)

        val c = a matmul b

        assertArrayEquals(longArrayOf(2, 2), c.shape)
        // row 0: 1*2 + 2*1 = 4; 1*0 + 2*2 = 4
        assertEquals(4.0, c[0, 0], 1e-9)
        assertEquals(4.0, c[0, 1], 1e-9)
        // row 1: 3*2 + 4*1 = 10; 3*0 + 4*2 = 8
        assertEquals(10.0, c[1, 0], 1e-9)
        assertEquals(8.0, c[1, 1], 1e-9)
    }

    @Test
    fun testInfixDot() {
        val u = JNum.from(doubleArrayOf(1.0, 3.0, -5.0), 3)
        val v = JNum.from(doubleArrayOf(4.0, -2.0, -1.0), 3)

        val result = u dot v
        // 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3.0
        assertEquals(3.0, result, 1e-9)
    }
}
