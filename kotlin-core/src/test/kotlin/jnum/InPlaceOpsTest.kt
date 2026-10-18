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

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class InPlaceOpsTest {

    @Test
    fun testInPlaceArrayAssignments() {
        val a = JNum.from(doubleArrayOf(10.0, 20.0), 2)
        val b = JNum.from(doubleArrayOf(2.0, 5.0), 2)

        a += b
        assertEquals(12.0, a[0], 1e-9)
        assertEquals(25.0, a[1], 1e-9)

        a -= b
        assertEquals(10.0, a[0], 1e-9)
        assertEquals(20.0, a[1], 1e-9)

        a *= b
        assertEquals(20.0, a[0], 1e-9)
        assertEquals(100.0, a[1], 1e-9)

        a /= b
        assertEquals(10.0, a[0], 1e-9)
        assertEquals(20.0, a[1], 1e-9)
    }

    @Test
    fun testInPlaceDoubleAssignments() {
        val a = JNum.from(doubleArrayOf(10.0, 20.0), 2)

        a += 5.0
        assertEquals(15.0, a[0], 1e-9)

        a -= 3.0
        assertEquals(12.0, a[0], 1e-9)

        a *= 2.0
        assertEquals(24.0, a[0], 1e-9)

        a /= 4.0
        assertEquals(6.0, a[0], 1e-9)
    }

    @Test
    fun testInPlaceFloatAssignments() {
        val a = JNum.from(floatArrayOf(10.0f, 20.0f), 2)

        a += 5.0f
        assertEquals(15.0, a[0], 1e-5)

        a -= 3.0f
        assertEquals(12.0, a[0], 1e-5)

        a *= 2.0f
        assertEquals(24.0, a[0], 1e-5)

        a /= 4.0f
        assertEquals(6.0, a[0], 1e-5)
    }

    @Test
    fun testInPlaceIntAssignments() {
        val a = JNum.from(intArrayOf(10, 20), 2)

        a += 5
        assertEquals(15.0, a[0], 1e-9)

        a -= 3
        assertEquals(12.0, a[0], 1e-9)

        a *= 2
        assertEquals(24.0, a[0], 1e-9)

        a /= 4
        assertEquals(6.0, a[0], 1e-9)
    }
}
