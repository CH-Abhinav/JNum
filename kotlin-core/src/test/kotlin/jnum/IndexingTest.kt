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
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Test

class IndexingTest {

    @Test
    fun testAllAndUnderscoreTokens() {
        assertEquals(all.start(), Slice.all().start())
        assertEquals(all.stop(), Slice.all().stop())
        assertEquals(`_`.start(), Slice.all().start())
    }

    @Test
    fun testElementGetters1D2D3DVararg() {
        // 1D
        val a1 = JNum.from(doubleArrayOf(10.0, 20.0, 30.0), 3)
        assertEquals(10.0, a1[0], 1e-9)
        assertEquals(30.0, a1[2], 1e-9)

        // 2D
        val a2 = JNum.from(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)
        assertEquals(1.0, a2[0, 0], 1e-9)
        assertEquals(4.0, a2[1, 1], 1e-9)

        // 3D
        val a3 = JNum.from(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), 2, 2, 2)
        assertEquals(1.0, a3[0, 0, 0], 1e-9)
        assertEquals(8.0, a3[1, 1, 1], 1e-9)

        // Vararg
        assertEquals(1.0, a3[0, 0, 0], 1e-9)
        assertEquals(8.0, a3[1, 1, 1], 1e-9)
    }

    @Test
    fun testElementSetters1D2D3D() {
        // 1D Double
        val a1Double = JNum.zeros(DType.f64, 3)
        a1Double[0] = 1.5
        a1Double[1] = 2.5
        a1Double[2] = 3.5
        assertEquals(1.5, a1Double[0], 1e-9)
        assertEquals(2.5, a1Double[1], 1e-9)
        assertEquals(3.5, a1Double[2], 1e-9)

        // 1D Float
        val a1Float = JNum.zeros(DType.f32, 2)
        a1Float[0] = 4.0f
        a1Float[1] = 5.0f
        assertEquals(4.0, a1Float[0], 1e-5)
        assertEquals(5.0, a1Float[1], 1e-5)

        // 1D Int
        val a1Int = JNum.zeros(DType.i32, 2)
        a1Int[0] = 10
        a1Int[1] = 20
        assertEquals(10.0, a1Int[0], 1e-9)
        assertEquals(20.0, a1Int[1], 1e-9)

        // 2D setters
        val a2 = JNum.zeros(DType.f64, 2, 2)
        a2[0, 0] = 10.0
        a2[0, 1] = 20.0
        a2[1, 0] = 30.0
        assertEquals(10.0, a2[0, 0], 1e-9)
        assertEquals(20.0, a2[0, 1], 1e-9)
        assertEquals(30.0, a2[1, 0], 1e-9)

        // 3D setters
        val a3 = JNum.zeros(DType.f64, 2, 2, 2)
        a3[0, 0, 0] = 100.0
        a3[1, 1, 1] = 200.0
        assertEquals(100.0, a3[0, 0, 0], 1e-9)
        assertEquals(200.0, a3[1, 1, 1], 1e-9)
    }

    @Test
    fun testSlicingWithSliceObjects() {
        val a = JNum.from(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), 2, 3)
        val s = a[Slice.range(0, 1), Slice.range(1, 3)]
        assertArrayEquals(longArrayOf(1, 2), s.shape)
        assertEquals(2.0, s[0, 0], 1e-9)
        assertEquals(3.0, s[0, 1], 1e-9)
    }

    @Test
    fun testSlicingWithIntProgressions() {
        val a = JNum.from(doubleArrayOf(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ), 3, 4)

        // 2D progression slice
        val sub = a[0..1, 1..2]
        assertArrayEquals(longArrayOf(2, 2), sub.shape)
        assertEquals(2.0, sub[0, 0], 1e-9)
        assertEquals(3.0, sub[0, 1], 1e-9)
        assertEquals(6.0, sub[1, 0], 1e-9)
        assertEquals(7.0, sub[1, 1], 1e-9)

        // Row with progression
        val rowSub = a[1, 0..1]
        assertArrayEquals(longArrayOf(1, 2), rowSub.shape)
        assertEquals(5.0, rowSub[0, 0], 1e-9)
        assertEquals(6.0, rowSub[0, 1], 1e-9)

        // Progression with Col
        val colSub = a[0..1, 2]
        assertArrayEquals(longArrayOf(2, 1), colSub.shape)
        assertEquals(3.0, colSub[0, 0], 1e-9)
        assertEquals(7.0, colSub[1, 0], 1e-9)

        // Slice with progression & progression with slice
        val allCols = a[0..1, all]
        assertArrayEquals(longArrayOf(2, 4), allCols.shape)

        val allRows = a[`_`, 1..2]
        assertArrayEquals(longArrayOf(3, 2), allRows.shape)
    }

    @Test
    fun testSteppedProgression() {
        val a = JNum.from(doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0), 5)

        val stepped = a[0..4 step 2]
        assertArrayEquals(longArrayOf(3), stepped.shape)
        assertEquals(10.0, stepped[0], 1e-9)
        assertEquals(30.0, stepped[1], 1e-9)
        assertEquals(50.0, stepped[2], 1e-9)
    }

    @Test
    fun testNegativeRowColIndexingInSlices() {
        val a = JNum.from(doubleArrayOf(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ), 3, 4)

        // Last row (-1) with progression
        val lastRow = a[-1, 0..1]
        assertArrayEquals(longArrayOf(1, 2), lastRow.shape)
        assertEquals(9.0, lastRow[0, 0], 1e-9)
        assertEquals(10.0, lastRow[0, 1], 1e-9)

        // Progression with last col (-1)
        val lastCol = a[0..1, -1]
        assertArrayEquals(longArrayOf(2, 1), lastCol.shape)
        assertEquals(4.0, lastCol[0, 0], 1e-9)
        assertEquals(8.0, lastCol[1, 0], 1e-9)

        // Last row (-1) with all slice
        val lastRowAll = a[-1, all]
        assertArrayEquals(longArrayOf(1, 4), lastRowAll.shape)
        assertEquals(9.0, lastRowAll[0, 0], 1e-9)
        assertEquals(12.0, lastRowAll[0, 3], 1e-9)

        // All slice with last col (-1)
        val allLastCol = a[all, -1]
        assertArrayEquals(longArrayOf(3, 1), allLastCol.shape)
        assertEquals(4.0, allLastCol[0, 0], 1e-9)
        assertEquals(12.0, allLastCol[2, 0], 1e-9)
    }

    @Test
    fun testMixedIntAndSliceOverloads2D() {
        val a = JNum.from(doubleArrayOf(
            1.0, 2.0,
            3.0, 4.0
        ), 2, 2)

        // row: Int, slice: Slice
        val row0 = a[0, all]
        assertArrayEquals(longArrayOf(1, 2), row0.shape)
        assertEquals(1.0, row0[0, 0], 1e-9)
        assertEquals(2.0, row0[0, 1], 1e-9)

        // slice: Slice, col: Int
        val col0 = a[all, 0]
        assertArrayEquals(longArrayOf(2, 1), col0.shape)
        assertEquals(1.0, col0[0, 0], 1e-9)
        assertEquals(3.0, col0[1, 0], 1e-9)
    }

    @Test
    fun testMixed3DSlices() {
        // Shape: [2, 3, 4]
        val data = DoubleArray(24) { it.toDouble() }
        val t = JNum.from(data, 2, 3, 4)

        // (IntProgression, IntProgression, Slice) - e.g. image crop recipe
        val crop = t[0..1, 1..2, all]
        assertArrayEquals(longArrayOf(2, 2, 4), crop.shape)

        // (Slice, Slice, IntProgression)
        val s1 = t[all, all, 0..1]
        assertArrayEquals(longArrayOf(2, 3, 2), s1.shape)

        // (IntProgression, Slice, Slice)
        val s2 = t[0..0, all, all]
        assertArrayEquals(longArrayOf(1, 3, 4), s2.shape)

        // (Slice, IntProgression, Slice)
        val s3 = t[all, 1..2, all]
        assertArrayEquals(longArrayOf(2, 2, 4), s3.shape)
    }
}
