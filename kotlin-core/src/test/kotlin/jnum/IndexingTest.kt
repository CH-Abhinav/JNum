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
}
