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
