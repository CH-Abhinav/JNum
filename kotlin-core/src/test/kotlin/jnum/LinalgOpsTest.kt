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
