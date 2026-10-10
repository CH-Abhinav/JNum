package jnum

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class OperatorsTest {

    @Test
    fun testArrayBinaryOperators() {
        val a = JNum.from(doubleArrayOf(2.0, 6.0), 2)
        val b = JNum.from(doubleArrayOf(4.0, 3.0), 2)

        val add = a + b
        assertEquals(6.0, add[0], 1e-9)
        assertEquals(9.0, add[1], 1e-9)

        val sub = a - b
        assertEquals(-2.0, sub[0], 1e-9)
        assertEquals(3.0, sub[1], 1e-9)

        val mul = a * b
        assertEquals(8.0, mul[0], 1e-9)
        assertEquals(18.0, mul[1], 1e-9)

        val div = a / b
        assertEquals(0.5, div[0], 1e-9)
        assertEquals(2.0, div[1], 1e-9)
    }

    @Test
    fun testUnaryOperators() {
        val a = JNum.from(doubleArrayOf(5.0, -10.0), 2)

        val neg = -a
        assertEquals(-5.0, neg[0], 1e-9)
        assertEquals(10.0, neg[1], 1e-9)

        val pos = +a
        assertEquals(5.0, pos[0], 1e-9)
        assertEquals(-10.0, pos[1], 1e-9)

        val b = JNum.zeros(DType.bool, 2)
        b.setBoolean(true, 0)
        b.setBoolean(false, 1)

        val notB = !b
        assertFalse(notB.getBoolean(0))
        assertTrue(notB.getBoolean(1))
    }

    @Test
    fun testBooleanInfixOperators() {
        val a = JNum.zeros(DType.bool, 4)
        val b = JNum.zeros(DType.bool, 4)

        // a: [T, T, F, F]
        a.setBoolean(true, 0); a.setBoolean(true, 1)
        a.setBoolean(false, 2); a.setBoolean(false, 3)

        // b: [T, F, T, F]
        b.setBoolean(true, 0); b.setBoolean(false, 1)
        b.setBoolean(true, 2); b.setBoolean(false, 3)

        val andRes = a and b
        assertTrue(andRes.getBoolean(0))
        assertFalse(andRes.getBoolean(1))
        assertFalse(andRes.getBoolean(2))
        assertFalse(andRes.getBoolean(3))

        val orRes = a or b
        assertTrue(orRes.getBoolean(0))
        assertTrue(orRes.getBoolean(1))
        assertTrue(orRes.getBoolean(2))
        assertFalse(orRes.getBoolean(3))

        val xorRes = a xor b
        assertFalse(xorRes.getBoolean(0))
        assertTrue(xorRes.getBoolean(1))
        assertTrue(xorRes.getBoolean(2))
        assertFalse(xorRes.getBoolean(3))
    }

    @Test
    fun testRightHandScalarOperators() {
        val a = JNum.from(doubleArrayOf(10.0, 20.0), 2)

        // Double
        val dAdd = a + 2.0; assertEquals(12.0, dAdd[0], 1e-9)
        val dSub = a - 2.0; assertEquals(8.0, dSub[0], 1e-9)
        val dMul = a * 2.0; assertEquals(20.0, dMul[0], 1e-9)
        val dDiv = a / 2.0; assertEquals(5.0, dDiv[0], 1e-9)

        // Float
        val fAdd = a + 3.0f; assertEquals(13.0, fAdd[0], 1e-5)
        val fSub = a - 3.0f; assertEquals(7.0, fSub[0], 1e-5)
        val fMul = a * 3.0f; assertEquals(30.0, fMul[0], 1e-5)
        val fDiv = a / 2.0f; assertEquals(5.0, fDiv[0], 1e-5)

        // Int
        val iAdd = a + 4; assertEquals(14.0, iAdd[0], 1e-9)
        val iSub = a - 4; assertEquals(6.0, iSub[0], 1e-9)
        val iMul = a * 4; assertEquals(40.0, iMul[0], 1e-9)
        val iDiv = a / 5; assertEquals(2.0, iDiv[0], 1e-9)
    }

    @Test
    fun testLeftHandScalarOperators() {
        val a = JNum.from(doubleArrayOf(2.0, 5.0), 2)

        // Double
        val dAdd = 10.0 + a; assertEquals(12.0, dAdd[0], 1e-9)
        val dSub = 10.0 - a; assertEquals(8.0, dSub[0], 1e-9)
        val dMul = 3.0 * a; assertEquals(6.0, dMul[0], 1e-9)

        // Float
        val fAdd = 10.0f + a; assertEquals(12.0, fAdd[0], 1e-5)
        val fSub = 10.0f - a; assertEquals(8.0, fSub[0], 1e-5)
        val fMul = 3.0f * a; assertEquals(6.0, fMul[0], 1e-5)

        // Int
        val iAdd = 10 + a; assertEquals(12.0, iAdd[0], 1e-9)
        val iSub = 10 - a; assertEquals(8.0, iSub[0], 1e-9)
        val iMul = 3 * a; assertEquals(6.0, iMul[0], 1e-9)
    }
}
