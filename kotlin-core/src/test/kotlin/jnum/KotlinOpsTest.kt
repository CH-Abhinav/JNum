package jnum

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class KotlinOpsTest {

    @Test
    fun testBinaryArithmetic() {
        val a = JNum.from(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)
        val b = JNum.from(doubleArrayOf(10.0, 20.0, 30.0, 40.0), 2, 2)

        // a + b
        val add = a + b
        assertEquals(11.0, add[0, 0])
        assertEquals(22.0, add[0, 1])
        assertEquals(33.0, add[1, 0])
        assertEquals(44.0, add[1, 1])

        // b - a
        val sub = b - a
        assertEquals(9.0, sub[0, 0])
        assertEquals(18.0, sub[0, 1])
        assertEquals(27.0, sub[1, 0])
        assertEquals(36.0, sub[1, 1])

        // a * b
        val mul = a * b
        assertEquals(10.0, mul[0, 0])
        assertEquals(40.0, mul[0, 1])
        assertEquals(90.0, mul[1, 0])
        assertEquals(160.0, mul[1, 1])

        // b / a
        val div = b / a
        assertEquals(10.0, div[0, 0])
        assertEquals(10.0, div[0, 1])
        assertEquals(10.0, div[1, 0])
        assertEquals(10.0, div[1, 1])
    }

    @Test
    fun testScalarArithmetic() {
        val a = JNum.from(doubleArrayOf(2.0, 4.0), 2)

        // Right-hand scalars
        val rAdd = a + 5.0
        assertEquals(7.0, rAdd[0])
        assertEquals(9.0, rAdd[1])

        val rSub = a - 1.0
        assertEquals(1.0, rSub[0])
        assertEquals(3.0, rSub[1])

        val rMul = a * 3.0
        assertEquals(6.0, rMul[0])
        assertEquals(12.0, rMul[1])

        val rDiv = a / 2.0
        assertEquals(1.0, rDiv[0])
        assertEquals(2.0, rDiv[1])

        // Left-hand scalars
        val lAdd = 10.0 + a
        assertEquals(12.0, lAdd[0])
        assertEquals(14.0, lAdd[1])

        val lSub = 10.0 - a
        assertEquals(8.0, lSub[0])
        assertEquals(6.0, lSub[1])

        val lMul = 2.0 * a
        assertEquals(4.0, lMul[0])
        assertEquals(8.0, lMul[1])
    }

    @Test
    fun testUnaryAndBooleanOperators() {
        val a = JNum.from(doubleArrayOf(2.0, -5.0), 2)

        // -a (unary minus)
        val neg = -a
        assertEquals(-2.0, neg[0])
        assertEquals(5.0, neg[1])

        // +a (unary plus)
        val pos = +a
        assertEquals(2.0, pos[0])
        assertEquals(-5.0, pos[1])

        // Boolean operations
        val boolA = JNum.zeros(DType.bool, 2)
        val boolB = JNum.ones(DType.bool, 2)
        boolA.setBoolean(true, 0)
        boolA.setBoolean(false, 1)

        // !boolA
        val notA = !boolA
        assertFalse(notA.getBoolean(0))
        assertTrue(notA.getBoolean(1))

        // boolA and boolB
        val andRes = boolA and boolB
        assertTrue(andRes.getBoolean(0))
        assertFalse(andRes.getBoolean(1))

        // boolA or boolB
        val orRes = boolA or boolB
        assertTrue(orRes.getBoolean(0))
        assertTrue(orRes.getBoolean(1))

        // boolA xor boolB
        val xorRes = boolA xor boolB
        assertFalse(xorRes.getBoolean(0))
        assertTrue(xorRes.getBoolean(1))
    }

    @Test
    fun testInPlaceAssignments() {
        val a = JNum.from(doubleArrayOf(1.0, 2.0), 2)
        val b = JNum.from(doubleArrayOf(3.0, 4.0), 2)

        a += b
        assertEquals(4.0, a[0])
        assertEquals(6.0, a[1])

        a -= 1.0
        assertEquals(3.0, a[0])
        assertEquals(5.0, a[1])

        a *= 2.0
        assertEquals(6.0, a[0])
        assertEquals(10.0, a[1])

        a /= 2.0
        assertEquals(3.0, a[0])
        assertEquals(5.0, a[1])
    }

    @Test
    fun testIndexingAndMutation() {
        val a = JNum.zeros(DType.f64, 3, 3)

        // Direct 2D mutation
        a[1, 2] = 42.0
        assertEquals(42.0, a[1, 2])

        // Direct 1D array mutation
        val v = JNum.zeros(DType.f64, 5)
        v[3] = 99.0
        assertEquals(99.0, v[3])
    }

    @Test
    fun testPythonStyleSlicing() {
        // Create 4x4 matrix
        val data = DoubleArray(16) { it.toDouble() }
        val mat = JNum.from(data, 4, 4)

        // 2D Range Slice: mat[0..1, 0..1] -> 2x2 submatrix
        val block = mat[0..1, 0..1]
        assertEquals(2L, block.shape[0])
        assertEquals(2L, block.shape[1])
        assertEquals(0.0, block[0, 0])
        assertEquals(1.0, block[0, 1])
        assertEquals(4.0, block[1, 0])
        assertEquals(5.0, block[1, 1])

        // Half-open range (0 until 2)
        val blockUntil = mat[0 until 2, 0 until 2]
        assertEquals(2L, blockUntil.shape[0])
        assertEquals(2L, blockUntil.shape[1])
        assertEquals(0.0, blockUntil[0, 0])

        // Full axis slice using 'all' and '`_`' token: mat[all, 1 until 3]
        val colSlice = mat[all, 1 until 3]
        assertEquals(4L, colSlice.shape[0])
        assertEquals(2L, colSlice.shape[1])
        assertEquals(1.0, colSlice[0, 0])
        assertEquals(2.0, colSlice[0, 1])

        // Row slice with progression: mat[1, 0 until 4]
        val rowSlice = mat[1, 0 until 4]
        assertEquals(1L, rowSlice.shape[0])
        assertEquals(4L, rowSlice.shape[1])
        assertEquals(4.0, rowSlice[0, 0])

        // Strided slice: mat[0..3 step 2, `_`]
        val strided = mat[0..3 step 2, `_`]
        assertEquals(2L, strided.shape[0])
        assertEquals(4L, strided.shape[1])
        assertEquals(0.0, strided[0, 0])
        assertEquals(8.0, strided[1, 0])
    }

    @Test
    fun testInfixMatmulAndDot() {
        // [ [1, 2], [3, 4] ]
        val a = JNum.from(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)
        // [ [5, 6], [7, 8] ]
        val b = JNum.from(doubleArrayOf(5.0, 6.0, 7.0, 8.0), 2, 2)

        val c = a matmul b
        // Row 0, Col 0: 1*5 + 2*7 = 19
        assertEquals(19.0, c[0, 0])
        // Row 0, Col 1: 1*6 + 2*8 = 22
        assertEquals(22.0, c[0, 1])
        // Row 1, Col 0: 3*5 + 4*7 = 43
        assertEquals(43.0, c[1, 0])
        // Row 1, Col 1: 3*6 + 4*8 = 50
        assertEquals(50.0, c[1, 1])

        // 1D dot product
        val v1 = JNum.from(doubleArrayOf(1.0, 2.0, 3.0), 3)
        val v2 = JNum.from(doubleArrayOf(4.0, 5.0, 6.0), 3)
        // 1*4 + 2*5 + 3*6 = 32
        val dotProduct = v1 dot v2
        assertEquals(32.0, dotProduct)
    }
}
