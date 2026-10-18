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
package jnum.internal.kernel.linalg.matmul;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.ReferenceOps;
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;

public class MatMulTier0NanoTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulTier0Nano> constructor = MatMulTier0Nano.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Tier 0 Nano Float: 1x1, 2x3*3x2, 4x4")
    void testNanoFloat() {
        try (Arena arena = Arena.ofConfined()) {
            // 2x3 and 3x2
            NDArray a = TestArrayFactory.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            NDArray b = TestArrayFactory.matrix(new float[][]{{7, 8}, {9, 1}, {2, 3}});
            NDArray c = JNum.zeros(arena, DType.f32, 2, 2);
            MatMulTier0Nano.matmulFloat(a, b, c, 2, 3, 2);

            assertEquals(31.0f, c.getFloat(0, 0), 1e-5f);
            assertEquals(19.0f, c.getFloat(0, 1), 1e-5f);
            assertEquals(85.0f, c.getFloat(1, 0), 1e-5f);
            assertEquals(55.0f, c.getFloat(1, 1), 1e-5f);

            // 4x4 random
            NDArray a4 = TestArrayFactory.random(11L, DType.f32, 4, 4);
            NDArray b4 = TestArrayFactory.random(12L, DType.f32, 4, 4);
            NDArray c4 = JNum.zeros(arena, DType.f32, 4, 4);
            MatMulTier0Nano.matmulFloat(a4, b4, c4, 4, 4, 4);

            float[] ref = ReferenceOps.matmulFloat(TestArrayFactory.toFloatArray(a4), TestArrayFactory.toFloatArray(b4), 4, 4, 4);
            TestAssertions.assertArrayClose(ref, TestArrayFactory.toFloatArray(c4), 1e-5f, 1e-5f);
        }
    }

    @Test
    @DisplayName("Tier 0 Nano Double: 1x1, 3x3")
    void testNanoDouble() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a1 = TestArrayFactory.matrix(new double[][]{{3.5}});
            NDArray b1 = TestArrayFactory.matrix(new double[][]{{2.0}});
            NDArray c1 = JNum.zeros(arena, DType.f64, 1, 1);
            MatMulTier0Nano.matmulDouble(a1, b1, c1, 1, 1, 1);
            assertEquals(7.0, c1.getDouble(0, 0), 1e-12);

            NDArray a3 = TestArrayFactory.random(21L, DType.f64, 3, 3);
            NDArray b3 = TestArrayFactory.random(22L, DType.f64, 3, 3);
            NDArray c3 = JNum.zeros(arena, DType.f64, 3, 3);
            MatMulTier0Nano.matmulDouble(a3, b3, c3, 3, 3, 3);

            double[] ref = ReferenceOps.matmulDouble(TestArrayFactory.toDoubleArray(a3), TestArrayFactory.toDoubleArray(b3), 3, 3, 3);
            TestAssertions.assertArrayClose(ref, TestArrayFactory.toDoubleArray(c3), 1e-11, 1e-11);
        }
    }

    @Test
    @DisplayName("Tier 0 Nano Int: 3x3 identity, 4x4")
    void testNanoInt() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray eye = TestArrayFactory.eye(3, DType.i32);
            NDArray b = TestArrayFactory.matrix(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            NDArray c = JNum.zeros(arena, DType.i32, 3, 3);
            MatMulTier0Nano.matmulInt(eye, b, c, 3, 3, 3);

            assertArrayEquals(new int[]{1, 2, 3}, new int[]{c.getInt(0, 0), c.getInt(0, 1), c.getInt(0, 2)});
            assertArrayEquals(new int[]{4, 5, 6}, new int[]{c.getInt(1, 0), c.getInt(1, 1), c.getInt(1, 2)});
            assertArrayEquals(new int[]{7, 8, 9}, new int[]{c.getInt(2, 0), c.getInt(2, 1), c.getInt(2, 2)});
        }
    }

    @Test
    @DisplayName("Tier 0 Nano Non-contiguous transpose")
    void testNanoNonContiguous() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = TestArrayFactory.matrix(new float[][]{{1, 2}, {3, 4}, {5, 6}}).transpose(); // 2x3
            NDArray b = TestArrayFactory.matrix(new float[][]{{7, 8}, {9, 1}, {2, 3}}); // 3x2
            NDArray c = JNum.zeros(arena, DType.f32, 2, 2);
            MatMulTier0Nano.matmulFloat(a, b, c, 2, 3, 2);

            // A is [[1, 3, 5], [2, 4, 6]]
            // c[0, 0] = 1*7 + 3*9 + 5*2 = 7 + 27 + 10 = 44
            // c[0, 1] = 1*8 + 3*1 + 5*3 = 8 + 3 + 15 = 26
            // c[1, 0] = 2*7 + 4*9 + 6*2 = 14 + 36 + 12 = 62
            // c[1, 1] = 2*8 + 4*1 + 6*3 = 16 + 4 + 18 = 38
            assertEquals(44.0f, c.getFloat(0, 0), 1e-5f);
            assertEquals(26.0f, c.getFloat(0, 1), 1e-5f);
            assertEquals(62.0f, c.getFloat(1, 0), 1e-5f);
            assertEquals(38.0f, c.getFloat(1, 1), 1e-5f);
        }
    }
}
