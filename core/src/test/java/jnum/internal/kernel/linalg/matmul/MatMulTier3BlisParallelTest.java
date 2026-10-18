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
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;

public class MatMulTier3BlisParallelTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulTier3BlisParallel> constructor = MatMulTier3BlisParallel.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Tier 3 BLIS Parallel Float: 260x260 identity and rectangular")
    void testBlisParallelFloat() {
        try (Arena arena = Arena.ofShared()) {
            int dim = 260;
            NDArray a = TestArrayFactory.random(601L, DType.f32, dim, dim);
            NDArray eye = TestArrayFactory.eye(dim, DType.f32);
            NDArray c = JNum.zeros(arena, DType.f32, dim, dim);
            MatMulTier3BlisParallel.matmulFloat(a, eye, c, dim, dim, dim);
            TestAssertions.assertNDArrayClose(a, c, 1e-5, 1e-5);
        }
    }

    @Test
    @DisplayName("Tier 3 BLIS Parallel Double: 260x260 identity")
    void testBlisParallelDouble() {
        try (Arena arena = Arena.ofShared()) {
            int dim = 260;
            NDArray a = TestArrayFactory.random(701L, DType.f64, dim, dim);
            NDArray eye = TestArrayFactory.eye(dim, DType.f64);
            NDArray c = JNum.zeros(arena, DType.f64, dim, dim);
            MatMulTier3BlisParallel.matmulDouble(a, eye, c, dim, dim, dim);
            TestAssertions.assertNDArrayClose(a, c, 1e-11, 1e-11);
        }
    }

    @Test
    @DisplayName("Tier 3 BLIS Parallel Int: 260x260 identity")
    void testBlisParallelInt() {
        try (Arena arena = Arena.ofShared()) {
            int dim = 260;
            NDArray a = JNum.zeros(DType.i32, dim, dim);
            for (int r = 0; r < dim; r++) {
                for (int c = 0; c < dim; c++) {
                    a.setInt((r * 3 + c) % 17, r, c);
                }
            }
            NDArray eye = TestArrayFactory.eye(dim, DType.i32);
            NDArray c = JNum.zeros(arena, DType.i32, dim, dim);
            MatMulTier3BlisParallel.matmulInt(a, eye, c, dim, dim, dim);

            for (int r = 0; r < dim; r += 26) {
                for (int col = 0; col < dim; col += 26) {
                    assertEquals(a.getInt(r, col), c.getInt(r, col));
                }
            }
        }
    }
}
