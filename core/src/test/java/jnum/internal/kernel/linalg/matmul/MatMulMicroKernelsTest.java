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

import static jnum.internal.Constants.*;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class MatMulMicroKernelsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulMicroKernels> constructor = MatMulMicroKernels.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("MicroKernel Float: 6x2vec and scalar fallback")
    void testMicroKernelsFloat() {
        try (Arena arena = Arena.ofConfined()) {
            int kc = 16;
            int nr = NR_F32;
            int mr = MR; // 6
            MemorySegment pA = arena.allocate((long) mr * kc * 4L);
            MemorySegment pB = arena.allocate((long) nr * kc * 4L);
            MemorySegment c1 = arena.allocate((long) mr * nr * 4L);
            MemorySegment c2 = arena.allocate((long) mr * nr * 4L);

            // Populate pA and pB
            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mr; r++) {
                    pA.set(VF, ((long) k * mr + r) * 4L, (float)(r + 1));
                }
                for (int c = 0; c < nr; c++) {
                    pB.set(VF, ((long) k * nr + c) * 4L, (float)(c + 1));
                }
            }

            // Test SIMD 6x2vec
            MatMulMicroKernels.microKernel6x2vec_Float(pA, 0L, pB, 0L, c1, 0, 0, kc, nr, true);

            // Test Scalar
            MatMulMicroKernels.microKernelScalar_Float(pA, 0L, 0, pB, 0L, c2, 0, 0, kc, nr, mr, nr, mr, nr, true);

            for (int r = 0; r < mr; r++) {
                for (int c = 0; c < nr; c++) {
                    float expected = (float)(r + 1) * (float)(c + 1) * kc;
                    assertEquals(expected, c1.get(VF, ((long) r * nr + c) * 4L), 1e-4f);
                    assertEquals(expected, c2.get(VF, ((long) r * nr + c) * 4L), 1e-4f);
                }
            }

            // Test 2x2vec and 1x2vec
            MemorySegment c2vec = arena.allocate(2L * nr * 4L);
            MatMulMicroKernels.microKernel2x2vec_Float(pA, 0L, 0, pB, 0L, c2vec, 0, 0, kc, nr, true);
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < nr; c++) {
                    float expected = (float)(r + 1) * (float)(c + 1) * kc;
                    assertEquals(expected, c2vec.get(VF, ((long) r * nr + c) * 4L), 1e-4f);
                }
            }

            MemorySegment c1vec = arena.allocate((long) nr * 4L);
            MatMulMicroKernels.microKernel1x2vec_Float(pA, 0L, 0, pB, 0L, c1vec, 0, 0, kc, nr, true);
            for (int c = 0; c < nr; c++) {
                float expected = 1.0f * (float)(c + 1) * kc;
                assertEquals(expected, c1vec.get(VF, (long) c * 4L), 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("MicroKernel Double: 6x2vec and scalar fallback")
    void testMicroKernelsDouble() {
        try (Arena arena = Arena.ofConfined()) {
            int kc = 8;
            int nr = NR_F64;
            int mr = MR;
            MemorySegment pA = arena.allocate((long) mr * kc * 8L);
            MemorySegment pB = arena.allocate((long) nr * kc * 8L);
            MemorySegment c1 = arena.allocate((long) mr * nr * 8L);
            MemorySegment c2 = arena.allocate((long) mr * nr * 8L);

            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mr; r++) {
                    pA.set(VD, ((long) k * mr + r) * 8L, (double)(r + 1));
                }
                for (int c = 0; c < nr; c++) {
                    pB.set(VD, ((long) k * nr + c) * 8L, (double)(c + 1));
                }
            }

            MatMulMicroKernels.microKernel6x2vec_Double(pA, 0L, pB, 0L, c1, 0, 0, kc, nr, true);
            MatMulMicroKernels.microKernelScalar_Double(pA, 0L, 0, pB, 0L, c2, 0, 0, kc, nr, mr, nr, mr, nr, true);

            for (int r = 0; r < mr; r++) {
                for (int c = 0; c < nr; c++) {
                    double expected = (double)(r + 1) * (double)(c + 1) * kc;
                    assertEquals(expected, c1.get(VD, ((long) r * nr + c) * 8L), 1e-10);
                    assertEquals(expected, c2.get(VD, ((long) r * nr + c) * 8L), 1e-10);
                }
            }

            MemorySegment c2vec = arena.allocate(2L * nr * 8L);
            MatMulMicroKernels.microKernel2x2vec_Double(pA, 0L, 0, pB, 0L, c2vec, 0, 0, kc, nr, true);
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < nr; c++) {
                    double expected = (double)(r + 1) * (double)(c + 1) * kc;
                    assertEquals(expected, c2vec.get(VD, ((long) r * nr + c) * 8L), 1e-10);
                }
            }

            MemorySegment c1vec = arena.allocate((long) nr * 8L);
            MatMulMicroKernels.microKernel1x2vec_Double(pA, 0L, 0, pB, 0L, c1vec, 0, 0, kc, nr, true);
            for (int c = 0; c < nr; c++) {
                double expected = 1.0 * (double)(c + 1) * kc;
                assertEquals(expected, c1vec.get(VD, (long) c * 8L), 1e-10);
            }
        }
    }

    @Test
    @DisplayName("MicroKernel Int: 6x2vec and scalar fallback")
    void testMicroKernelsInt() {
        try (Arena arena = Arena.ofConfined()) {
            int kc = 12;
            int nr = NR_I32;
            int mr = MR;
            MemorySegment pA = arena.allocate((long) mr * kc * 4L);
            MemorySegment pB = arena.allocate((long) nr * kc * 4L);
            MemorySegment c1 = arena.allocate((long) mr * nr * 4L);
            MemorySegment c2 = arena.allocate((long) mr * nr * 4L);

            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mr; r++) {
                    pA.set(VI, ((long) k * mr + r) * 4L, r + 2);
                }
                for (int c = 0; c < nr; c++) {
                    pB.set(VI, ((long) k * nr + c) * 4L, c + 3);
                }
            }

            MatMulMicroKernels.microKernel6x2vec_Int(pA, 0L, pB, 0L, c1, 0, 0, kc, nr, true);
            MatMulMicroKernels.microKernelScalar_Int(pA, 0L, 0, pB, 0L, c2, 0, 0, kc, nr, mr, nr, mr, nr, true);

            for (int r = 0; r < mr; r++) {
                for (int c = 0; c < nr; c++) {
                    int expected = (r + 2) * (c + 3) * kc;
                    assertEquals(expected, c1.get(VI, ((long) r * nr + c) * 4L));
                    assertEquals(expected, c2.get(VI, ((long) r * nr + c) * 4L));
                }
            }

            MemorySegment c2vec = arena.allocate(2L * nr * 4L);
            MatMulMicroKernels.microKernel2x2vec_Int(pA, 0L, 0, pB, 0L, c2vec, 0, 0, kc, nr, true);
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < nr; c++) {
                    int expected = (r + 2) * (c + 3) * kc;
                    assertEquals(expected, c2vec.get(VI, ((long) r * nr + c) * 4L));
                }
            }

            MemorySegment c1vec = arena.allocate((long) nr * 4L);
            MatMulMicroKernels.microKernel1x2vec_Int(pA, 0L, 0, pB, 0L, c1vec, 0, 0, kc, nr, true);
            for (int c = 0; c < nr; c++) {
                int expected = 2 * (c + 3) * kc;
                assertEquals(expected, c1vec.get(VI, (long) c * 4L));
            }
        }
    }
}
