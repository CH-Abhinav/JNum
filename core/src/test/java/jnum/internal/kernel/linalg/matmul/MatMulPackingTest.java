package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class MatMulPackingTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulPacking> constructor = MatMulPacking.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Packing Float: Panel A and Panel B (contiguous and strided)")
    void testPackingFloat() {
        try (Arena arena = Arena.ofConfined()) {
            int mc = 8; // MR=6, 1 full panel + 2 tail rows
            int kc = 12;
            int nc = NR_F32 + 3; // 1 full panel + 3 tail cols

            // Test Pack A contiguous
            MemorySegment srcA = arena.allocate((long) mc * kc * 4L);
            MemorySegment dstA = arena.allocate((long) ((mc + MR - 1) / MR) * MR * kc * 4L);
            for (int r = 0; r < mc; r++) {
                for (int k = 0; k < kc; k++) {
                    srcA.set(VF, ((long) r * kc + k) * 4L, (float)(r * 100 + k));
                }
            }
            MatMulPacking.packA_panel_Float(srcA, dstA, 0, mc, 0, kc, kc, 1L);

            // Verify full panel A: for panel 0, layout is k * MR + r
            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < MR; r++) {
                    float expected = (float)(r * 100 + k);
                    assertEquals(expected, dstA.get(VF, ((long) k * MR + r) * 4L), 1e-6f);
                }
            }
            // Verify tail rows in second panel
            long tailBase = (long) MR * kc * 4L;
            for (int r = 0; r < 2; r++) {
                for (int k = 0; k < kc; k++) {
                    float expected = (float)((6 + r) * 100 + k);
                    assertEquals(expected, dstA.get(VF, tailBase + ((long) k * MR + r) * 4L), 1e-6f);
                }
            }
            // Verify zero-padding in tail rows
            for (int r = 2; r < MR; r++) {
                for (int k = 0; k < kc; k++) {
                    assertEquals(0.0f, dstA.get(VF, tailBase + ((long) k * MR + r) * 4L), 1e-6f);
                }
            }

            // Test Pack B (both sequential and parallel paths)
            MemorySegment srcB = arena.allocate((long) kc * nc * 4L);
            MemorySegment dstBSeq = arena.allocate((long) ((nc + NR_F32 - 1) / NR_F32) * NR_F32 * kc * 4L);
            MemorySegment dstBPar = arena.allocate((long) ((nc + NR_F32 - 1) / NR_F32) * NR_F32 * kc * 4L);
            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < nc; c++) {
                    srcB.set(VF, ((long) k * nc + c) * 4L, (float)(k * 1000 + c));
                }
            }

            MatMulPacking.packB_panel_sequential_Float(srcB, dstBSeq, 0, 0, kc, nc, nc, 1L);
            MatMulPacking.packB_panel_Float(srcB, dstBPar, 0, 0, kc, nc, nc, 1L);

            // Both sequential and parallel packings should be bit-for-bit identical
            for (long offset = 0; offset < dstBSeq.byteSize(); offset += 4L) {
                assertEquals(dstBSeq.get(VF, offset), dstBPar.get(VF, offset));
            }

            // Verify full panel B
            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < NR_F32; c++) {
                    float expected = (float)(k * 1000 + c);
                    assertEquals(expected, dstBSeq.get(VF, ((long) k * NR_F32 + c) * 4L), 1e-6f);
                }
            }
            // Verify tail panel B zero-padding
            long tailBBase = (long) NR_F32 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < 3; c++) {
                    float expected = (float)(k * 1000 + NR_F32 + c);
                    assertEquals(expected, dstBSeq.get(VF, tailBBase + ((long) k * NR_F32 + c) * 4L), 1e-6f);
                }
                for (int c = 3; c < NR_F32; c++) {
                    assertEquals(0.0f, dstBSeq.get(VF, tailBBase + ((long) k * NR_F32 + c) * 4L), 1e-6f);
                }
            }
        }
    }

    @Test
    @DisplayName("Packing Double: Panel A and Panel B")
    void testPackingDouble() {
        try (Arena arena = Arena.ofConfined()) {
            int mc = 7;
            int kc = 10;
            int nc = NR_F64 + 2;

            MemorySegment srcA = arena.allocate((long) mc * kc * 8L);
            MemorySegment dstA = arena.allocate((long) ((mc + MR - 1) / MR) * MR * kc * 8L);
            for (int r = 0; r < mc; r++) {
                for (int k = 0; k < kc; k++) {
                    srcA.set(VD, ((long) r * kc + k) * 8L, (double)(r * 50 + k));
                }
            }
            MatMulPacking.packA_panel_Double(srcA, dstA, 0, mc, 0, kc, kc, 1L);

            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < MR; r++) {
                    double expected = (double)(r * 50 + k);
                    assertEquals(expected, dstA.get(VD, ((long) k * MR + r) * 8L), 1e-12);
                }
            }

            MemorySegment srcB = arena.allocate((long) kc * nc * 8L);
            MemorySegment dstB = arena.allocate((long) ((nc + NR_F64 - 1) / NR_F64) * NR_F64 * kc * 8L);
            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < nc; c++) {
                    srcB.set(VD, ((long) k * nc + c) * 8L, (double)(k * 100 + c));
                }
            }
            MatMulPacking.packB_panel_sequential_Double(srcB, dstB, 0, 0, kc, nc, nc, 1L);

            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < NR_F64; c++) {
                    double expected = (double)(k * 100 + c);
                    assertEquals(expected, dstB.get(VD, ((long) k * NR_F64 + c) * 8L), 1e-12);
                }
            }
        }
    }

    @Test
    @DisplayName("Packing Int: Panel A and Panel B")
    void testPackingInt() {
        try (Arena arena = Arena.ofConfined()) {
            int mc = 8;
            int kc = 8;
            int nc = NR_I32 + 1;

            MemorySegment srcA = arena.allocate((long) mc * kc * 4L);
            MemorySegment dstA = arena.allocate((long) ((mc + MR - 1) / MR) * MR * kc * 4L);
            for (int r = 0; r < mc; r++) {
                for (int k = 0; k < kc; k++) {
                    srcA.set(VI, ((long) r * kc + k) * 4L, r * 20 + k);
                }
            }
            MatMulPacking.packA_panel_Int(srcA, dstA, 0, mc, 0, kc, kc, 1L);

            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < MR; r++) {
                    assertEquals(r * 20 + k, dstA.get(VI, ((long) k * MR + r) * 4L));
                }
            }

            MemorySegment srcB = arena.allocate((long) kc * nc * 4L);
            MemorySegment dstB = arena.allocate((long) ((nc + NR_I32 - 1) / NR_I32) * NR_I32 * kc * 4L);
            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < nc; c++) {
                    srcB.set(VI, ((long) k * nc + c) * 4L, k * 50 + c);
                }
            }
            MatMulPacking.packB_panel_sequential_Int(srcB, dstB, 0, 0, kc, nc, nc, 1L);

            for (int k = 0; k < kc; k++) {
                for (int c = 0; c < NR_I32; c++) {
                    assertEquals(k * 50 + c, dstB.get(VI, ((long) k * NR_I32 + c) * 4L));
                }
            }
        }
    }
}
