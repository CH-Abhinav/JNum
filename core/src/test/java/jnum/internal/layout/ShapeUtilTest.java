package jnum.internal.layout;

import jnum.DType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ShapeUtil - Strides, Broadcasting & Reductions Tests")
class ShapeUtilTest {

    // =========================================================================
    // 01. Default Stride Calculations
    // =========================================================================
    @Test
    @DisplayName("calculateDefaultStrides computes correct row-major strides")
    void defaultStrides() {
        assertArrayEquals(new long[0], ShapeUtil.calculateDefaultStrides(new long[0]));
        assertArrayEquals(new long[]{1}, ShapeUtil.calculateDefaultStrides(new long[]{10}));
        assertArrayEquals(new long[]{3, 1}, ShapeUtil.calculateDefaultStrides(new long[]{2, 3}));
        assertArrayEquals(new long[]{12, 4, 1}, ShapeUtil.calculateDefaultStrides(new long[]{2, 3, 4}));
        assertArrayEquals(new long[]{60, 20, 5, 1}, ShapeUtil.calculateDefaultStrides(new long[]{2, 3, 4, 5}));
    }

    // =========================================================================
    // 02. Broadcasting Rules & Incompatible Rejection
    // =========================================================================
    @Test
    @DisplayName("calculateBroadcastShape handles identical, singleton, and rank-expanded shapes")
    void broadcastShapeValid() {
        assertArrayEquals(new long[]{2, 3},
            ShapeUtil.calculateBroadcastShape(new long[]{2, 3}, new long[]{2, 3}));

        assertArrayEquals(new long[]{2, 3},
            ShapeUtil.calculateBroadcastShape(new long[]{2, 1}, new long[]{1, 3}));

        assertArrayEquals(new long[]{2, 3},
            ShapeUtil.calculateBroadcastShape(new long[]{3}, new long[]{2, 3}));

        assertArrayEquals(new long[]{5, 2, 3},
            ShapeUtil.calculateBroadcastShape(new long[]{5, 1, 3}, new long[]{2, 3}));
    }

    @Test
    @DisplayName("calculateBroadcastShape rejects incompatible dimensions with IllegalArgumentException")
    void broadcastShapeIncompatible() {
        assertThrows(IllegalArgumentException.class,
            () -> ShapeUtil.calculateBroadcastShape(new long[]{2, 3}, new long[]{2, 4}));

        assertThrows(IllegalArgumentException.class,
            () -> ShapeUtil.calculateBroadcastShape(new long[]{5, 2}, new long[]{5, 3}));
    }

    // =========================================================================
    // 03. Reduction Shape Calculations
    // =========================================================================
    @Test
    @DisplayName("calculateReductionShape drops specified axis correctly")
    void reductionShapeValid() {
        assertArrayEquals(new long[]{3, 4},
            ShapeUtil.calculateReductionShape(new long[]{2, 3, 4}, 0));

        assertArrayEquals(new long[]{2, 4},
            ShapeUtil.calculateReductionShape(new long[]{2, 3, 4}, 1));

        assertArrayEquals(new long[]{2, 3},
            ShapeUtil.calculateReductionShape(new long[]{2, 3, 4}, 2));
    }

    @Test
    @DisplayName("calculateReductionShape rejects out-of-bounds axis")
    void reductionShapeAxisOutOfBounds() {
        assertThrows(IllegalArgumentException.class,
            () -> ShapeUtil.calculateReductionShape(new long[]{2, 3, 4}, -1));

        assertThrows(IllegalArgumentException.class,
            () -> ShapeUtil.calculateReductionShape(new long[]{2, 3, 4}, 3));
    }

    // =========================================================================
    // 04. Byte Offset Calculations
    // =========================================================================
    @Test
    @DisplayName("getByteOffset computes exact linear byte position")
    void byteOffset() {
        long[] coords = {1, 2};
        long[] strides = {3, 1};

        // (1 * 3 + 2 * 1) * 4 = 20
        assertEquals(20L, ShapeUtil.getByteOffset(coords, strides, DType.f32));
        // (1 * 3 + 2 * 1) * 8 = 40
        assertEquals(40L, ShapeUtil.getByteOffset(coords, strides, DType.f64));
        // (1 * 3 + 2 * 1) * 1 = 5
        assertEquals(5L, ShapeUtil.getByteOffset(coords, strides, DType.bool));
    }
}
