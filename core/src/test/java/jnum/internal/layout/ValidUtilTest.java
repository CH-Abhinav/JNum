package jnum.internal.layout;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ValidUtil - Input, Shape & Buffer Preconditions Tests")
class ValidUtilTest {

    // =========================================================================
    // 01. MatMul Input Validation
    // =========================================================================
    @Test
    @DisplayName("validateMatmulInputs accepts compatible 2D matrices")
    void matmulValid() {
        NDArray a = JNum.zeros(2, 3);
        NDArray b = JNum.zeros(3, 4);
        assertDoesNotThrow(() -> ValidUtil.validateMatmulInputs(a, b));
    }

    @Test
    @DisplayName("validateMatmulInputs rejects non-2D matrices")
    void matmulRejectsNon2D() {
        NDArray a1D = JNum.zeros(3);
        NDArray b2D = JNum.zeros(3, 3);
        NDArray c3D = JNum.zeros(2, 3, 3);

        assertThrows(IllegalArgumentException.class, () -> ValidUtil.validateMatmulInputs(a1D, b2D));
        assertThrows(IllegalArgumentException.class, () -> ValidUtil.validateMatmulInputs(b2D, a1D));
        assertThrows(IllegalArgumentException.class, () -> ValidUtil.validateMatmulInputs(c3D, b2D));
    }

    @Test
    @DisplayName("validateMatmulInputs rejects inner dimension mismatch")
    void matmulRejectsInnerDimMismatch() {
        NDArray a = JNum.zeros(2, 3);
        NDArray b = JNum.zeros(4, 5);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> ValidUtil.validateMatmulInputs(a, b));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    // =========================================================================
    // 02. DType Consistency Validation
    // =========================================================================
    @Test
    @DisplayName("checkSameDtype accepts identical types and rejects differing types")
    void checkSameDtypeTests() {
        NDArray aF32 = JNum.zeros(DType.f32, 2, 2);
        NDArray bF32 = JNum.zeros(DType.f32, 2, 2);
        NDArray cF64 = JNum.zeros(DType.f64, 2, 2);

        assertDoesNotThrow(() -> ValidUtil.checkSameDtype(aF32, bF32));
        assertThrows(IllegalArgumentException.class, () -> ValidUtil.checkSameDtype(aF32, cF64));
    }

    // =========================================================================
    // 03. Output Buffer Contiguity Validation
    // =========================================================================
    @Test
    @DisplayName("validateOutputBuffer rejects non-contiguous output buffers")
    void validateOutputBufferContiguity() {
        NDArray dense = JNum.zeros(2, 3);
        assertDoesNotThrow(() -> ValidUtil.validateOutputBuffer(dense));

        NDArray transposed = dense.transpose();
        assertThrows(IllegalArgumentException.class, () -> ValidUtil.validateOutputBuffer(transposed));
    }

    // =========================================================================
    // 04. Result Array Validation
    // =========================================================================
    @Test
    @DisplayName("validateResultArray checks shape and dtype matches")
    void validateResultArrayTests() {
        NDArray res = JNum.zeros(DType.f32, 2, 3);

        assertSame(res, ValidUtil.validateResultArray(res, DType.f32, new long[]{2, 3}));

        assertThrows(IllegalArgumentException.class,
            () -> ValidUtil.validateResultArray(res, DType.f64, new long[]{2, 3}));

        assertThrows(IllegalArgumentException.class,
            () -> ValidUtil.validateResultArray(res, DType.f32, new long[]{3, 2}));
    }

    // =========================================================================
    // 05. Broadcast Preparation
    // =========================================================================
    @Test
    @DisplayName("prepareBroadcastOperand broadcasts and casts or throws on incompatibility")
    void prepareBroadcastOperandTests() {
        NDArray a = JNum.from(new float[]{1f, 2f}, 1, 2);
        NDArray prepared = ValidUtil.prepareBroadcastOperand(a, new long[]{3, 2}, DType.f64);

        assertEquals(DType.f64, prepared.getDType());
        assertArrayEquals(new long[]{3, 2}, prepared.getShape());

        assertThrows(IllegalArgumentException.class,
            () -> ValidUtil.prepareBroadcastOperand(a, new long[]{3, 3}, DType.f64));
    }
}
