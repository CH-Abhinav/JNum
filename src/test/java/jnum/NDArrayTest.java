package jnum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NDArrayTest {

    @Test
    void viewSemanticsAndCopyPreserveLogicalElements() {
        NDArray dense = NDArray.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray transposed = dense.transpose();
        NDArray broadcast = NDArray.from(new float[]{5f, 6f}, 1, 2).broadcastTo(3, 2);

        assertTrue(dense.isContiguous());
        assertFalse(transposed.isContiguous());
        assertFalse(broadcast.isContiguous());

        NDArray copied = transposed.copy();

        assertTrue(copied.isContiguous());
        assertEquals(DType.FLOAT, copied.getDType());
        assertEquals(1f, copied.getFloat(0, 0), 1e-6f);
        assertEquals(3f, copied.getFloat(0, 1), 1e-6f);
        assertEquals(2f, copied.getFloat(1, 0), 1e-6f);
        assertEquals(4f, copied.getFloat(1, 1), 1e-6f);
    }

    @Test
    void dotPromotesMixedDtypesCorrectly() {
        NDArray left = NDArray.from(new float[]{1f, 2f, 3f}, 3);
        NDArray right = NDArray.from(new double[]{0.5, 1.5, 2.0}, 3);

        double result = left.dot(right);

        assertEquals(9.5, result, 1e-9);
    }

    @Test
    void matmulPromotesMixedDtypesCorrectly() {
        NDArray left = NDArray.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray right = NDArray.from(new double[]{5.0, 6.0, 7.0, 8.0}, 2, 2);

        NDArray result = left.matmul(right);

        assertEquals(DType.DOUBLE, result.getDType());
        assertArrayEquals(new int[]{2, 2}, result.shape());
        assertEquals(19.0, result.get(0, 0), 1e-9);
        assertEquals(22.0, result.get(0, 1), 1e-9);
        assertEquals(43.0, result.get(1, 0), 1e-9);
        assertEquals(50.0, result.get(1, 1), 1e-9);
    }

    @Test
    void matmulRejectsNonContiguousOutputBuffer() {
        NDArray left = NDArray.ones(DType.FLOAT, 2, 2);
        NDArray right = NDArray.ones(DType.FLOAT, 2, 2);
        NDArray nonContiguousOutput = NDArray.zeros(DType.FLOAT, 2, 2).transpose();

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> left.matmul(right, nonContiguousOutput)
        );

        assertTrue(ex.getMessage().contains("contiguous"));
    }
}
