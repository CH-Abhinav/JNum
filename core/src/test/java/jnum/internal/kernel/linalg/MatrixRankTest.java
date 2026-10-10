package jnum.internal.kernel.linalg;

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

public class MatrixRankTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatrixRank> constructor = MatrixRank.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1 = JNum.zeros(arena, DType.f32, 5);
            assertThrows(IllegalArgumentException.class, () -> MatrixRank.matrixRank(v1, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> MatrixRank.matrixRank(t3, arena));
        }
    }

    @Test
    @DisplayName("Zero dimensions or zero matrix")
    void testZeroRank() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray empty = JNum.zeros(arena, DType.f32, 0, 3);
            assertEquals(0, MatrixRank.matrixRank(empty, arena));

            NDArray allZeros = JNum.zeros(arena, DType.f64, 4, 4);
            assertEquals(0, MatrixRank.matrixRank(allZeros, arena));
        }
    }

    @Test
    @DisplayName("Full-rank identity and diagonal matrices: Float and Double")
    void testFullRank() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray eyeF = TestArrayFactory.eye(5, DType.f32);
            assertEquals(5, MatrixRank.matrixRank(eyeF, arena));

            NDArray eyeD = TestArrayFactory.eye(8, DType.f64);
            assertEquals(8, MatrixRank.matrixRank(eyeD, arena));
        }
    }

    @Test
    @DisplayName("Rank-deficient matrices (rank 1 and rank 2)")
    void testRankDeficient() {
        try (Arena arena = Arena.ofConfined()) {
            // Rank-1 matrix: identical rows
            NDArray rank1 = TestArrayFactory.matrix(new float[][]{
                {1f, 2f, 3f},
                {1f, 2f, 3f},
                {1f, 2f, 3f}
            });
            assertEquals(1, MatrixRank.matrixRank(rank1, arena));

            // Rank-2 matrix: row2 = row0 + row1
            NDArray rank2 = TestArrayFactory.matrix(new double[][]{
                {1.0, 0.0, 2.0},
                {0.0, 1.0, 3.0},
                {1.0, 1.0, 5.0}
            });
            assertEquals(2, MatrixRank.matrixRank(rank2, arena));
        }
    }

    @Test
    @DisplayName("Custom tolerance threshold")
    void testCustomTolerance() {
        try (Arena arena = Arena.ofConfined()) {
            // Matrix with one very small singular value ~ 1e-4
            NDArray m = TestArrayFactory.matrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1e-4}
            });
            // With default machine epsilon, rank is 2
            assertEquals(2, MatrixRank.matrixRank(m, arena));
            // With tol = 1e-3, small singular value falls below threshold -> rank 1
            assertEquals(1, MatrixRank.matrixRank(m, 1e-3, arena));
        }
    }
}
