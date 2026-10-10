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

public class EigTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Eig> constructor = Eig.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-square or non-2D matrices throw IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray rect = JNum.zeros(arena, DType.f32, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> Eig.eig(rect, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Eig.eig(t3, arena));
        }
    }

    @Test
    @DisplayName("Boundary dimensions: 0x0 and 1x1 matrices")
    void testBoundaryDimensions() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray empty = JNum.zeros(arena, DType.f32, 0, 0);
            Eig.EigResult resEmpty = Eig.eig(empty, arena);
            assertEquals(0, resEmpty.realEigenvalues().getSize());
            assertEquals(0, resEmpty.imagEigenvalues().getSize());

            NDArray singleF = TestArrayFactory.matrix(new float[][]{{7.5f}});
            Eig.EigResult resSingleF = Eig.eig(singleF, arena);
            assertEquals(7.5f, resSingleF.realEigenvalues().getFloat(0), 1e-6f);
            assertEquals(0.0f, resSingleF.imagEigenvalues().getFloat(0), 1e-6f);

            NDArray singleD = TestArrayFactory.matrix(new double[][]{{-3.2}});
            Eig.EigResult resSingleD = Eig.eig(singleD, arena);
            assertEquals(-3.2, resSingleD.realEigenvalues().getDouble(0), 1e-12);
            assertEquals(0.0, resSingleD.imagEigenvalues().getDouble(0), 1e-12);
        }
    }

    @Test
    @DisplayName("Diagonal matrix yields diagonal elements as real eigenvalues")
    void testDiagonalMatrix() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray diagF = TestArrayFactory.matrix(new float[][]{
                {2.0f, 0.0f},
                {0.0f, 5.0f}
            });
            Eig.EigResult resF = Eig.eig(diagF, arena);
            float r0 = resF.realEigenvalues().getFloat(0);
            float r1 = resF.realEigenvalues().getFloat(1);
            // Eigenvalues could be [2, 5] or [5, 2]
            assertTrue((Math.abs(r0 - 2.0f) < 1e-4f && Math.abs(r1 - 5.0f) < 1e-4f) ||
                       (Math.abs(r0 - 5.0f) < 1e-4f && Math.abs(r1 - 2.0f) < 1e-4f));
            assertEquals(0.0f, resF.imagEigenvalues().getFloat(0), 1e-4f);
            assertEquals(0.0f, resF.imagEigenvalues().getFloat(1), 1e-4f);
        }
    }

    @Test
    @DisplayName("2x2 rotation matrix yields complex conjugate eigenvalues (lambda = +/- i)")
    void testComplexEigenvalues() {
        try (Arena arena = Arena.ofConfined()) {
            // [[0, -1], [1, 0]] has eigenvalues 0 + 1i and 0 - 1i
            NDArray rot = TestArrayFactory.matrix(new double[][]{
                {0.0, -1.0},
                {1.0, 0.0}
            });
            Eig.EigResult res = Eig.eig(rot, arena);
            assertEquals(0.0, res.realEigenvalues().getDouble(0), 1e-6);
            assertEquals(0.0, res.realEigenvalues().getDouble(1), 1e-6);

            double im0 = res.imagEigenvalues().getDouble(0);
            double im1 = res.imagEigenvalues().getDouble(1);
            assertEquals(1.0, Math.abs(im0), 1e-6);
            assertEquals(1.0, Math.abs(im1), 1e-6);
            assertEquals(0.0, im0 + im1, 1e-6); // Conjugate pair sum to 0
        }
    }
}
