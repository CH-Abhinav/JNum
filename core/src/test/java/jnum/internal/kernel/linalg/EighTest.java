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
import jnum.testutil.TestAssertions;

public class EighTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Eigh> constructor = Eigh.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-square or non-2D matrices throw IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray rect = JNum.zeros(arena, DType.f32, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> Eigh.eigh(rect, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Eigh.eigh(t3, arena));
        }
    }

    @Test
    @DisplayName("1x1 symmetric matrix")
    void test1x1Matrix() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray mF = TestArrayFactory.matrix(new float[][]{{4.5f}});
            Eigh.EighResult resF = Eigh.eigh(mF, arena);
            assertEquals(4.5f, resF.eigenvalues().getFloat(0), 1e-6f);
            assertEquals(1.0f, resF.eigenvectors().getFloat(0, 0), 1e-6f);

            NDArray mD = TestArrayFactory.matrix(new double[][]{{-2.5}});
            Eigh.EighResult resD = Eigh.eigh(mD, arena);
            assertEquals(-2.5, resD.eigenvalues().getDouble(0), 1e-12);
            assertEquals(1.0, resD.eigenvectors().getDouble(0, 0), 1e-12);
        }
    }

    @Test
    @DisplayName("Known 2x2 symmetric matrix: A * v == lambda * v and V^T * V == I")
    void test2x2Symmetric() {
        try (Arena arena = Arena.ofConfined()) {
            // A = [[2, 1], [1, 2]] -> eigenvalues 1 and 3
            NDArray aF = TestArrayFactory.matrix(new float[][]{
                {2.0f, 1.0f},
                {1.0f, 2.0f}
            });
            Eigh.EighResult resF = Eigh.eigh(aF, arena);
            NDArray wF = resF.eigenvalues();
            NDArray vF = resF.eigenvectors();

            float l0 = wF.getFloat(0);
            float l1 = wF.getFloat(1);
            assertTrue((Math.abs(l0 - 1.0f) < 1e-4f && Math.abs(l1 - 3.0f) < 1e-4f) ||
                       (Math.abs(l0 - 3.0f) < 1e-4f && Math.abs(l1 - 1.0f) < 1e-4f));

            // V^T * V == I
            NDArray vtvF = JNum.zeros(arena, DType.f32, 2, 2);
            MatMul.matmulFloat(vF.transpose(), vF, vtvF);
            NDArray eye2F = TestArrayFactory.eye(2, DType.f32);
            TestAssertions.assertNDArrayClose(eye2F, vtvF, 1e-4, 1e-4);
        }
    }

    @Test
    @DisplayName("Random SPD matrix: A * v_i == lambda_i * v_i for all eigenvectors")
    void testSpdEigenpairs() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 4;
            NDArray aD = TestArrayFactory.spd(n);
            Eigh.EighResult res = Eigh.eigh(aD, arena);
            NDArray w = res.eigenvalues();
            NDArray v = res.eigenvectors();

            // Verify each eigenvector: A * v[:, i] == w[i] * v[:, i]
            for (int i = 0; i < n; i++) {
                double lambda = w.getDouble(i);
                NDArray vi = v.slice(":, " + i).contiguous(); // column i

                // A * vi
                double[] Avi = new double[n];
                for (int r = 0; r < n; r++) {
                    for (int c = 0; c < n; c++) {
                        Avi[r] += aD.getDouble(r, c) * vi.getDouble(c);
                    }
                }

                // lambda * vi
                double[] lvi = new double[n];
                for (int r = 0; r < n; r++) {
                    lvi[r] = lambda * vi.getDouble(r);
                }

                TestAssertions.assertArrayClose(lvi, Avi, 1e-4, 1e-4);
            }
        }
    }
}
