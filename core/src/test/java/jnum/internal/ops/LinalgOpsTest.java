package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;

public class LinalgOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<LinalgOps> constructor = LinalgOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Matmul with promotion: i32 * f32 -> f32")
    void testMatmulPromotion() {
        NDArray a = JNum.from(new int[]{1, 2, 3, 4}, 2, 2);
        NDArray b = JNum.from(new float[]{0.5f, 0f, 0f, 0.5f}, 2, 2);
        NDArray c = LinalgOps.matmul(a, b);
        assertEquals(DType.f32, c.getDType());
        assertEquals(0.5f, c.getFloat(0, 0), 1e-6f);
        assertEquals(1.0f, c.getFloat(0, 1), 1e-6f);
        assertEquals(1.5f, c.getFloat(1, 0), 1e-6f);
        assertEquals(2.0f, c.getFloat(1, 1), 1e-6f);
    }

    @Test
    @DisplayName("Det and Inv high-level facade")
    void testDetAndInv() {
        NDArray a = TestArrayFactory.matrix(new double[][]{{4.0, 7.0}, {2.0, 6.0}});
        double d = LinalgOps.det(a);
        assertEquals(10.0, d, 1e-9);

        NDArray invA = LinalgOps.inv(a);
        NDArray prod = LinalgOps.matmul(a, invA);
        NDArray eye = TestArrayFactory.eye(2, DType.f64);
        TestAssertions.assertNDArrayClose(eye, prod, 1e-9, 1e-9);
    }

    @Test
    @DisplayName("Solve, QR, Cholesky, SVD, Norm, Cond, Pinv, Eig, Eigh, Trace, CosineSimilarity facades")
    void testAllLinalgFacades() {
        // Solve
        NDArray a = TestArrayFactory.matrix(new double[][]{{2.0, 1.0}, {-3.0, 4.0}});
        NDArray b = JNum.from(new double[]{5.0, 9.0}, 2);
        NDArray x = LinalgOps.solve(a, b);
        assertEquals(1.0, x.getDouble(0), 1e-6);
        assertEquals(3.0, x.getDouble(1), 1e-6);

        // QR
        var qrRes = LinalgOps.qr(a);
        assertNotNull(qrRes.q());
        assertNotNull(qrRes.r());

        // Cholesky on SPD
        NDArray spd = TestArrayFactory.spd(3);
        NDArray l = LinalgOps.cholesky(spd);
        assertNotNull(l);

        // SVD
        var svdRes = LinalgOps.svd(a);
        assertNotNull(svdRes.u());
        assertNotNull(svdRes.s());
        assertNotNull(svdRes.vt());

        // Norm
        double n2 = LinalgOps.norm(b, 2);
        assertEquals(Math.sqrt(25.0 + 81.0), n2, 1e-6);

        // Cond
        double condNum = LinalgOps.cond(TestArrayFactory.eye(3, DType.f64));
        assertEquals(1.0, condNum, 1e-6);

        // Pinv
        NDArray pinvA = LinalgOps.pinv(a);
        assertNotNull(pinvA);

        // Eig and Eigh
        var eigRes = LinalgOps.eig(a);
        assertNotNull(eigRes.realEigenvalues());
        var eighRes = LinalgOps.eigh(spd);
        assertNotNull(eighRes.eigenvalues());

        // Trace
        assertEquals(6.0, LinalgOps.trace(a, 0), 1e-9);

        // CosineSimilarity
        NDArray v1 = JNum.from(new double[]{1.0, 0.0}, 2);
        NDArray v2 = JNum.from(new double[]{0.0, 1.0}, 2);
        assertEquals(0.0, LinalgOps.cosineSimilarity(v1, v2), 1e-9);
    }
}
