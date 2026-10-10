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

public class QRTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<QR> constructor = QR.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1D = JNum.zeros(arena, DType.f32, 5);
            assertThrows(IllegalArgumentException.class, () -> QR.qr(v1D, arena));

            NDArray t3D = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> QR.qr(t3D, arena));
        }
    }

    @Test
    @DisplayName("Square matrix QR: Float and Double")
    void testSquareQR() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 4;
            NDArray aF = TestArrayFactory.random(701L, DType.f32, n, n);
            QR.QRResult qrF = QR.qr(aF, arena);
            NDArray qF = qrF.q();
            NDArray rF = qrF.r();

            assertEquals(n, qF.internalShapeUnsafe()[0]);
            assertEquals(n, qF.internalShapeUnsafe()[1]);
            assertEquals(n, rF.internalShapeUnsafe()[0]);
            assertEquals(n, rF.internalShapeUnsafe()[1]);

            // Check Q * R == A
            NDArray reconF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(qF, rF, reconF);
            TestAssertions.assertNDArrayClose(aF, reconF, 1e-4, 1e-4);

            // Check Q^T * Q == I
            NDArray qtqF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(qF.transpose(), qF, qtqF);
            NDArray eyeF = TestArrayFactory.eye(n, DType.f32);
            TestAssertions.assertNDArrayClose(eyeF, qtqF, 1e-4, 1e-4);

            // Check R is upper triangular
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    assertEquals(0.0f, rF.getFloat(i, j), 1e-4f);
                }
            }

            // Double
            NDArray aD = TestArrayFactory.random(702L, DType.f64, n, n);
            QR.QRResult qrD = QR.qr(aD, arena);
            NDArray qD = qrD.q();
            NDArray rD = qrD.r();

            NDArray reconD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(qD, rD, reconD);
            TestAssertions.assertNDArrayClose(aD, reconD, 1e-9, 1e-9);

            NDArray qtqD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(qD.transpose(), qD, qtqD);
            NDArray eyeD = TestArrayFactory.eye(n, DType.f64);
            TestAssertions.assertNDArrayClose(eyeD, qtqD, 1e-9, 1e-9);
        }
    }

    @Test
    @DisplayName("Rectangular matrix QR: Tall (m > n)")
    void testTallQR() {
        try (Arena arena = Arena.ofConfined()) {
            int m = 6, n = 3;
            NDArray a = TestArrayFactory.random(703L, DType.f64, m, n);
            QR.QRResult res = QR.qr(a, arena);
            NDArray q = res.q();
            NDArray r = res.r();

            assertEquals(m, q.internalShapeUnsafe()[0]);
            assertEquals(m, q.internalShapeUnsafe()[1]);
            assertEquals(m, r.internalShapeUnsafe()[0]);
            assertEquals(n, r.internalShapeUnsafe()[1]);

            // Q * R == A
            NDArray recon = JNum.zeros(arena, DType.f64, m, n);
            MatMul.matmulDouble(q, r, recon);
            TestAssertions.assertNDArrayClose(a, recon, 1e-9, 1e-9);

            // Q^T * Q == I_m
            NDArray qtq = JNum.zeros(arena, DType.f64, m, m);
            MatMul.matmulDouble(q.transpose(), q, qtq);
            NDArray eyeM = TestArrayFactory.eye(m, DType.f64);
            TestAssertions.assertNDArrayClose(eyeM, qtq, 1e-9, 1e-9);
        }
    }

    @Test
    @DisplayName("Non-contiguous view QR decomposition")
    void testNonContiguousQR() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray base = TestArrayFactory.random(704L, DType.f32, 5, 5);
            NDArray aT = base.transpose();
            QR.QRResult res = QR.qr(aT, arena);

            NDArray recon = JNum.zeros(arena, DType.f32, 5, 5);
            MatMul.matmulFloat(res.q(), res.r(), recon);
            TestAssertions.assertNDArrayClose(aT, recon, 1e-4, 1e-4);
        }
    }
}
