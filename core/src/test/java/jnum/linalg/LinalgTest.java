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
package jnum.linalg;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Linalg - Public Facade and Decompositions Tests")
class LinalgTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorCannotBeInstantiated() throws Exception {
        Constructor<Linalg> constructor = Linalg.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("matmul and zero-allocation destination buffer produce correct product")
    void testMatmul() {
        NDArray a = JNum.from(new float[]{
            1, 2,
            3, 4
        }, 2, 2);
        NDArray b = JNum.from(new float[]{
            2, 0,
            1, 2
        }, 2, 2);

        NDArray c = Linalg.matmul(a, b);
        assertEquals(4.0f, c.getFloat(0, 0));
        assertEquals(4.0f, c.getFloat(0, 1));
        assertEquals(10.0f, c.getFloat(1, 0));
        assertEquals(8.0f, c.getFloat(1, 1));

        NDArray out = JNum.zeros(2, 2);
        NDArray returned = Linalg.matmul(a, b, out);
        assertSame(out, returned);
        assertEquals(4.0f, out.getFloat(0, 0));
    }

    @Test
    @DisplayName("dot product and cosine similarity of 1D vectors")
    void testDotAndCosineSimilarity() {
        NDArray u = JNum.from(new float[]{1.0f, 2.0f, 3.0f}, 3);
        NDArray v = JNum.from(new float[]{4.0f, -5.0f, 6.0f}, 3);

        double dot = Linalg.dot(u, v);
        assertEquals(12.0, dot, 1e-5);

        NDArray uCollinear = JNum.from(new float[]{1.0f, 2.0f, 3.0f}, 3);
        NDArray vCollinear = JNum.from(new float[]{2.0f, 4.0f, 6.0f}, 3);
        assertEquals(1.0, Linalg.cosineSimilarity(uCollinear, vCollinear), 1e-5);
    }

    @Test
    @DisplayName("trace, norm, det, and slogdet")
    void testMatrixMetrics() {
        NDArray m = JNum.from(new double[]{
            2, 1,
            1, 2
        }, 2, 2);

        assertEquals(4.0, Linalg.trace(m), 1e-6);
        assertEquals(1.0, Linalg.trace(m, 1), 1e-6);

        NDArray vec = JNum.from(new double[]{3.0, 4.0}, 2);
        assertEquals(5.0, Linalg.norm(vec), 1e-6);
        assertEquals(7.0, Linalg.norm(vec, 1), 1e-6);

        double determinant = Linalg.det(m);
        assertEquals(3.0, determinant, 1e-6);

        SlogdetResult slog = Linalg.slogdet(m);
        assertEquals(1.0, slog.sign(), 1e-6);
        assertEquals(Math.log(3.0), slog.logAbsDet(), 1e-6);
    }

    @Test
    @DisplayName("inv, solve, pinv, matrixRank, and cond")
    void testInversesAndSolvers() {
        NDArray a = JNum.from(new double[]{
            4, 7,
            2, 6
        }, 2, 2);

        NDArray aInv = Linalg.inv(a);
        NDArray eye = Linalg.matmul(a, aInv);
        assertEquals(1.0, eye.getDouble(0, 0), 1e-5);
        assertEquals(0.0, eye.getDouble(0, 1), 1e-5);
        assertEquals(0.0, eye.getDouble(1, 0), 1e-5);
        assertEquals(1.0, eye.getDouble(1, 1), 1e-5);

        NDArray b = JNum.from(new double[]{1, 2}, 2, 1);
        NDArray x = Linalg.solve(a, b);
        NDArray bRecovered = Linalg.matmul(a, x);
        assertEquals(1.0, bRecovered.getDouble(0, 0), 1e-5);
        assertEquals(2.0, bRecovered.getDouble(1, 0), 1e-5);

        assertEquals(2, Linalg.matrixRank(a));
        assertTrue(Linalg.cond(a) > 0.0);

        NDArray pInv = Linalg.pinv(a);
        assertEquals(aInv.getDouble(0, 0), pInv.getDouble(0, 0), 1e-4);
    }

    @Test
    @DisplayName("Cholesky, QR, and SVD factorizations return valid records")
    void testDecompositions() {
        // Cholesky on positive-definite matrix
        NDArray symm = JNum.from(new double[]{
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        }, 3, 3);
        NDArray l = Linalg.cholesky(symm);
        NDArray lLt = Linalg.matmul(l, l.transpose());
        assertEquals(4.0, lLt.getDouble(0, 0), 1e-5);
        assertEquals(98.0, lLt.getDouble(2, 2), 1e-5);

        // QR Factorization
        NDArray matrix = JNum.from(new double[]{
            12, -51, 4,
            6, 167, -68,
            -4, 24, -41
        }, 3, 3);
        QRResult qr = Linalg.qr(matrix);
        assertNotNull(qr.q());
        assertNotNull(qr.r());
        NDArray reconstructed = Linalg.matmul(qr.q(), qr.r());
        assertEquals(12.0, reconstructed.getDouble(0, 0), 1e-4);
        assertEquals(167.0, reconstructed.getDouble(1, 1), 1e-4);

        // SVD Factorization
        SVDResult svd = Linalg.svd(matrix);
        assertNotNull(svd.u());
        assertNotNull(svd.s());
        assertNotNull(svd.vt());
        assertTrue(svd.s().getDouble(0) >= svd.s().getDouble(1));
    }

    @Test
    @DisplayName("Eigh and Eig eigenvalue decompositions")
    void testEigensystems() {
        NDArray cov = JNum.from(new double[]{
            2, 1,
            1, 2
        }, 2, 2);

        EighResult eigh = Linalg.eigh(cov);
        assertEquals(1.0, eigh.eigenvalues().getDouble(0), 1e-5);
        assertEquals(3.0, eigh.eigenvalues().getDouble(1), 1e-5);

        EigResult eig = Linalg.eig(cov);
        double r0 = eig.realEigenvalues().getDouble(0);
        double r1 = eig.realEigenvalues().getDouble(1);
        assertTrue((Math.abs(r0 - 1.0) < 1e-5 && Math.abs(r1 - 3.0) < 1e-5) ||
                   (Math.abs(r0 - 3.0) < 1e-5 && Math.abs(r1 - 1.0) < 1e-5));
    }

    @Test
    @DisplayName("Arena scoped memory overloads")
    void testArenaLifecycles() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = JNum.from(new double[]{2, 1, 1, 2}, 2, 2);
            NDArray inv = Linalg.inv(a, arena);
            QRResult qr = Linalg.qr(a, arena);
            SVDResult svd = Linalg.svd(a, arena);
            EighResult eigh = Linalg.eigh(a, arena);
            EigResult eig = Linalg.eig(a, arena);

            assertNotNull(inv);
            assertNotNull(qr.q());
            assertNotNull(svd.u());
            assertNotNull(eigh.eigenvalues());
            assertNotNull(eig.realEigenvalues());
        }
    }
}
