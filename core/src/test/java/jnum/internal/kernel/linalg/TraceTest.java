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

public class TraceTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Trace> constructor = Trace.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1 = JNum.zeros(arena, DType.f32, 5);
            assertThrows(IllegalArgumentException.class, () -> Trace.compute(v1, 0));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Trace.compute(t3, 0));
        }
    }

    @Test
    @DisplayName("Main diagonal trace (offset = 0) for f32, f64, i32, bool")
    void testMainDiagonal() {
        // Float
        NDArray fArr = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        assertEquals(15.0, Trace.compute(fArr, 0), 1e-6);

        // Double
        NDArray dArr = TestArrayFactory.matrix(new double[][]{
            {2.5, 0.0},
            {1.0, 3.5}
        });
        assertEquals(6.0, Trace.compute(dArr, 0), 1e-12);

        // Int
        NDArray iArr = TestArrayFactory.matrix(new int[][]{
            {10, 20},
            {30, 40}
        });
        assertEquals(50.0, Trace.compute(iArr, 0), 1e-12);

        // Bool (counts true along diagonal)
        NDArray bArr = TestArrayFactory.matrix(new boolean[][]{
            {true, false, true},
            {false, true, false},
            {true, false, false}
        });
        assertEquals(2.0, Trace.compute(bArr, 0), 1e-12);
    }

    @Test
    @DisplayName("Positive and negative offsets")
    void testOffsets() {
        // [ [1, 2, 3],
        //   [4, 5, 6],
        //   [7, 8, 9] ]
        NDArray m = TestArrayFactory.matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        });

        // Offset +1 (super-diagonal): entries (0,1)=2.0, (1,2)=6.0 -> sum = 8.0
        assertEquals(8.0, Trace.compute(m, 1), 1e-12);

        // Offset +2: entry (0,2)=3.0 -> sum = 3.0
        assertEquals(3.0, Trace.compute(m, 2), 1e-12);

        // Offset -1 (sub-diagonal): entries (1,0)=4.0, (2,1)=8.0 -> sum = 12.0
        assertEquals(12.0, Trace.compute(m, -1), 1e-12);

        // Offset -2: entry (2,0)=7.0 -> sum = 7.0
        assertEquals(7.0, Trace.compute(m, -2), 1e-12);
    }

    @Test
    @DisplayName("Out-of-bounds offset returns 0.0")
    void testOutOfBoundsOffset() {
        NDArray m = TestArrayFactory.eye(3, DType.f64);
        assertEquals(0.0, Trace.compute(m, 3), 1e-12);
        assertEquals(0.0, Trace.compute(m, -3), 1e-12);
        assertEquals(0.0, Trace.compute(m, 100), 1e-12);
        assertEquals(0.0, Trace.compute(m, -100), 1e-12);
    }

    @Test
    @DisplayName("Rectangular matrices and transposed view")
    void testRectangularAndTransposed() {
        NDArray rect = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f, 4f},
            {5f, 6f, 7f, 8f}
        });
        // Main diagonal has 2 elements: (0,0)=1, (1,1)=6 -> 7
        assertEquals(7.0, Trace.compute(rect, 0), 1e-6);

        // Transposed trace(A^T) == trace(A)
        NDArray sq = TestArrayFactory.matrix(new double[][]{
            {1.0, 5.0},
            {2.0, 4.0}
        });
        assertEquals(Trace.compute(sq, 0), Trace.compute(sq.transpose(), 0), 1e-12);
    }
}
