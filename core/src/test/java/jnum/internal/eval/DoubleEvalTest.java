package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class DoubleEvalTest {

    @Test
    @DisplayName("DoubleEval compiles and executes AST directly")
    void testDirectASTExecution() {
        // ($0 + $1) * 2.0
        ASTNode tree = new ASTNode.Mul(
                new ASTNode.Add(new ASTNode.Var(0), new ASTNode.Var(1)),
                new ASTNode.Const(2.0)
        );

        DoubleEval eval = new DoubleEval(tree);
        NDArray a = JNum.from(new double[]{1.5, 2.5}, 2);
        NDArray b = JNum.from(new double[]{4.5, 5.5}, 2);

        NDArray res = eval.execute(new NDArray[]{a, b});
        assertEquals(12.0, res.getDouble(0), 1e-12);
        assertEquals(16.0, res.getDouble(1), 1e-12);
    }

    @Test
    @DisplayName("SIMD boundary lane sizes (DoubleVector loop + scalar tail)")
    void testLaneBoundaries() {
        ASTNode fmaTree = new ASTNode.Fma(
                new ASTNode.Var(0),
                new ASTNode.Var(1),
                new ASTNode.Var(2)
        );
        DoubleEval eval = new DoubleEval(fmaTree);

        for (int size : new int[]{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}) {
            NDArray a = TestArrayFactory.random(100L + size, DType.f64, size);
            NDArray b = TestArrayFactory.random(200L + size, DType.f64, size);
            NDArray c = TestArrayFactory.random(300L + size, DType.f64, size);

            NDArray res = eval.execute(new NDArray[]{a, b, c});
            for (int i = 0; i < size; i++) {
                double expected = a.getDouble(i) * b.getDouble(i) + c.getDouble(i);
                assertEquals(expected, res.getDouble(i), 1e-9);
            }
        }
    }

    @Test
    @DisplayName("Division and Subtraction nodes in DoubleEval")
    void testSubAndDivNodes() {
        ASTNode tree = new ASTNode.Div(
                new ASTNode.Sub(new ASTNode.Var(0), new ASTNode.Const(2.0)),
                new ASTNode.Var(1)
        );
        DoubleEval eval = new DoubleEval(tree);

        NDArray a = JNum.from(new double[]{12.0, 22.0}, 2);
        NDArray b = JNum.from(new double[]{2.0, 4.0}, 2);
        NDArray res = eval.execute(new NDArray[]{a, b});

        assertEquals(5.0, res.getDouble(0), 1e-12);
        assertEquals(5.0, res.getDouble(1), 1e-12);
    }
}
