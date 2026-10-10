package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class IntEvalTest {

    @Test
    @DisplayName("IntEval compiles and executes AST directly")
    void testDirectASTExecution() {
        // ($0 + $1) * 2
        ASTNode tree = new ASTNode.Mul(
                new ASTNode.Add(new ASTNode.Var(0), new ASTNode.Var(1)),
                new ASTNode.Const(2.0)
        );

        IntEval eval = new IntEval(tree);
        NDArray a = JNum.from(new int[]{1, 2, 3}, 3);
        NDArray b = JNum.from(new int[]{4, 5, 6}, 3);

        NDArray res = eval.execute(new NDArray[]{a, b});
        assertEquals(10, res.getInt(0));
        assertEquals(14, res.getInt(1));
        assertEquals(18, res.getInt(2));
    }

    @Test
    @DisplayName("SIMD boundary lane sizes (IntVector loop + scalar tail)")
    void testLaneBoundaries() {
        ASTNode fmaTree = new ASTNode.Fma(
                new ASTNode.Var(0),
                new ASTNode.Var(1),
                new ASTNode.Var(2)
        );
        IntEval eval = new IntEval(fmaTree);

        for (int size : new int[]{1, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
            NDArray a = TestArrayFactory.random(500L + size, DType.i32, size);
            NDArray b = TestArrayFactory.random(600L + size, DType.i32, size);
            NDArray c = TestArrayFactory.random(700L + size, DType.i32, size);

            NDArray res = eval.execute(new NDArray[]{a, b, c});
            for (int i = 0; i < size; i++) {
                int expected = a.getInt(i) * b.getInt(i) + c.getInt(i);
                assertEquals(expected, res.getInt(i));
            }
        }
    }

    @Test
    @DisplayName("Division and Subtraction nodes in IntEval")
    void testSubAndDivNodes() {
        ASTNode tree = new ASTNode.Div(
                new ASTNode.Sub(new ASTNode.Var(0), new ASTNode.Const(1.0)),
                new ASTNode.Var(1)
        );
        IntEval eval = new IntEval(tree);

        NDArray a = JNum.from(new int[]{11, 21}, 2);
        NDArray b = JNum.from(new int[]{2, 4}, 2);
        NDArray res = eval.execute(new NDArray[]{a, b});

        assertEquals(5, res.getInt(0)); // (11 - 1) / 2 = 5
        assertEquals(5, res.getInt(1)); // (21 - 1) / 4 = 5
    }
}
