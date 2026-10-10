package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class FloatEvalTest {

    @Test
    @DisplayName("FloatEval compiles and executes AST directly")
    void testDirectASTExecution() {
        // ($0 + $1) * 2.0f
        ASTNode tree = new ASTNode.Mul(
                new ASTNode.Add(new ASTNode.Var(0), new ASTNode.Var(1)),
                new ASTNode.Const(2.0)
        );

        FloatEval eval = new FloatEval(tree);
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray b = JNum.from(new float[]{4f, 5f, 6f}, 3);

        NDArray res = eval.execute(new NDArray[]{a, b});
        assertEquals(10f, res.getFloat(0), 1e-6f);
        assertEquals(14f, res.getFloat(1), 1e-6f);
        assertEquals(18f, res.getFloat(2), 1e-6f);
    }

    @Test
    @DisplayName("SIMD boundary lane sizes (vector loop + scalar tail)")
    void testLaneBoundaries() {
        // FMA: a * b + c
        ASTNode fmaTree = new ASTNode.Fma(
                new ASTNode.Var(0),
                new ASTNode.Var(1),
                new ASTNode.Var(2)
        );
        FloatEval eval = new FloatEval(fmaTree);

        for (int size : new int[]{1, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
            NDArray a = TestArrayFactory.random(10L + size, DType.f32, size);
            NDArray b = TestArrayFactory.random(20L + size, DType.f32, size);
            NDArray c = TestArrayFactory.random(30L + size, DType.f32, size);

            NDArray res = eval.execute(new NDArray[]{a, b, c});
            for (int i = 0; i < size; i++) {
                float expected = a.getFloat(i) * b.getFloat(i) + c.getFloat(i);
                assertEquals(expected, res.getFloat(i), 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("Division and Subtraction nodes in FloatEval")
    void testSubAndDivNodes() {
        ASTNode tree = new ASTNode.Div(
                new ASTNode.Sub(new ASTNode.Var(0), new ASTNode.Const(1.0)),
                new ASTNode.Var(1)
        );
        FloatEval eval = new FloatEval(tree);

        NDArray a = JNum.from(new float[]{11f, 21f}, 2);
        NDArray b = JNum.from(new float[]{2f, 4f}, 2);
        NDArray res = eval.execute(new NDArray[]{a, b});

        assertEquals(5.0f, res.getFloat(0), 1e-6f); // (11 - 1) / 2 = 5
        assertEquals(5.0f, res.getFloat(1), 1e-6f); // (21 - 1) / 4 = 5
    }
}
