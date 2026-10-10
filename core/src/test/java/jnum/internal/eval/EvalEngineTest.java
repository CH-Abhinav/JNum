package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class EvalEngineTest {

    @Test
    @DisplayName("Positional evaluation: Float, Double, Int")
    void testPositionalEvaluation() {
        NDArray aF = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray bF = JNum.from(new float[]{10f, 20f, 30f}, 3);
        NDArray resF = EvalEngine.evaluatePositional("$0 * 2 + $1", new NDArray[]{aF, bF});
        assertEquals(DType.f32, resF.getDType());
        assertEquals(12f, resF.getFloat(0), 1e-6f);
        assertEquals(24f, resF.getFloat(1), 1e-6f);
        assertEquals(36f, resF.getFloat(2), 1e-6f);

        // Double
        NDArray aD = JNum.from(new double[]{2.0, 3.0}, 2);
        NDArray bD = JNum.from(new double[]{4.0, 5.0}, 2);
        NDArray resD = EvalEngine.evaluatePositional("$0 * $1 - 1.5", new NDArray[]{aD, bD});
        assertEquals(DType.f64, resD.getDType());
        assertEquals(6.5, resD.getDouble(0), 1e-12);
        assertEquals(13.5, resD.getDouble(1), 1e-12);

        // Int
        NDArray aI = JNum.from(new int[]{5, 10}, 2);
        NDArray bI = JNum.from(new int[]{2, 3}, 2);
        NDArray resI = EvalEngine.evaluatePositional("$0 / $1", new NDArray[]{aI, bI});
        assertEquals(DType.i32, resI.getDType());
        assertEquals(2, resI.getInt(0));
        assertEquals(3, resI.getInt(1));
    }

    @Test
    @DisplayName("Named evaluation with variable map")
    void testNamedEvaluation() {
        Map<String, NDArray> vars = new LinkedHashMap<>();
        vars.put("x", JNum.from(new float[]{2f, 4f}, 2));
        vars.put("y", JNum.from(new float[]{3f, 5f}, 2));

        NDArray res = EvalEngine.evaluateNamed("x * y + 10", vars);
        assertEquals(DType.f32, res.getDType());
        assertEquals(16f, res.getFloat(0), 1e-6f);
        assertEquals(30f, res.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Mixed type promotion: i32 and f32 promote to f32")
    void testMixedTypePromotion() {
        NDArray aI = JNum.from(new int[]{1, 2}, 2);
        NDArray bF = JNum.from(new float[]{0.5f, 1.5f}, 2);

        NDArray res = EvalEngine.evaluatePositional("$0 + $1", new NDArray[]{aI, bF});
        assertEquals(DType.f32, res.getDType());
        assertEquals(1.5f, res.getFloat(0), 1e-6f);
        assertEquals(3.5f, res.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Empty variables argument throws IllegalArgumentException")
    void testEmptyVariables() {
        assertThrows(IllegalArgumentException.class, () -> JNum.eval("1 + 2", new NDArray[0]));
        assertThrows(IllegalArgumentException.class, () -> JNum.eval("1 + 2", Map.of()));
    }
}
