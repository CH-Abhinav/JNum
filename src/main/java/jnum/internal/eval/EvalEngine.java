package jnum.internal.eval;

import java.util.Map;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import jnum.NDArray;
import jnum.DType;
import jnum.internal.layout.TypeUtil;

public class EvalEngine {
    private static final ConcurrentHashMap<String, FloatEval> FLOAT_CACHE = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, DoubleEval> DOUBLE_CACHE = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, IntEval> INT_CACHE = new ConcurrentHashMap<>();

    public static NDArray evaluatePositional(String expr, NDArray[] vars) {
        // Cache key is just the pure expression for positional varargs
        return routeAndExecute(expr, expr, null, vars);
    }

    public static NDArray evaluateNamed(String expr, Map<String, NDArray> variables) {
        String[] varNames = variables.keySet().toArray(new String[0]);
        NDArray[] vars = new NDArray[varNames.length];
        for (int i = 0; i < varNames.length; i++) vars[i] = variables.get(varNames[i]);

        // Cache key includes variable names, but we keep the pure expr separate for the parser
        String cacheKey = expr + Arrays.toString(varNames);
        return routeAndExecute(expr, cacheKey, varNames, vars);
    }

    private static NDArray routeAndExecute(String pureExpr, String cacheKey, String[] varNames, NDArray[] vars) {
        // 1. Determine the highest precision type (Promotion)
        DType targetType = vars[0].getDType();
        for (NDArray v : vars) targetType = TypeUtil.promoteTypes(targetType, v.getDType());

        // 2. Cast all inputs to the unified type (Zero-copy if already matching)
        NDArray[] promotedVars = new NDArray[vars.length];
        for (int i = 0; i < vars.length; i++) promotedVars[i] = vars[i].cast(targetType);

        // 3. Parse AST once, then route to the hardware-specific cache
        return switch (targetType) {
            case f32 -> FLOAT_CACHE.computeIfAbsent(cacheKey, k -> {
                ASTNode genericTree = (varNames == null)
                        ? ExprParser.parsePositional(pureExpr)
                        : ExprParser.parseNamed(pureExpr, varNames);
                return new FloatEval(genericTree);
            }).execute(promotedVars);

            case bool -> throw new IllegalArgumentException("Math eval() does not support booleans.");
            default -> throw new UnsupportedOperationException("Eval engine currently supports f32 only. Generate DoubleEval and IntEval to support " + targetType);
        };
    }
}