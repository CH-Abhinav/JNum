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
        return evaluatePositional(java.lang.foreign.Arena.ofAuto(), expr, vars);
    }

    public static NDArray evaluatePositional(java.lang.foreign.Arena arena, String expr, NDArray[] vars) {
        // Cache key is just the pure expression for positional varargs
        return routeAndExecute(arena, expr, expr, null, vars);
    }

    public static NDArray evaluateNamed(String expr, Map<String, NDArray> variables) {
        return evaluateNamed(java.lang.foreign.Arena.ofAuto(), expr, variables);
    }

    public static NDArray evaluateNamed(java.lang.foreign.Arena arena, String expr, Map<String, NDArray> variables) {
        String[] varNames = variables.keySet().toArray(new String[0]);
        Arrays.sort(varNames);
        NDArray[] vars = new NDArray[varNames.length];
        for (int i = 0; i < varNames.length; i++) vars[i] = variables.get(varNames[i]);

        // Cache key includes variable names, but we keep the pure expr separate for the parser
        String cacheKey = expr + Arrays.toString(varNames);
        return routeAndExecute(arena, expr, cacheKey, varNames, vars);
    }

    private static NDArray routeAndExecute(java.lang.foreign.Arena arena, String pureExpr, String cacheKey, String[] varNames, NDArray[] vars) {
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
            }).execute(arena, promotedVars);

            case f64 -> DOUBLE_CACHE.computeIfAbsent(cacheKey, k -> {
                ASTNode genericTree = (varNames == null)
                        ? ExprParser.parsePositional(pureExpr)
                        : ExprParser.parseNamed(pureExpr, varNames);
                return new DoubleEval(genericTree);
            }).execute(arena, promotedVars);

            case i32 -> INT_CACHE.computeIfAbsent(cacheKey, k -> {
                ASTNode genericTree = (varNames == null)
                        ? ExprParser.parsePositional(pureExpr)
                        : ExprParser.parseNamed(pureExpr, varNames);
                return new IntEval(genericTree);
            }).execute(arena, promotedVars);

            case bool -> throw new IllegalArgumentException("Math eval() does not support booleans.");
        };
    }
}