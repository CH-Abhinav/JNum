# Chapter 9: Symbolic Expression Engine

When evaluating compound mathematical formulas involving multiple tensors (such as $A \cdot B + C - D$), a typical object-oriented library executes operations step-by-step:

```java
// Allocates 3 intermediate arrays in memory:
NDArray step1 = a.mul(b); // Temp Array 1
NDArray step2 = step1.add(c); // Temp Array 2
NDArray step3 = step2.sub(d); // Final Result (Temp 1 & 2 are discarded)
```

In long-running or high-throughput systems, discarding millions of temporary arrays causes memory fragmentation and severe garbage collection pressure.

JNum solves this through its **Symbolic Evaluation Engine** ([`JNum.eval`](../../core/src/main/java/jnum/JNum.java#L480)). It compiles mathematical expression strings into an Abstract Syntax Tree (AST), optimizes execution order, and recycles intermediate memory buffers with zero temporary allocations.

---

## 1. Positional Expressions: `$0, $1, $2`

The simplest way to use the symbolic engine is with positional placeholders `$0, $1, \dots, \$N`, corresponding to the arguments passed to `JNum.eval()`:

```java
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray a = JNum.from(new float[]{1, 2, 3, 4}, 2, 2);
    NDArray b = JNum.from(new float[]{10, 20, 30, 40}, 2, 2);
    NDArray c = JNum.from(new float[]{5, 5, 5, 5}, 2, 2);

    // Evaluates a * b + c with optimized buffer allocation
    NDArray result = JNum.eval("$0 * $1 + $2", a, b, c);

    IO.println(result);
}
```

Output:
```text
[[15.0, 45.0],
 [65.0, 85.0]]
```

---

## 2. Named Variable Expressions

For complex or self-documenting formulas, use named variable identifiers passed via a `Map<String, NDArray>`:

```java
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;

void main() {
    NDArray x       = JNum.rand(1000, 64);
    NDArray weights = JNum.rand(1000, 64);
    NDArray bias    = JNum.ones(1000, 64);

    // Express neural affine projection symbolically
    NDArray output = JNum.eval(
        "x * weights + bias",
        Map.of("x", x, "weights", weights, "bias", bias)
    );

    IO.println("Output shape: " + output.shapeString());
}
```

> [!TIP]
> To bound native memory deterministically, pass an explicit `Arena`:
> `JNum.eval(arena, "$0 * $1 + $2", a, b, c)` or `JNum.eval(arena, "x + y", map)`.

---

## 3. Grammar, Precedence, and Parentheses

The symbolic engine adheres to standard algebraic precedence rules:

1. **Parentheses**: `(...)` force inner sub-expression evaluation.
2. **Unary Negation**: `-x` negates the operand.
3. **Multiplicative**: `*` and `/` have higher precedence than additive operations.
4. **Additive**: `+` and `-` evaluate left-to-right.

```java
// Explicit grouping with parentheses
NDArray res = JNum.eval("($0 + $1) * ($2 - $3) / 2.0", a, b, c, d);
```

---

## 4. Under the Hood: AST Compilation & Buffer Recycling

How does the symbolic engine achieve zero-allocation evaluation?

1. **Zero-Regex Tokenization**: A custom character-by-character scanner ([`ExprParser`](../../core/src/main/java/jnum/internal/eval/ExprParser.java)) tokenizes the expression string without allocating intermediate regular-expression pattern objects.
2. **Shunting-Yard AST Generation**: Dijkstra's Shunting-Yard algorithm converts infix tokens into an Abstract Syntax Tree ([`ASTNode`](../../core/src/main/java/jnum/internal/eval/ASTNode.java)).
3. **Topological Buffer Recycling**: During tree traversal, temporary buffers allocated for left-hand sub-expressions are recycled and reused for subsequent right-hand sub-expressions, bounding total memory consumption to the tree depth rather than the number of operators.
4. **Type-Specialized Kernels**: Execution dispatches to specialized evaluators ([`FloatEval`](../../core/src/main/java/jnum/internal/eval/FloatEval.java), [`DoubleEval`](../../core/src/main/java/jnum/internal/eval/DoubleEval.java), or [`IntEval`](../../core/src/main/java/jnum/internal/eval/IntEval.java)) to eliminate primitive boxing.

---

## 5. Practical Recipe: Black-Scholes Intermediate Value Calculation

The following recipe evaluates the core numerator formula for Black-Scholes option pricing:

$$\text{Numerator} = \ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) T$$

```java
import java.util.Map;
import jnum.JNum;
import jnum.NDArray;

void main() {
    int optionsCount = 100_000;

    NDArray S = JNum.rand(90.0f, 110.0f, jnum.DType.f32, optionsCount); // Spot price
    NDArray K = JNum.rand(95.0f, 105.0f, jnum.DType.f32, optionsCount); // Strike price
    NDArray r = JNum.ones(optionsCount).mul(0.05f);                      // Risk-free rate
    NDArray T = JNum.ones(optionsCount).mul(1.0f);                       // 1 year to expiry

    // Precompute ln(S / K)
    NDArray logSK = S.div(K).log();

    // Evaluate symbolic term: logSK + (r * T)
    NDArray term = JNum.eval(
        "logSK + (r * T)",
        Map.of("logSK", logSK, "r", r, "T", T)
    );

    IO.println("Computed " + term.getSize() + " option terms. Sample: " + term.getFloat(0));
}
```

Now proceed to **[Chapter 10: Tabular & NumPy I/O](10-tabular-and-numpy-io.md)** to load and save data from CSV and NumPy `.npy` / `.npz` files.
