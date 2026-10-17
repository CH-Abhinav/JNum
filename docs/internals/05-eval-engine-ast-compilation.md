# Systems Internals: AST Compilation & The Symbolic Engine

When writing high-level tensor mathematics, expressions such as $A \cdot B + C$ or $X \cdot W + B$ are naturally expressed as algebraic formulas. In naive implementations, either:
1. Every operator allocates an intermediate tensor buffer, or
2. The user is forced to write tedious, error-prone manual loops.

This document details the systems engineering of JNum's **Symbolic Evaluation Engine** ([`EvalEngine.java`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/internal/eval/EvalEngine.java)): how mathematical strings are tokenized without regex allocations, how Shunting-Yard parses expressions into an Abstract Syntax Tree (AST), how algebraic trees are rewritten into hardware FMA (Fused Multiply-Add) instructions, and how buffers are recycled during topological traversal.

---

## 1. Compilation Pipeline Architecture

The symbolic engine functions as a domain-specific JIT compiler for tensor expressions:

```
Expression String (e.g. "$0 * $1 + $2" or "x * w + b")
       │
       ▼ Lexical Analysis (ExprParser.tokenize)
Tokens Stream: [VAR($0), OP(*), VAR($1), OP(+), VAR($2)]
       │
       ▼ Shunting-Yard Algorithm (ExprParser.toRPN)
Reverse Polish Notation (RPN): [$0, $1, *, $2, +]
       │
       ▼ AST Construction (ExprParser.buildAST)
Raw AST: Add(Mul(Var(0), Var(1)), Var(2))
       │
       ▼ Algebraic Optimization Pass (ExprParser.optimizeFMA)
Optimized AST: Fma(Var(0), Var(1), Var(2))  <-- Direct Hardware FMA!
       │
       ▼ Type-Specialized Evaluator Dispatch (FloatEval / DoubleEval / IntEval)
Topological Evaluation with Buffer Recycling
```

---

## 2. Zero-Regex Lexical Tokenizer

Many string parsers rely on `java.util.regex.Pattern` or `String.split()`. In high-throughput scoring services, compiling regex patterns or creating `Matcher` objects inside the hot path causes substantial allocation overhead.

JNum's [`ExprParser`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/internal/eval/ExprParser.java) uses a **deterministic character-level state scanner**:
* Scans characters with a single pass through `expr.charAt(i)`.
* Recognizes bracket pairs `()`, `{}`, `[]` and normalizes them to parentheses.
* Extracts numeric constants directly into IEEE 754 `double` values.
* Resolves variable identifiers:
  * Positional syntax `$0`, `$1` decodes the integer index.
  * Named variables (`x`, `weights`) are matched against a pre-sorted string dictionary with index lookups.

---

## 3. Reverse Polish Notation (RPN) & AST Generation

To respect algebraic precedence without recursion limits, JNum uses Dijkstra's **Shunting-Yard algorithm**:

* Multiplicative operators (`*`, `/`) carry precedence level 2.
* Additive operators (`+`, `-`) carry precedence level 1.
* Operators are pushed onto an operator stack (`Deque<Token>`), draining higher-precedence operators into the output list.
* Mismatched parentheses are detected and rejected with informative `IllegalArgumentException` messages.

From the RPN token list, an Abstract Syntax Tree ([`ASTNode`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/internal/eval/ASTNode.java)) is constructed using a stack of tree nodes.

---

## 4. Hardware FMA Pattern Matching Pass (`optimizeFMA`)

Before executing the AST, JNum passes the tree through an **algebraic optimization pass**:

```java
private static ASTNode optimizeFMA(ASTNode node) {
    if (node instanceof ASTNode.Add(ASTNode left, ASTNode right)) {
        left = optimizeFMA(left);
        right = optimizeFMA(right);

        // Pattern 1: (a * b) + c -> Hardware FMA
        if (left instanceof ASTNode.Mul(ASTNode a, ASTNode b)) {
            return new ASTNode.Fma(a, b, right);
        }

        // Pattern 2: a + (b * c) -> Hardware FMA
        if (right instanceof ASTNode.Mul(ASTNode a, ASTNode b)) {
            return new ASTNode.Fma(a, b, left);
        }

        return new ASTNode.Add(left, right);
    }
    // ... recursively optimizes other operators
    return node;
}
```

### Why FMA Optimization Matters

When an expression matches $(A \times B) + C$:
1. **Instruction Fusion**: On AVX2 and AVX-512, separate multiplication (`vmulps`) and addition (`vaddps`) instructions are fused into a single hardware FMA instruction (`vfmadd231ps`).
2. **Double Precision Throughput**: FMA issues both a multiply and an add in a single CPU cycle, doubling peak theoretical arithmetic throughput.
3. **Elimination of Intermediate Rounding**: Intermediate products are preserved with infinite precision before addition, preventing numerical drift.
4. **Intermediate Buffer Elimination**: Instead of allocating a buffer for $A \times B$ and then adding $C$, the FMA kernel streams $C$ directly into the accumulator.

---

## 5. Topological Execution & Buffer Recycling

During evaluation in [`FloatEval`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/internal/eval/FloatEval.java) and [`DoubleEval`](file:///c:/Users/DELL/Desktop/JNum/core/src/main/java/jnum/internal/eval/DoubleEval.java), the AST is evaluated in post-order (topological dependency order).

### Buffer Recycling Strategy

To prevent memory leaks and GC churn during complex tree evaluations:
* When an operator (like `Add` or `Mul`) finishes evaluating its left and right sub-trees, if either child was an intermediate temporary tensor (and not an immutable user input), that child's underlying native memory segment is **recycled and repurposed as the output buffer** for the parent operation.
* The maximum native memory required to evaluate any arbitrary expression tree is bounded by $\mathcal{O}(\text{Tree Depth})$ rather than $\mathcal{O}(\text{Total Operators})$.
