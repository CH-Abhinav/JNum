package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class ASTNodeTest {

    @Test
    @DisplayName("Record nodes instantiate and store fields correctly")
    void testNodeRecords() {
        ASTNode.Var v0 = new ASTNode.Var(0);
        assertEquals(0, v0.index());

        ASTNode.Const c = new ASTNode.Const(3.14159);
        assertEquals(3.14159, c.value(), 1e-12);

        ASTNode.Add add = new ASTNode.Add(v0, c);
        assertEquals(v0, add.left());
        assertEquals(c, add.right());

        ASTNode.Sub sub = new ASTNode.Sub(v0, c);
        assertEquals(v0, sub.left());
        assertEquals(c, sub.right());

        ASTNode.Mul mul = new ASTNode.Mul(v0, c);
        assertEquals(v0, mul.left());
        assertEquals(c, mul.right());

        ASTNode.Div div = new ASTNode.Div(v0, c);
        assertEquals(v0, div.left());
        assertEquals(c, div.right());

        ASTNode.Fma fma = new ASTNode.Fma(v0, c, add);
        assertEquals(v0, fma.a());
        assertEquals(c, fma.b());
        assertEquals(add, fma.c());
    }

    @Test
    @DisplayName("Record equality and hashCode contract")
    void testEqualityAndHashCode() {
        ASTNode.Var v1 = new ASTNode.Var(2);
        ASTNode.Var v2 = new ASTNode.Var(2);
        ASTNode.Var v3 = new ASTNode.Var(3);

        assertEquals(v1, v2);
        assertEquals(v1.hashCode(), v2.hashCode());
        assertNotEquals(v1, v3);

        ASTNode.Add a1 = new ASTNode.Add(v1, new ASTNode.Const(10.0));
        ASTNode.Add a2 = new ASTNode.Add(v2, new ASTNode.Const(10.0));
        assertEquals(a1, a2);
    }

    @Test
    @DisplayName("Sealed interface pattern matching exhaustiveness")
    void testPatternMatching() {
        ASTNode node = new ASTNode.Fma(new ASTNode.Var(0), new ASTNode.Var(1), new ASTNode.Const(5.0));
        String typeName = switch (node) {
            case ASTNode.Var v -> "var";
            case ASTNode.Const c -> "const";
            case ASTNode.Add a -> "add";
            case ASTNode.Sub s -> "sub";
            case ASTNode.Mul m -> "mul";
            case ASTNode.Div d -> "div";
            case ASTNode.Fma f -> "fma";
        };
        assertEquals("fma", typeName);
    }
}
