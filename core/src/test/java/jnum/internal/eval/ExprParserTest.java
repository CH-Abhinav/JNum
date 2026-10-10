package jnum.internal.eval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class ExprParserTest {

    @Test
    @DisplayName("Positional variable parsing and simple binary operators")
    void testPositionalParsing() {
        ASTNode n1 = ExprParser.parsePositional("$0 + $1");
        assertInstanceOf(ASTNode.Add.class, n1);
        ASTNode.Add add = (ASTNode.Add) n1;
        assertEquals(new ASTNode.Var(0), add.left());
        assertEquals(new ASTNode.Var(1), add.right());

        ASTNode n2 = ExprParser.parsePositional("$0 - $1");
        assertInstanceOf(ASTNode.Sub.class, n2);

        ASTNode n3 = ExprParser.parsePositional("$0 / $1");
        assertInstanceOf(ASTNode.Div.class, n3);
    }

    @Test
    @DisplayName("Operator precedence: multiplication before addition")
    void testPrecedence() {
        // $0 + $1 * $2 -> Add($0, Mul($1, $2)) or Fma($1, $2, $0)
        ASTNode node = ExprParser.parsePositional("$0 + $1 * $2");
        assertInstanceOf(ASTNode.Fma.class, node);
        ASTNode.Fma fma = (ASTNode.Fma) node;
        assertEquals(new ASTNode.Var(1), fma.a());
        assertEquals(new ASTNode.Var(2), fma.b());
        assertEquals(new ASTNode.Var(0), fma.c());
    }

    @Test
    @DisplayName("Parentheses and bracket aliases: (), [], {}")
    void testBracketAliases() {
        ASTNode node = ExprParser.parsePositional("[$0 + 2] * {$1 - 3}");
        assertInstanceOf(ASTNode.Mul.class, node);
        ASTNode.Mul mul = (ASTNode.Mul) node;
        assertInstanceOf(ASTNode.Add.class, mul.left());
        assertInstanceOf(ASTNode.Sub.class, mul.right());
    }

    @Test
    @DisplayName("Named variable parsing")
    void testNamedParsing() {
        String[] varNames = new String[]{"alpha", "beta"};
        ASTNode node = ExprParser.parseNamed("alpha * beta + 5", varNames);
        assertInstanceOf(ASTNode.Fma.class, node);
        ASTNode.Fma fma = (ASTNode.Fma) node;
        assertEquals(new ASTNode.Var(0), fma.a());
        assertEquals(new ASTNode.Var(1), fma.b());
        assertEquals(new ASTNode.Const(5.0), fma.c());
    }

    @Test
    @DisplayName("Syntax errors throw IllegalArgumentException")
    void testSyntaxErrors() {
        // Missing index after $
        assertThrows(IllegalArgumentException.class, () -> ExprParser.parsePositional("$ + 2"));

        // Named variable inside positional parser
        assertThrows(IllegalArgumentException.class, () -> ExprParser.parsePositional("x + 1"));

        // Unrecognized named variable
        assertThrows(IllegalArgumentException.class, () -> ExprParser.parseNamed("x + y", new String[]{"x"}));

        // Unmatched parenthesis
        assertThrows(IllegalArgumentException.class, () -> ExprParser.parsePositional("($0 + 1"));
    }
}
