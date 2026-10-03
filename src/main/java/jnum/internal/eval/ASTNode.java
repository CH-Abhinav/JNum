package jnum.internal.eval;

public sealed interface ASTNode {
    record Var(int index) implements ASTNode {}
    record Const(double value) implements ASTNode {}
    record Add(ASTNode left, ASTNode right) implements ASTNode {}
    record Sub(ASTNode left, ASTNode right) implements ASTNode {}
    record Mul(ASTNode left, ASTNode right) implements ASTNode {}
    record Div(ASTNode left, ASTNode right) implements ASTNode {}
    record Fma(ASTNode a, ASTNode b, ASTNode c) implements ASTNode {}
}