package jnum.internal.eval;

import java.util.*;

public final class ExprParser {

    private enum TokenType { NUMBER, POSITIONAL_VAR, NAMED_VAR, OPERATOR, LPAREN, RPAREN }
    private record Token(TokenType type, String text, double numVal, int varIndex) {}

    public static ASTNode parsePositional(String rawExpr) {
        return buildAndOptimize(rawExpr, null);
    }

    public static ASTNode parseNamed(String rawExpr, String[] varNames) {
        return buildAndOptimize(rawExpr, varNames);
    }

    private static ASTNode buildAndOptimize(String rawExpr, String[] varNames) {
        String clean = rawExpr.replace('{', '(').replace('}', ')')
                .replace('[', '(').replace(']', ')')
                .replaceAll("\\s+", "");

        List<Token> tokens = tokenize(clean, varNames);
        List<Token> rpn = toRPN(tokens);
        ASTNode root = buildAST(rpn);
        return optimizeFMA(root);
    }

    private static List<Token> tokenize(String expr, String[] varNames) {
        List<Token> tokens = new ArrayList<>();
        int i = 0, len = expr.length();

        while (i < len) {
            char c = expr.charAt(i);

            if (c == '(') {
                tokens.add(new Token(TokenType.LPAREN, "(", 0, -1)); i++;
            } else if (c == ')') {
                tokens.add(new Token(TokenType.RPAREN, ")", 0, -1)); i++;
            } else if (c == '+' || c == '-' || c == '*' || c == '/') {
                tokens.add(new Token(TokenType.OPERATOR, String.valueOf(c), 0, -1)); i++;
            } else if (Character.isDigit(c) || c == '.') {
                int start = i;
                while (i < len && (Character.isDigit(expr.charAt(i)) || expr.charAt(i) == '.')) i++;
                // Parse as double to preserve maximum precision for ASTNode
                tokens.add(new Token(TokenType.NUMBER, "", Double.parseDouble(expr.substring(start, i)), -1));
            } else if (c == '$') {
                i++;
                int start = i;
                while (i < len && Character.isDigit(expr.charAt(i))) i++;
                if (start == i) throw new IllegalArgumentException("Expected number after '$' for positional variable.");
                int index = Integer.parseInt(expr.substring(start, i));
                tokens.add(new Token(TokenType.POSITIONAL_VAR, "", 0, index));
            } else if (Character.isLetter(c) || c == '_') {
                int start = i;
                while (i < len && (Character.isLetterOrDigit(expr.charAt(i)) || expr.charAt(i) == '_')) i++;
                String name = expr.substring(start, i);

                if (varNames == null) throw new IllegalArgumentException("Named variable '" + name + "' found, but varargs API requires positional variables ($0, $1).");

                int index = Arrays.asList(varNames).indexOf(name);
                if (index == -1) throw new IllegalArgumentException("Variable '" + name + "' not found in provided Map.");
                tokens.add(new Token(TokenType.NAMED_VAR, name, 0, index));
            } else {
                throw new IllegalArgumentException("Unexpected character in expression: " + c);
            }
        }
        return tokens;
    }

    private static List<Token> toRPN(List<Token> tokens) {
        List<Token> output = new ArrayList<>();
        Deque<Token> stack = new ArrayDeque<>();

        for (Token t : tokens) {
            switch (t.type) {
                case NUMBER, POSITIONAL_VAR, NAMED_VAR -> output.add(t);
                case OPERATOR -> {
                    while (!stack.isEmpty() && stack.peek().type == TokenType.OPERATOR &&
                            precedence(stack.peek().text) >= precedence(t.text)) {
                        output.add(stack.pop());
                    }
                    stack.push(t);
                }
                case LPAREN -> stack.push(t);
                case RPAREN -> {
                    while (!stack.isEmpty() && stack.peek().type != TokenType.LPAREN) {
                        output.add(stack.pop());
                    }
                    if (stack.isEmpty()) throw new IllegalArgumentException("Mismatched brackets.");
                    stack.pop(); // Discard the '('
                }
            }
        }
        while (!stack.isEmpty()) {
            Token t = stack.pop();
            if (t.type == TokenType.LPAREN) throw new IllegalArgumentException("Mismatched brackets.");
            output.add(t);
        }
        return output;
    }

    private static int precedence(String op) {
        return (op.equals("*") || op.equals("/")) ? 2 : 1;
    }

    private static ASTNode buildAST(List<Token> rpn) {
        Deque<ASTNode> nodeStack = new ArrayDeque<>();
        for (Token t : rpn) {
            if (t.type == TokenType.NUMBER) {
                nodeStack.push(new ASTNode.Const(t.numVal));
            } else if (t.type == TokenType.POSITIONAL_VAR || t.type == TokenType.NAMED_VAR) {
                nodeStack.push(new ASTNode.Var(t.varIndex));
            } else if (t.type == TokenType.OPERATOR) {
                ASTNode right = nodeStack.pop();
                ASTNode left = nodeStack.pop();
                nodeStack.push(switch (t.text) {
                    case "+" -> new ASTNode.Add(left, right);
                    case "-" -> new ASTNode.Sub(left, right);
                    case "*" -> new ASTNode.Mul(left, right);
                    case "/" -> new ASTNode.Div(left, right);
                    default -> throw new UnsupportedOperationException("Operator not supported: " + t.text);
                });
            }
        }
        return nodeStack.pop();
    }

    private static ASTNode optimizeFMA(ASTNode node) {
        if (node instanceof ASTNode.Add(ASTNode left, ASTNode right)) {
            left = optimizeFMA(left);
            right = optimizeFMA(right);

            // Matches: (a * b) + c -> Hardware FMA
            if (left instanceof ASTNode.Mul(ASTNode a, ASTNode b)) return new ASTNode.Fma(a, b, right);

            // Matches: a + (b * c) -> Hardware FMA
            if (right instanceof ASTNode.Mul(ASTNode a, ASTNode b)) return new ASTNode.Fma(a, b, left);

            return new ASTNode.Add(left, right);
        } else if (node instanceof ASTNode.Sub(ASTNode left, ASTNode right)) {
            return new ASTNode.Sub(optimizeFMA(left), optimizeFMA(right));
        } else if (node instanceof ASTNode.Mul(ASTNode left, ASTNode right)) {
            return new ASTNode.Mul(optimizeFMA(left), optimizeFMA(right));
        } else if (node instanceof ASTNode.Div(ASTNode left, ASTNode right)) {
            return new ASTNode.Div(optimizeFMA(left), optimizeFMA(right));
        }
        return node;
    }
}