package jnum.internal.eval;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jnum.NDArray;
import jnum.JNum;
import jnum.DType;

public class IntEval {
    private static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private final IntNode root;

    public IntEval(ASTNode genericTree) {
        this.root = compile(genericTree);
    }

    public NDArray execute(NDArray[] vars) {
        MemorySegment[] env = new MemorySegment[vars.length];
        long totalElements = vars[0].getSize();
        for (int i = 0; i < vars.length; i++) env[i] = vars[i].contiguous().getData();

        NDArray result = JNum.zeros(DType.i32, vars[0].getShape());
        MemorySegment dst = result.getData();

        long i = 0, loopBound = SPECIES.loopBound(totalElements);
        for (; i < loopBound; i += SPECIES.length()) {
            long offset = i * Integer.BYTES;
            root.evalVec(SPECIES, offset, env).intoMemorySegment(dst, offset, ORDER);
        }
        for (; i < totalElements; i++) {
            long offset = i * Integer.BYTES;
            dst.set(ValueLayout.JAVA_INT, offset, root.evalScalar(offset, env));
        }
        return result;
    }

    private IntNode compile(ASTNode node) {
        return switch (node) {
            case ASTNode.Var v -> new VarNode(v.index());
            case ASTNode.Const c -> new ConstNode((int) c.value());
            case ASTNode.Add a -> new AddNode(compile(a.left()), compile(a.right()));
            case ASTNode.Sub s -> new SubNode(compile(s.left()), compile(s.right()));
            case ASTNode.Mul m -> new MulNode(compile(m.left()), compile(m.right()));
            case ASTNode.Div d -> new DivNode(compile(d.left()), compile(d.right()));
            case ASTNode.Fma f -> new FmaNode(compile(f.a()), compile(f.b()), compile(f.c()));
        };
    }

    interface IntNode {
        IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env);
        int evalScalar(long off, MemorySegment[] env);
    }

    record VarNode(int idx) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return IntVector.fromMemorySegment(s, env[idx], off, ORDER); }
        public int evalScalar(long off, MemorySegment[] env) { return env[idx].get(ValueLayout.JAVA_INT, off); }
    }
    record ConstNode(int val) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return IntVector.broadcast(s, val); }
        public int evalScalar(long off, MemorySegment[] env) { return val; }
    }
    record AddNode(IntNode l, IntNode r) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).add(r.evalVec(s, off, env)); }
        public int evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) + r.evalScalar(off, env); }
    }
    record SubNode(IntNode l, IntNode r) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).sub(r.evalVec(s, off, env)); }
        public int evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) - r.evalScalar(off, env); }
    }
    record MulNode(IntNode l, IntNode r) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).mul(r.evalVec(s, off, env)); }
        public int evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) * r.evalScalar(off, env); }
    }
    record DivNode(IntNode l, IntNode r) implements IntNode {
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).div(r.evalVec(s, off, env)); }
        public int evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) / r.evalScalar(off, env); }
    }
    record FmaNode(IntNode a, IntNode b, IntNode c) implements IntNode {
        // IntVector has no fma(), so we do a.mul(b).add(c)
        public IntVector evalVec(VectorSpecies<Integer> s, long off, MemorySegment[] env) { return a.evalVec(s, off, env).mul(b.evalVec(s, off, env)).add(c.evalVec(s, off, env)); }
        public int evalScalar(long off, MemorySegment[] env) { return (a.evalScalar(off, env) * b.evalScalar(off, env)) + c.evalScalar(off, env); }
    }
}