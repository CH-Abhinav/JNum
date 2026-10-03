package jnum.internal.eval;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jnum.NDArray;
import jnum.JNum;
import jnum.DType;

public class DoubleEval {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private final DoubleNode root;

    public DoubleEval(ASTNode genericTree) {
        this.root = compile(genericTree);
    }

    public NDArray execute(NDArray[] vars) {
        MemorySegment[] env = new MemorySegment[vars.length];
        long totalElements = vars[0].getSize();
        for (int i = 0; i < vars.length; i++) env[i] = vars[i].contiguous().getData();

        NDArray result = JNum.zeros(DType.f64, vars[0].getShape());
        MemorySegment dst = result.getData();

        long i = 0, loopBound = SPECIES.loopBound(totalElements);
        for (; i < loopBound; i += SPECIES.length()) {
            long offset = i * Double.BYTES;
            root.evalVec(SPECIES, offset, env).intoMemorySegment(dst, offset, ORDER);
        }
        for (; i < totalElements; i++) {
            long offset = i * Double.BYTES;
            dst.set(ValueLayout.JAVA_DOUBLE, offset, root.evalScalar(offset, env));
        }
        return result;
    }

    private DoubleNode compile(ASTNode node) {
        return switch (node) {
            case ASTNode.Var v -> new VarNode(v.index());
            case ASTNode.Const c -> new ConstNode(c.value()); // Const already holds double
            case ASTNode.Add a -> new AddNode(compile(a.left()), compile(a.right()));
            case ASTNode.Sub s -> new SubNode(compile(s.left()), compile(s.right()));
            case ASTNode.Mul m -> new MulNode(compile(m.left()), compile(m.right()));
            case ASTNode.Div d -> new DivNode(compile(d.left()), compile(d.right()));
            case ASTNode.Fma f -> new FmaNode(compile(f.a()), compile(f.b()), compile(f.c()));
        };
    }

    interface DoubleNode {
        DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env);
        double evalScalar(long off, MemorySegment[] env);
    }

    record VarNode(int idx) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return DoubleVector.fromMemorySegment(s, env[idx], off, ORDER); }
        public double evalScalar(long off, MemorySegment[] env) { return env[idx].get(ValueLayout.JAVA_DOUBLE, off); }
    }
    record ConstNode(double val) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return DoubleVector.broadcast(s, val); }
        public double evalScalar(long off, MemorySegment[] env) { return val; }
    }
    record AddNode(DoubleNode l, DoubleNode r) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).add(r.evalVec(s, off, env)); }
        public double evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) + r.evalScalar(off, env); }
    }
    record SubNode(DoubleNode l, DoubleNode r) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).sub(r.evalVec(s, off, env)); }
        public double evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) - r.evalScalar(off, env); }
    }
    record MulNode(DoubleNode l, DoubleNode r) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).mul(r.evalVec(s, off, env)); }
        public double evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) * r.evalScalar(off, env); }
    }
    record DivNode(DoubleNode l, DoubleNode r) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).div(r.evalVec(s, off, env)); }
        public double evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) / r.evalScalar(off, env); }
    }
    record FmaNode(DoubleNode a, DoubleNode b, DoubleNode c) implements DoubleNode {
        public DoubleVector evalVec(VectorSpecies<Double> s, long off, MemorySegment[] env) { return a.evalVec(s, off, env).fma(b.evalVec(s, off, env), c.evalVec(s, off, env)); }
        public double evalScalar(long off, MemorySegment[] env) { return Math.fma(a.evalScalar(off, env), b.evalScalar(off, env), c.evalScalar(off, env)); }
    }
}