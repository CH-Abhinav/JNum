package jnum.internal.eval;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jnum.NDArray;
import jnum.JNum;
import jnum.DType;

public class FloatEval {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private final FloatNode root;

    // FIX: Replaced String parsing with direct ASTNode injection
    public FloatEval(ASTNode genericTree) {
        this.root = compile(genericTree);
    }

    public NDArray execute(NDArray[] vars) {
        MemorySegment[] env = new MemorySegment[vars.length];
        long totalElements = vars[0].getSize();
        for (int i = 0; i < vars.length; i++) env[i] = vars[i].contiguous().getData();

        NDArray result = JNum.zeros(DType.f32, vars[0].getShape());
        MemorySegment dst = result.getData();

        long i = 0, loopBound = SPECIES.loopBound(totalElements);
        for (; i < loopBound; i += SPECIES.length()) {
            long offset = i * Float.BYTES;
            root.evalVec(SPECIES, offset, env).intoMemorySegment(dst, offset, ORDER);
        }
        for (; i < totalElements; i++) {
            long offset = i * Float.BYTES;
            dst.set(ValueLayout.JAVA_FLOAT, offset, root.evalScalar(offset, env));
        }
        return result;
    }

    // --- HARDWARE COMPILATION & NODES ---

    private FloatNode compile(ASTNode node) {
        return switch (node) {
            case ASTNode.Var v -> new VarNode(v.index());
            case ASTNode.Const c -> new ConstNode((float) c.value());
            case ASTNode.Add a -> new AddNode(compile(a.left()), compile(a.right()));
            case ASTNode.Sub s -> new SubNode(compile(s.left()), compile(s.right()));
            case ASTNode.Mul m -> new MulNode(compile(m.left()), compile(m.right()));
            case ASTNode.Div d -> new DivNode(compile(d.left()), compile(d.right()));
            case ASTNode.Fma f -> new FmaNode(compile(f.a()), compile(f.b()), compile(f.c()));
        };
    }

    interface FloatNode {
        FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env);
        float evalScalar(long off, MemorySegment[] env);
    }

    record VarNode(int idx) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return FloatVector.fromMemorySegment(s, env[idx], off, ORDER); }
        public float evalScalar(long off, MemorySegment[] env) { return env[idx].get(ValueLayout.JAVA_FLOAT, off); }
    }
    record ConstNode(float val) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return FloatVector.broadcast(s, val); }
        public float evalScalar(long off, MemorySegment[] env) { return val; }
    }
    record AddNode(FloatNode l, FloatNode r) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).add(r.evalVec(s, off, env)); }
        public float evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) + r.evalScalar(off, env); }
    }
    record SubNode(FloatNode l, FloatNode r) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).sub(r.evalVec(s, off, env)); }
        public float evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) - r.evalScalar(off, env); }
    }
    record MulNode(FloatNode l, FloatNode r) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).mul(r.evalVec(s, off, env)); }
        public float evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) * r.evalScalar(off, env); }
    }
    record DivNode(FloatNode l, FloatNode r) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return l.evalVec(s, off, env).div(r.evalVec(s, off, env)); }
        public float evalScalar(long off, MemorySegment[] env) { return l.evalScalar(off, env) / r.evalScalar(off, env); }
    }
    record FmaNode(FloatNode a, FloatNode b, FloatNode c) implements FloatNode {
        public FloatVector evalVec(VectorSpecies<Float> s, long off, MemorySegment[] env) { return a.evalVec(s, off, env).fma(b.evalVec(s, off, env), c.evalVec(s, off, env)); }
        public float evalScalar(long off, MemorySegment[] env) { return Math.fma(a.evalScalar(off, env), b.evalScalar(off, env), c.evalScalar(off, env)); }
    }
}