package jnum.internal.ops;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.reduce.*;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.layout.TypeUtil;

public class ReduceOps {

    private ReduceOps(){
        throw new AssertionError();
    }

    // =========================================================================
    // High-Level Reductions
    // =========================================================================

    public static double max(NDArray a) {
        return switch (a.getDType()) {
            case f32 -> Max.maxFloat(a);
            case f64 -> Max.maxDouble(a);
            case i32 -> Max.maxInt(a);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static double min(NDArray a) {
        return switch (a.getDType()) {
            case f32 -> Min.minFloat(a);
            case f64 -> Min.minDouble(a);
            case i32 -> Min.minInt(a);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static double sum(NDArray a) {
        return switch (a.getDType()) {
            case f32 -> Sum.sumFloat(a);
            case f64 -> Sum.sumDouble(a);
            case i32 -> Sum.sumInt(a);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static NDArray sum(NDArray a, int axis) {
        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(a.getDType(), reducedShape);
        return switch (a.getDType()) {
            case f32 -> Sum.sumFloatAxis(a, axis, resArray);
            case f64 -> Sum.sumDoubleAxis(a, axis, resArray);
            case i32 -> Sum.sumIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static NDArray max(NDArray a, int axis) {
        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(a.getDType(), reducedShape);
        return switch (a.getDType()) {
            case f32 -> Max.maxFloatAxis(a, axis, resArray);
            case f64 -> Max.maxDoubleAxis(a, axis, resArray);
            case i32 -> Max.maxIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static NDArray min(NDArray a, int axis) {
        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(a.getDType(), reducedShape);
        return switch (a.getDType()) {
            case f32 -> Min.minFloatAxis(a, axis, resArray);
            case f64 -> Min.minDoubleAxis(a, axis, resArray);
            case i32 -> Min.minIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
        };
    }

    public static double dot(NDArray a, NDArray b) {
        if (a.dim() != 1 || b.dim() != 1) {
            throw new IllegalArgumentException("Dot product requires 1D vectors. Shapes: " + a.shapeString() + ", " + b.shapeString());
        }
        if (a.getSize() != b.getSize()) {
            throw new IllegalArgumentException("Vector sizes must match for dot product.");
        }
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        NDArray A = a.cast(targetType);
        NDArray B = b.cast(targetType);
        if (!A.isContiguous()) {
            A = A.contiguous();
        }
        if (!B.isContiguous()) {
            B = B.contiguous();
        }
        return switch (targetType) {
            case f32 -> Dot.dotFloat(A, B);
            case i32 -> Dot.dotInt(A, B);
            case f64 -> Dot.dotDouble(A, B);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // =========================================================================
    // Low-Level Kernel Forwarders
    // =========================================================================

    public static double sumFloat(NDArray a){
        return Sum.sumFloat(a);
    }

    public static double sumInt(NDArray a){
        return Sum.sumInt(a);
    }

    public static double sumDouble(NDArray a){
        return Sum.sumDouble(a);
    }

    public static NDArray sumFloatAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumFloatAxis(a, axis, resArray);
    }

    public static NDArray sumIntAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumIntAxis(a, axis, resArray);
    }

    public static NDArray sumDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Sum.sumDoubleAxis(a, axis, resArray);
    }

    public static double maxFloat(NDArray a){
        return Max.maxFloat(a);
    }

    public static NDArray maxFloatAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxFloatAxis(a, axis, resArray);
    }

    public static double maxInt(NDArray a) {
        return Max.maxInt(a);
    }

    public static NDArray maxIntAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxIntAxis(a, axis, resArray);
    }

    public static double maxDouble(NDArray a) {
        return Max.maxDouble(a);
    }

    public static NDArray maxDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Max.maxDoubleAxis(a, axis, resArray);
    }

    public static double minFloat(NDArray a){
        return Min.minFloat(a);
    }

    public static NDArray minFloatAxis(NDArray a,int axis,NDArray resArray){
        return Min.minFloatAxis(a, axis, resArray);
    }

    public static double minInt(NDArray a) {
        return Min.minInt(a);
    }

    public static NDArray minIntAxis(NDArray a,int axis,NDArray resArray){
        return Min.minIntAxis(a, axis, resArray);
    }

    public static double minDouble(NDArray a) {
        return Min.minDouble(a);
    }

    public static NDArray minDoubleAxis(NDArray a,int axis,NDArray resArray){
        return Min.minDoubleAxis(a, axis, resArray);
    }

    public static double dotFloat(NDArray a,NDArray b){
        return Dot.dotFloat(a, b);
    }

    public static double dotInt(NDArray a,NDArray b){
        return Dot.dotInt(a, b);
    }

    public static double dotDouble(NDArray a,NDArray b){
        return Dot.dotDouble(a, b);
    }
}
