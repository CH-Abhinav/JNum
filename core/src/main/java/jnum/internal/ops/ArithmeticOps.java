/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.internal.ops;

import java.lang.foreign.Arena;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.kernel.arithmetic.*;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.layout.TypeUtil;
import jnum.internal.layout.ValidUtil;

/**
 * High-level orchestration facade for binary arithmetic operations (add, sub, mul, div).
 *
 * <p>Handles NumPy-compatible broadcasting shape derivation, operand type promotion via
 * {@link TypeUtil#promoteTypes}, optional destination buffer validation, and dispatch to
 * type-specialized SIMD kernels in {@link jnum.internal.kernel.arithmetic}.</p>
 */
public final class ArithmeticOps {

    private ArithmeticOps() {
        throw new AssertionError("Utility class cannot be instantiated.");
    }

    // =========================================================================
    // Addition (Array-Array & Array-Scalar)
    // =========================================================================

    /**
     * Element-wise addition with broadcasting and automatic type promotion. Allocates output.
     */
    public static NDArray add(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Add.addFloat(A, B, resArray);
            case f64 -> Add.addDouble(A, B, resArray);
            case i32 -> Add.addInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, NDArray b, Arena arena) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        return add(a, b, JNum.zeros(arena, targetType, targetShape));
    }

    public static NDArray add(NDArray a, NDArray b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch (targetType) {
            case f32 -> Add.addFloat(A, B, targetRes);
            case f64 -> Add.addDouble(A, B, targetRes);
            case i32 -> Add.addInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, b, resArray);
            case f64 -> Add.addDouble(A, b, resArray);
            case i32 -> Add.addInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, b, resArray);
            case f64 -> Add.addDouble(A, b, resArray);
            case i32 -> Add.addInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, (float) b, resArray);
            case f64 -> Add.addDouble(A, b, resArray);
            case i32 -> Add.addInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, float b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, b, targetRes);
            case f64 -> Add.addDouble(A, b, targetRes);
            case i32 -> Add.addInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, int b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, b, targetRes);
            case f64 -> Add.addDouble(A, b, targetRes);
            case i32 -> Add.addInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray add(NDArray a, double b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Add.addFloat(A, (float) b, targetRes);
            case f64 -> Add.addDouble(A, b, targetRes);
            case i32 -> Add.addInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // =========================================================================
    // High-Level Subtraction
    // =========================================================================

    public static NDArray sub(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, B, resArray);
            case f64 -> Sub.subDouble(A, B, resArray);
            case i32 -> Sub.subInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, NDArray b, Arena arena) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        return sub(a, b, JNum.zeros(arena, targetType, targetShape));
    }

    public static NDArray sub(NDArray a, NDArray b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, B, targetRes);
            case f64 -> Sub.subDouble(A, B, targetRes);
            case i32 -> Sub.subInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, b, resArray);
            case f64 -> Sub.subDouble(A, b, resArray);
            case i32 -> Sub.subInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, b, resArray);
            case f64 -> Sub.subDouble(A, b, resArray);
            case i32 -> Sub.subInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, (float) b, resArray);
            case f64 -> Sub.subDouble(A, b, resArray);
            case i32 -> Sub.subInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, float b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, b, targetRes);
            case f64 -> Sub.subDouble(A, b, targetRes);
            case i32 -> Sub.subInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, int b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, b, targetRes);
            case f64 -> Sub.subDouble(A, b, targetRes);
            case i32 -> Sub.subInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray sub(NDArray a, double b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Sub.subFloat(A, (float) b, targetRes);
            case f64 -> Sub.subDouble(A, b, targetRes);
            case i32 -> Sub.subInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // =========================================================================
    // High-Level Multiplication
    // =========================================================================

    public static NDArray mul(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, B, resArray);
            case f64 -> Mul.mulDouble(A, B, resArray);
            case i32 -> Mul.mulInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, NDArray b, Arena arena) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        return mul(a, b, JNum.zeros(arena, targetType, targetShape));
    }

    public static NDArray mul(NDArray a, NDArray b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, B, targetRes);
            case f64 -> Mul.mulDouble(A, B, targetRes);
            case i32 -> Mul.mulInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, b, resArray);
            case f64 -> Mul.mulDouble(A, b, resArray);
            case i32 -> Mul.mulInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, b, resArray);
            case f64 -> Mul.mulDouble(A, b, resArray);
            case i32 -> Mul.mulInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, (float) b, resArray);
            case f64 -> Mul.mulDouble(A, b, resArray);
            case i32 -> Mul.mulInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, float b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, b, targetRes);
            case f64 -> Mul.mulDouble(A, b, targetRes);
            case i32 -> Mul.mulInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, int b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, b, targetRes);
            case f64 -> Mul.mulDouble(A, b, targetRes);
            case i32 -> Mul.mulInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray mul(NDArray a, double b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Mul.mulFloat(A, (float) b, targetRes);
            case f64 -> Mul.mulDouble(A, b, targetRes);
            case i32 -> Mul.mulInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // =========================================================================
    // High-Level Division
    // =========================================================================

    public static NDArray div(NDArray a, NDArray b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> Div.divFloat(A, B, resArray);
            case f64 -> Div.divDouble(A, B, resArray);
            case i32 -> Div.divInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, NDArray b, Arena arena) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        return div(a, b, JNum.zeros(arena, targetType, targetShape));
    }

    public static NDArray div(NDArray a, NDArray b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(a.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(a, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch (targetType) {
            case f32 -> Div.divFloat(A, B, targetRes);
            case f64 -> Div.divDouble(A, B, targetRes);
            case i32 -> Div.divInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, float b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, b, resArray);
            case f64 -> Div.divDouble(A, b, resArray);
            case i32 -> Div.divInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, int b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, b, resArray);
            case f64 -> Div.divDouble(A, b, resArray);
            case i32 -> Div.divInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, double b) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, (float) b, resArray);
            case f64 -> Div.divDouble(A, b, resArray);
            case i32 -> Div.divInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, float b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, b, targetRes);
            case f64 -> Div.divDouble(A, b, targetRes);
            case i32 -> Div.divInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, int b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, b, targetRes);
            case f64 -> Div.divDouble(A, b, targetRes);
            case i32 -> Div.divInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray div(NDArray a, double b, NDArray resArray) {
        DType targetType = TypeUtil.promoteTypes(a.getDType(), TypeUtil.scalarType(b));
        NDArray A = a.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, a.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> Div.divFloat(A, (float) b, targetRes);
            case f64 -> Div.divDouble(A, b, targetRes);
            case i32 -> Div.divInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }
}
