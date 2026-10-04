package jnum.internal.ops;

import java.lang.foreign.Arena;
import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.layout.TypeUtil;
import jnum.internal.layout.ValidUtil;
import jnum.internal.kernel.linalg.Cholesky;
import jnum.internal.kernel.linalg.Cond;
import jnum.internal.kernel.linalg.CosineSimilarity;
import jnum.internal.kernel.linalg.Det;
import jnum.internal.kernel.linalg.Det.SlogdetResult;
import jnum.internal.kernel.linalg.Eig;
import jnum.internal.kernel.linalg.Eig.EigResult;
import jnum.internal.kernel.linalg.Eigh;
import jnum.internal.kernel.linalg.Eigh.EighResult;
import jnum.internal.kernel.linalg.Inv;
import jnum.internal.kernel.linalg.MatMul;
import jnum.internal.kernel.linalg.MatrixRank;
import jnum.internal.kernel.linalg.Norm;
import jnum.internal.kernel.linalg.Pinv;
import jnum.internal.kernel.linalg.QR;
import jnum.internal.kernel.linalg.QR.QRResult;
import jnum.internal.kernel.linalg.Solve;
import jnum.internal.kernel.linalg.SVD;
import jnum.internal.kernel.linalg.SVD.SVDResult;
import jnum.internal.kernel.linalg.Trace;

public final class LinalgOps {

    private LinalgOps() {
        throw new AssertionError("No jnum.internal.ops.LinalgOps instances for you!");
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        ValidUtil.validateMatmulInputs(a, b);
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        NDArray A = a.cast(targetType);
        NDArray B = b.cast(targetType);
        long[] targetShape = new long[]{a.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch (targetType) {
            case f32 -> MatMul.matmulFloat(A, B, resArray);
            case f64 -> MatMul.matmulDouble(A, B, resArray);
            case i32 -> MatMul.matmulInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    public static NDArray matmul(NDArray a, NDArray b, NDArray resArray) {
        ValidUtil.validateMatmulInputs(a, b);
        DType targetType = TypeUtil.promoteTypes(a.getDType(), b.getDType());
        NDArray A = a.cast(targetType);
        NDArray B = b.cast(targetType);
        long[] targetShape = new long[]{a.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        ValidUtil.validateOutputBuffer(targetRes);
        return switch (targetType) {
            case f32 -> MatMul.matmulFloat(A, B, targetRes);
            case f64 -> MatMul.matmulDouble(A, B, targetRes);
            case i32 -> MatMul.matmulInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype " + targetType + " doesn't support this method");
        };
    }

    // Trace
    public static double trace(NDArray a) {
        return Trace.compute(a, 0);
    }

    public static double trace(NDArray a, int offset) {
        return Trace.compute(a, offset);
    }

    // Norm
    public static double norm(NDArray a) {
        return Norm.norm(a, 2);
    }

    public static double norm(NDArray a, int ord) {
        return Norm.norm(a, ord);
    }

    // Determinant
    public static double det(NDArray a) {
        if (a.getDType() == DType.f32) {
            return Det.detFloat(a);
        }
        return Det.detDouble(a);
    }

    public static SlogdetResult slogdet(NDArray a) {
        return Det.slogdet(a);
    }

    // Inversion
    public static NDArray inv(NDArray a) {
        return Inv.inv(a, Arena.ofAuto());
    }

    public static NDArray inv(NDArray a, Arena arena) {
        return Inv.inv(a, arena);
    }

    // Linear Solve (Ax = b)
    public static NDArray solve(NDArray a, NDArray b) {
        return Solve.solve(a, b, Arena.ofAuto());
    }

    public static NDArray solve(NDArray a, NDArray b, Arena arena) {
        return Solve.solve(a, b, arena);
    }

    // Cholesky
    public static NDArray cholesky(NDArray a) {
        return Cholesky.cholesky(a, Arena.ofAuto());
    }

    public static NDArray cholesky(NDArray a, Arena arena) {
        return Cholesky.cholesky(a, arena);
    }

    // QR
    public static QRResult qr(NDArray a) {
        return QR.qr(a, Arena.ofAuto());
    }

    public static QRResult qr(NDArray a, Arena arena) {
        return QR.qr(a, arena);
    }

    // Eigh (Symmetric Eigenvalues)
    public static EighResult eigh(NDArray a) {
        return Eigh.eigh(a, Arena.ofAuto());
    }

    public static EighResult eigh(NDArray a, Arena arena) {
        return Eigh.eigh(a, arena);
    }

    // SVD
    public static SVDResult svd(NDArray a) {
        return SVD.svd(a, Arena.ofAuto());
    }

    public static SVDResult svd(NDArray a, Arena arena) {
        return SVD.svd(a, arena);
    }

    // Matrix Rank
    public static int matrixRank(NDArray a) {
        return MatrixRank.matrixRank(a, Arena.ofAuto());
    }

    public static int matrixRank(NDArray a, double tol) {
        return MatrixRank.matrixRank(a, tol, Arena.ofAuto());
    }

    public static int matrixRank(NDArray a, double tol, Arena arena) {
        return MatrixRank.matrixRank(a, tol, arena);
    }

    // Condition Number
    public static double cond(NDArray a) {
        return Cond.cond(a, Arena.ofAuto());
    }

    public static double cond(NDArray a, Arena arena) {
        return Cond.cond(a, arena);
    }

    // Pseudo-Inverse
    public static NDArray pinv(NDArray a) {
        return Pinv.pinv(a, Arena.ofAuto());
    }

    public static NDArray pinv(NDArray a, double rcond) {
        return Pinv.pinv(a, rcond, Arena.ofAuto());
    }

    public static NDArray pinv(NDArray a, double rcond, Arena arena) {
        return Pinv.pinv(a, rcond, arena);
    }

    // Eig (General Eigenvalues)
    public static EigResult eig(NDArray a) {
        return Eig.eig(a, Arena.ofAuto());
    }

    public static EigResult eig(NDArray a, Arena arena) {
        return Eig.eig(a, arena);
    }

    // Cosine Similarity
    public static double cosineSimilarity(NDArray a, NDArray b) {
        return CosineSimilarity.compute(a, b);
    }
}
