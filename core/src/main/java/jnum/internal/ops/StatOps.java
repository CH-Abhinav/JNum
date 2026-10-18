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
import jnum.internal.kernel.stat.Average;
import jnum.internal.kernel.stat.Cov;
import jnum.internal.kernel.stat.CumProd;
import jnum.internal.kernel.stat.CumSum;
import jnum.internal.kernel.stat.Diff;
import jnum.internal.kernel.stat.Mean;
import jnum.internal.kernel.stat.Median;
import jnum.internal.kernel.stat.Mode;
import jnum.internal.kernel.stat.Ptp;
import jnum.internal.kernel.stat.Quantile;
import jnum.internal.kernel.stat.Std;
import jnum.internal.kernel.stat.Var;
import jnum.internal.layout.ShapeUtil;
import jnum.stat.QuantileMethod;

/**
 * Internal orchestrator for statistical operations: validation, shape calculations,
 * dtype promotions, and kernel dispatching.
 */
public final class StatOps {

    private StatOps() {
        throw new AssertionError("StatOps cannot be instantiated.");
    }

    // =========================================================================
    // MEAN
    // =========================================================================

    public static double mean(NDArray a) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Mean.meanFloat(a);
            case f64 -> Mean.meanDouble(a);
            case i32 -> Mean.meanInt(a);
            case bool -> Mean.meanBool(a);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for mean.");
        };
    }

    public static NDArray mean(NDArray a, int axis, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = (a.getDType() == DType.f32) ? DType.f32 : DType.f64;
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Mean.meanFloatAxis(a, axis, resArray);
            case f64 -> Mean.meanDoubleAxis(a, axis, resArray);
            case i32 -> Mean.meanIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis mean.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // AVERAGE (WEIGHTED MEAN)
    // =========================================================================

    public static double average(NDArray a, NDArray weights) {
        if (weights == null) {
            return mean(a);
        }
        DType targetDType = (a.getDType() == DType.f64 || weights.getDType() == DType.f64 || a.getDType() == DType.i32)
                ? DType.f64 : DType.f32;

        NDArray safeA = a.getDType() == targetDType ? a : a.cast(targetDType);
        NDArray safeW = weights.getDType() == targetDType ? weights : weights.cast(targetDType);

        return switch (targetDType) {
            case f32 -> Average.averageFloat(safeA, safeW);
            case f64 -> Average.averageDouble(safeA, safeW);
            default -> throw new UnsupportedOperationException("Unsupported dtype for average: " + targetDType);
        };
    }

    public static NDArray average(NDArray a, int axis, NDArray weights, boolean keepDims, Arena arena) {
        if (weights == null) {
            return mean(a, axis, keepDims, arena);
        }
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        DType targetDType = (a.getDType() == DType.f64 || weights.getDType() == DType.f64 || a.getDType() == DType.i32)
                ? DType.f64 : DType.f32;

        NDArray safeA = a.getDType() == targetDType ? a : a.cast(targetDType);
        NDArray safeW = weights.getDType() == targetDType ? weights : weights.cast(targetDType);

        long[] reducedShape = ShapeUtil.calculateReductionShape(safeA.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(arena, targetDType, reducedShape);

        NDArray res = switch (targetDType) {
            case f32 -> Average.averageFloatAxis(safeA, axis, safeW, resArray);
            case f64 -> Average.averageDoubleAxis(safeA, axis, safeW, resArray);
            default -> throw new UnsupportedOperationException("Unsupported dtype for average: " + targetDType);
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // VARIANCE
    // =========================================================================

    public static double var(NDArray a, int ddof) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Var.varFloat(a, ddof);
            case f64 -> Var.varDouble(a, ddof);
            case i32 -> Var.varInt(a, ddof);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for variance.");
        };
    }

    public static NDArray var(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = (a.getDType() == DType.f32) ? DType.f32 : DType.f64;
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Var.varFloatAxis(a, axis, ddof, resArray);
            case f64 -> Var.varDoubleAxis(a, axis, ddof, resArray);
            case i32 -> Var.varIntAxis(a, axis, ddof, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis variance.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // STANDARD DEVIATION
    // =========================================================================

    public static double std(NDArray a, int ddof) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Std.stdFloat(a, ddof);
            case f64 -> Std.stdDouble(a, ddof);
            case i32 -> Std.stdInt(a, ddof);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for std.");
        };
    }

    public static NDArray std(NDArray a, int axis, int ddof, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = (a.getDType() == DType.f32) ? DType.f32 : DType.f64;
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Std.stdFloatAxis(a, axis, ddof, resArray);
            case f64 -> Std.stdDoubleAxis(a, axis, ddof, resArray);
            case i32 -> Std.stdIntAxis(a, axis, ddof, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis std.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // PEAK-TO-PEAK (PTP: MAX - MIN)
    // =========================================================================

    public static double ptp(NDArray a) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Ptp.ptpFloat(a);
            case f64 -> Ptp.ptpDouble(a);
            case i32 -> Ptp.ptpInt(a);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for ptp.");
        };
    }

    public static NDArray ptp(NDArray a, int axis, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = a.getDType();
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Ptp.ptpFloatAxis(a, axis, resArray);
            case f64 -> Ptp.ptpDoubleAxis(a, axis, resArray);
            case i32 -> Ptp.ptpIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis ptp.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // MEDIAN
    // =========================================================================

    public static double median(NDArray a) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Median.medianFloat(a);
            case f64 -> Median.medianDouble(a);
            case i32 -> Median.medianInt(a);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for median.");
        };
    }

    public static NDArray median(NDArray a, int axis, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = (a.getDType() == DType.f32) ? DType.f32 : DType.f64;
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Median.medianFloatAxis(a, axis, resArray);
            case f64 -> Median.medianDoubleAxis(a, axis, resArray);
            case i32 -> Median.medianIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis median.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // QUANTILE & PERCENTILE
    // =========================================================================

    public static double quantile(NDArray a, double q, QuantileMethod method) {
        if (q < 0.0 || q > 1.0) {
            throw new IllegalArgumentException("Quantile must be in the range [0.0, 1.0], got: " + q);
        }
        if (a.getSize() == 0) return Double.NaN;
        QuantileMethod safeMethod = method == null ? QuantileMethod.LINEAR : method;

        return switch (a.getDType()) {
            case f32 -> Quantile.quantileFloat(a, q, safeMethod);
            case f64 -> Quantile.quantileDouble(a, q, safeMethod);
            case i32 -> Quantile.quantileInt(a, q, safeMethod);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for quantile.");
        };
    }

    public static NDArray quantile(NDArray a, double q, int axis, QuantileMethod method, boolean keepDims, Arena arena) {
        if (q < 0.0 || q > 1.0) {
            throw new IllegalArgumentException("Quantile must be in the range [0.0, 1.0], got: " + q);
        }
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        QuantileMethod safeMethod = method == null ? QuantileMethod.LINEAR : method;
        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        DType outDType = (a.getDType() == DType.f32) ? DType.f32 : DType.f64;
        NDArray resArray = JNum.zeros(arena, outDType, reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Quantile.quantileFloatAxis(a, axis, q, safeMethod, resArray);
            case f64 -> Quantile.quantileDoubleAxis(a, axis, q, safeMethod, resArray);
            case i32 -> Quantile.quantileIntAxis(a, axis, q, safeMethod, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis quantile.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // INTERQUARTILE RANGE (IQR: Q75 - Q25)
    // =========================================================================

    public static double iqr(NDArray a) {
        return quantile(a, 0.75, QuantileMethod.LINEAR) - quantile(a, 0.25, QuantileMethod.LINEAR);
    }

    public static NDArray iqr(NDArray a, int axis, boolean keepDims, Arena arena) {
        NDArray q75 = quantile(a, 0.75, axis, QuantileMethod.LINEAR, keepDims, arena);
        NDArray q25 = quantile(a, 0.25, axis, QuantileMethod.LINEAR, keepDims, arena);
        return q75.sub(q25);
    }

    // =========================================================================
    // MODE
    // =========================================================================

    public static double mode(NDArray a) {
        if (a.getSize() == 0) return Double.NaN;
        return switch (a.getDType()) {
            case f32 -> Mode.modeFloat(a);
            case f64 -> Mode.modeDouble(a);
            case i32 -> Mode.modeInt(a);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for mode.");
        };
    }

    public static NDArray mode(NDArray a, int axis, boolean keepDims, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        long[] reducedShape = ShapeUtil.calculateReductionShape(a.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(arena, a.getDType(), reducedShape);

        NDArray res = switch (a.getDType()) {
            case f32 -> Mode.modeFloatAxis(a, axis, resArray);
            case f64 -> Mode.modeDoubleAxis(a, axis, resArray);
            case i32 -> Mode.modeIntAxis(a, axis, resArray);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for axis mode.");
        };

        if (keepDims) {
            return res.unsqueeze(axis);
        }
        return res;
    }

    // =========================================================================
    // COVARIANCE
    // =========================================================================

    public static double cov(NDArray x, NDArray y, int ddof) {
        if (x.getSize() != y.getSize()) {
            throw new IllegalArgumentException("Arrays must have the same size for covariance: " +
                    x.getSize() + " vs " + y.getSize());
        }
        if (x.getDType() == DType.f32 && y.getDType() == DType.f32) {
            return Cov.covFloat(x, y, ddof);
        } else if (x.getDType() == DType.i32 && y.getDType() == DType.i32) {
            return Cov.covInt(x, y, ddof);
        } else {
            NDArray xCast = (x.getDType() == DType.f64) ? x : x.cast(DType.f64);
            NDArray yCast = (y.getDType() == DType.f64) ? y : y.cast(DType.f64);
            return Cov.covDouble(xCast, yCast, ddof);
        }
    }

    public static NDArray cov(NDArray m, boolean rowvar, int ddof, Arena arena) {
        return Cov.covMatrix(m, rowvar, ddof, arena);
    }

    // =========================================================================
    // CUMSUM
    // =========================================================================

    public static NDArray cumsum(NDArray a, Arena arena) {
        long total = a.getSize();
        NDArray res = JNum.zeros(arena, a.getDType(), total);
        return switch (a.getDType()) {
            case f32 -> CumSum.cumsumFloatFlat(a, res);
            case f64 -> CumSum.cumsumDoubleFlat(a, res);
            case i32 -> CumSum.cumsumIntFlat(a, res);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for cumsum.");
        };
    }

    public static NDArray cumsum(NDArray a, int axis, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        NDArray res = JNum.zeros(arena, a.getDType(), a.internalShapeUnsafe());
        return switch (a.getDType()) {
            case f32 -> CumSum.cumsumFloatAxis(a, axis, res);
            case f64 -> CumSum.cumsumDoubleAxis(a, axis, res);
            case i32 -> CumSum.cumsumIntAxis(a, axis, res);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for cumsum.");
        };
    }

    // =========================================================================
    // CUMPROD
    // =========================================================================

    public static NDArray cumprod(NDArray a, Arena arena) {
        long total = a.getSize();
        NDArray res = JNum.zeros(arena, a.getDType(), total);
        return switch (a.getDType()) {
            case f32 -> CumProd.cumprodFloatFlat(a, res);
            case f64 -> CumProd.cumprodDoubleFlat(a, res);
            case i32 -> CumProd.cumprodIntFlat(a, res);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for cumprod.");
        };
    }

    public static NDArray cumprod(NDArray a, int axis, Arena arena) {
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        NDArray res = JNum.zeros(arena, a.getDType(), a.internalShapeUnsafe());
        return switch (a.getDType()) {
            case f32 -> CumProd.cumprodFloatAxis(a, axis, res);
            case f64 -> CumProd.cumprodDoubleAxis(a, axis, res);
            case i32 -> CumProd.cumprodIntAxis(a, axis, res);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for cumprod.");
        };
    }

    // =========================================================================
    // DIFF
    // =========================================================================

    public static NDArray diff(NDArray a, int n, int axis, Arena arena) {
        return Diff.diff(a, n, axis, arena);
    }
}
