package jnum.nn;

import jnum.NDArray;

/**
 * Fully connected (affine) linear transformation layer: {@code y = x * W^T + b}.
 *
 * <p>Supports arbitrary input ranks (e.g. 2D {@code [batch, in_features]} or N-D sequence tensors
 * {@code [batch, seq_len, in_features]}). For inputs with rank greater than 2, all leading dimensions
 * are preserved while transforming the trailing feature dimension.</p>
 */
public class Linear implements Module {

    private final NDArray weights;
    private final NDArray bias;

    /**
     * Constructs a linear layer with the given weights and optional bias.
     *
     * @param weights 2D tensor of shape {@code [out_features, in_features]}
     * @param bias 1D tensor of shape {@code [out_features]}, or {@code null} if no bias is applied
     * @throws IllegalArgumentException if {@code weights} is not a 2D matrix
     */
    public Linear(NDArray weights, NDArray bias) {
        if (weights.dim() != 2) {
            throw new IllegalArgumentException("Linear weights must be 2D [out_features, in_features]");
        }
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * Returns the weight tensor of this layer.
     *
     * @return 2D weights tensor
     */
    public NDArray weights() {
        return weights;
    }

    /**
     * Returns the bias tensor of this layer, or {@code null} if no bias was provided.
     *
     * @return bias tensor, or null
     */
    public NDArray bias() {
        return bias;
    }

    /**
     * Applies the affine transformation to the input tensor.
     *
     * @param input tensor whose last dimension equals {@code in_features}
     * @return transformed tensor whose last dimension equals {@code out_features}
     * @throws IllegalArgumentException if the trailing dimension of {@code input} does not match {@code in_features}
     */
    @Override
    public NDArray forward(NDArray input) {
        long[] originalShape = input.getShape();
        int dims = originalShape.length;

        long inFeatures = originalShape[dims - 1];
        if (inFeatures != weights.internalShapeUnsafe()[1]) {
            throw new IllegalArgumentException("Input features mismatch: expected " +
                    weights.internalShapeUnsafe()[1] + " but got " + inFeatures);
        }

        long collapsedBatch = 1;
        for (int i = 0; i < dims - 1; i++) {
            collapsedBatch *= originalShape[i];
        }

        NDArray flattenedInput = input.dim() == 2 ? input : input.reshape(collapsedBatch, inFeatures);

        NDArray output = flattenedInput.matmul(weights.transpose());
        if (bias != null) {
            output.addi(bias);
        }

        if (dims > 2) {
            long[] outShape = originalShape.clone();
            outShape[dims - 1] = weights.internalShapeUnsafe()[0];
            return output.reshape(outShape);
        }

        return output;
    }
}
