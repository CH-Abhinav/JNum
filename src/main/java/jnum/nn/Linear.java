package jnum.nn;

import jnum.NDArray;

public class Linear {
    private final NDArray weights;
    private final NDArray bias;

    public Linear(NDArray weights,NDArray bias){
        if(weights.dim()!=2){
            throw new IllegalArgumentException("Linear weights must be 2D [out_features, in_features]");
        }
        this.weights = weights;
        this.bias = bias;
    }

    public NDArray forward(NDArray input){
        long[] originalShape=input.getShape();
        int dims=originalShape.length;

        long inFeatures = originalShape[dims-1];
        if(inFeatures!=weights.internalShapeUnsafe()[1]){
            throw new IllegalArgumentException("Input features mismatch");
        }

        long collapsedBatch = 1;
        for (int i = 0; i < dims - 1; i++) {
            collapsedBatch *= originalShape[i];
        }

        NDArray flattenedInput = input.dim() == 2 ? input : input.reshape(collapsedBatch, inFeatures);

        NDArray out2D = flattenedInput.matmul(weights.transpose());
        if (bias != null) {
            out2D.addi(bias);
        }

        if (dims > 2) {
            long[] outShape = originalShape.clone();
            outShape[dims - 1] = weights.internalShapeUnsafe()[0];
            return out2D.reshape(outShape);
        }

        return out2D;
    }

}
