# Chapter 12: Neural Network Inference

Deploying trained neural networks in production Java environments (such as low-latency search ranking, recommendation engines, or fraud detection) typically requires heavy native dependencies like ONNX Runtime, TensorFlow Java, or LibTorch C++ JNI bindings. These runtimes introduce native crash risks, GC pinning bottlenecks, and deployment friction.

JNum includes **`jnum.nn`**, a lightweight, pure-Java neural network inference framework. Built on top of JNum's SIMD GEMM and off-heap memory, it provides standard PyTorch-style layers with zero native dependencies.

---

## 1. The `Module` Abstraction

All computational layers in `jnum.nn` implement the [`Module`](../../core/src/main/java/jnum/nn/Module.java) contract:

```java
package jnum.nn;

import jnum.NDArray;

public interface Module {
    NDArray forward(NDArray input);
}
```

A `Module` maps an input tensor to an output tensor during forward inference. Layers are thread-safe and re-entrant when evaluated across concurrent requests.

---

## 2. The Affine `Linear` Layer

The [`Linear`](../../core/src/main/java/jnum/nn/Linear.java) module implements a fully connected affine transformation:

$$y = x \cdot W^T + b$$

* **`weights`**: 2D tensor of shape `[out_features, in_features]`.
* **`bias`**: 1D tensor of shape `[out_features]`, or `null` if no bias is applied.

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.nn.Linear;

void main() {
    // 3 input features -> 4 output features
    NDArray W = JNum.rand(4, 3); // Shape [out_features, in_features]
    NDArray b = JNum.zeros(4);   // Shape [out_features]

    Linear layer = new Linear(W, b);

    // Forward pass with a batch of 2 samples: Shape [2, 3]
    NDArray x = JNum.rand(2, 3);
    NDArray y = layer.forward(x);

    IO.println("Output shape: " + y.shapeString()); // "(2, 4)"
}
```

### Rank Preservation for Sequences & Batches

Modern Transformer models operate on 3D or 4D tensors (e.g., `[batch, sequence_length, hidden_dim]`). 

JNum's `Linear` layer **automatically preserves all leading batch and sequence dimensions**, flattening only the trailing dimension into a 2D matrix for GEMM execution, and reshaping the output back to `[batch, sequence_length, out_features]` with zero intermediate copying:

```java
// Sequence of 16 tokens across a batch of 4: [4, 16, 512]
NDArray sequenceInput = JNum.rand(4, 16, 512);

// Project from hidden dimension 512 to 2048
Linear projection = new Linear(JNum.rand(2048, 512), JNum.zeros(2048));

NDArray projected = projection.forward(sequenceInput);
IO.println(projected.shapeString()); // Output: "(4, 16, 2048)"
```

---

## 3. Activation Functions

JNum provides four standard activation modules:

### A. Rectified Linear Unit (`ReLU`)

$$\text{ReLU}(x) = \max(x, 0)$$

Branchless SIMD clamping:

```java
import jnum.nn.ReLU;

ReLU relu = new ReLU();
NDArray activated = relu.forward(x);
```

### B. Logistic `Sigmoid`

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Used for binary classification and attention gate mechanisms:

```java
import jnum.nn.Sigmoid;

Sigmoid sigmoid = new Sigmoid();
NDArray gates = sigmoid.forward(x);
```

### C. Hyperbolic Tangent (`Tanh`)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Outputs zero-centered values in the interval $(-1.0, 1.0)$:

```java
import jnum.nn.Tanh;

Tanh tanh = new Tanh();
NDArray normalized = tanh.forward(x);
```

### D. Numerically Stabilized `Softmax`

The [`Softmax`](../../core/src/main/java/jnum/nn/Softmax.java) module normalizes unnormalized logits into a valid probability distribution that sums to `1.0`:

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

> [!NOTE]
> JNum automatically subtracts the maximum value along the axis before exponentiation (`shifted = input.sub(maxVals)`). This standard numerical stability trick prevents floating-point overflow (`Infinity`) when logits are large.

```java
import jnum.nn.Softmax;

// Normalizes across the last dimension (axis = -1 by default)
Softmax softmax = new Softmax();

NDArray logits = JNum.from(new float[]{2.0f, 1.0f, 0.1f}, 1, 3);
NDArray probabilities = softmax.forward(logits);

IO.println("Class probabilities:");
IO.println(probabilities);
IO.println("Sum of probabilities: " + probabilities.sum()); // 1.0
```

---

## 4. The `Sequential` Pipeline Container

Chain layers together using [`Sequential`](../../core/src/main/java/jnum/nn/Sequential.java). The output of each layer is passed as the input to the next:

```java
import jnum.JNum;
import jnum.NDArray;
import jnum.nn.*;

void main() {
    // Build a 3-layer Multi-Layer Perceptron (MLP)
    // Input: 128 -> Hidden 1: 64 -> Hidden 2: 32 -> Output: 2 classes
    Sequential classifier = new Sequential(
        new Linear(JNum.rand(64, 128), JNum.zeros(64)),
        new ReLU(),
        new Linear(JNum.rand(32, 64), JNum.zeros(32)),
        new ReLU(),
        new Linear(JNum.rand(2, 32), JNum.zeros(2)),
        new Softmax(-1)
    );

    // Forward inference on a batch of 8 samples: [8, 128]
    NDArray inputBatch = JNum.rand(8, 128);
    NDArray outputProbabilities = classifier.forward(inputBatch);

    IO.println("Predictions shape: " + outputProbabilities.shapeString()); // "(8, 2)"
}
```

---

## 5. Practical Recipe: End-to-End Inference from Safetensors

The following production recipe loads weights from a `.safetensors` file and runs forward inference on an embedding vector:

```java
import java.nio.file.Path;
import jnum.NDArray;
import jnum.io.JNumIO;
import jnum.io.MmapTensors;
import jnum.nn.*;

void main() throws Exception {
    Path modelPath = Path.of("models/classifier.safetensors");

    // 1. Memory-map weights directly from disk
    try (MmapTensors weights = JNumIO.readSafetensorsMmapScoped(modelPath)) {
        // 2. Instantiate layers using mapped weights
        Linear layer1 = new Linear(
            weights.get("layer1.weight"),
            weights.get("layer1.bias")
        );
        Linear layer2 = new Linear(
            weights.get("layer2.weight"),
            weights.get("layer2.bias")
        );

        Sequential model = new Sequential(
            layer1,
            new ReLU(),
            layer2,
            new Softmax()
        );

        // 3. Run inference on input vector
        NDArray sample = jnum.JNum.rand(1, 128);
        NDArray prediction = model.forward(sample);

        IO.println("Inference complete! Predicted probabilities: " + prediction);
    }
}
```

Now proceed to **[Chapter 13: Idiomatic Kotlin DSL](13-idiomatic-kotlin-dsl.md)** to see how Kotlin extensions unlock concise mathematical syntax.
