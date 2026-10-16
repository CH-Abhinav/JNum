/**
 * Neural network layers, activation functions, and computational graph containers.
 *
 * <p>Inspired by PyTorch's {@code torch.nn}, this package provides high-performance composable modules
 * for deep learning inference and forward-pass evaluation:
 * <ul>
 *   <li>{@link jnum.nn.Module}: Base interface for all network layers.</li>
 *   <li>{@link jnum.nn.Linear}: Fully-connected affine transformation ($y = x W^T + b$) with multi-dimensional batch support.</li>
 *   <li>{@link jnum.nn.Sequential}: Chained sequential pipeline container.</li>
 *   <li>Activations: {@link jnum.nn.ReLU}, {@link jnum.nn.Sigmoid}, {@link jnum.nn.Tanh}, and numerically stable {@link jnum.nn.Softmax}.</li>
 * </ul>
 * </p>
 */
package jnum.nn;