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