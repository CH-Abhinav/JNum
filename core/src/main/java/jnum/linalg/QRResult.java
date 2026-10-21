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
package jnum.linalg;

import jnum.NDArray;

/**
 * Container record holding the orthogonal matrix {@code Q} and upper-triangular matrix {@code R}
 * of a QR decomposition ({@code A = Q * R}).
 *
 * @param q orthogonal matrix {@code Q}
 * @param r upper-triangular matrix {@code R}
 */
public record QRResult(NDArray q, NDArray r) {}
