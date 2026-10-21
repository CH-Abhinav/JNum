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

/**
 * Container record holding the sign and log-absolute determinant of a square matrix.
 *
 * @param sign      sign of the determinant ({@code -1.0}, {@code 0.0}, or {@code +1.0})
 * @param logAbsDet natural logarithm of the absolute value of the determinant ({@code ln|det(A)|})
 */
public record SlogdetResult(double sign, double logAbsDet) {}
