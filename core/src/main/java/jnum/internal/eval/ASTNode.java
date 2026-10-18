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
package jnum.internal.eval;

public sealed interface ASTNode {
    record Var(int index) implements ASTNode {}
    record Const(double value) implements ASTNode {}
    record Add(ASTNode left, ASTNode right) implements ASTNode {}
    record Sub(ASTNode left, ASTNode right) implements ASTNode {}
    record Mul(ASTNode left, ASTNode right) implements ASTNode {}
    record Div(ASTNode left, ASTNode right) implements ASTNode {}
    record Fma(ASTNode a, ASTNode b, ASTNode c) implements ASTNode {}
}